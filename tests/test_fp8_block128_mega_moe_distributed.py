"""Two-rank distributed parity/adversarial coverage for FP8-block128 MegaMoE."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _weight_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups, rows, columns = weight.shape
    blocks = (
        weight.float()
        .view(groups, rows // 128, 128, columns // 128, 128)
        .permute(0, 1, 3, 2, 4)
    )
    scales = blocks.abs().amax(dim=(-1, -2)) / 448.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (
        (blocks / scales[..., None, None])
        .to(torch.float8_e4m3fn)
        .permute(0, 1, 3, 2, 4)
        .reshape_as(weight)
        .contiguous()
    )
    return quantized, scales.contiguous()


def _weight_dequantize(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    groups, rows, columns = quantized.shape
    return (
        quantized.float()
        .view(groups, rows // 128, 128, columns // 128, 128)
        .permute(0, 1, 3, 2, 4)
        .mul(scales[..., None, None])
        .permute(0, 1, 3, 2, 4)
        .reshape(groups, rows, columns)
    )


def _rank_inputs(rank: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    tokens, topk = 5 + rank * 2, 2
    generator = torch.Generator(device=device).manual_seed(9000 + rank)
    x = (
        torch.randn(
            tokens, 256, generator=generator, device=device, dtype=torch.bfloat16
        )
        * 0.1
    ).requires_grad_()
    token_index = torch.arange(tokens, device=device, dtype=torch.int64)
    # Every rank sends remotely, expert 2 receives skewed traffic, and expert 3
    # is globally empty.  The unequal token counts stress asymmetric splits.
    first = torch.remainder(token_index + rank, 2)
    second = torch.full_like(first, 2)
    ids = torch.stack((first, second), dim=-1).contiguous()
    raw_scores = torch.sigmoid(
        torch.randn(
            tokens, topk, generator=generator, device=device, dtype=torch.float32
        )
    )
    scores = (
        (raw_scores / raw_scores.sum(dim=-1, keepdim=True) * 2.5)
        .detach()
        .requires_grad_()
    )
    upstream = torch.randn(
        tokens, 256, generator=generator, device=device, dtype=torch.bfloat16
    )
    return x, ids, scores.contiguous(), upstream


def _reference(
    deep_gemm,
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
    full_w13_q: torch.Tensor,
    full_w13_s: torch.Tensor,
    full_w2_q: torch.Tensor,
    full_w2_s: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    x_ref = x.detach().clone()
    score_ref = scores.detach().clone()
    x_q, x_s = deep_gemm._C.sm103_fp8_block128_quantize(x_ref)
    x_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(x_q, x_s).float()
    w13 = _weight_dequantize(full_w13_q, full_w13_s)
    w2 = _weight_dequantize(full_w2_q, full_w2_s)
    flat_ids = ids.flatten()
    route_x = x_dequantized.repeat_interleave(ids.shape[1], dim=0)
    preactivation = (
        torch.bmm(w13.index_select(0, flat_ids), route_x.unsqueeze(-1))
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    up, gate = preactivation.float().chunk(2, dim=-1)
    hidden_q, hidden_s = deep_gemm._C.sm103_fp8_block128_swiglu_quantize(
        preactivation.detach()
    )
    hidden_dequantized_bf16 = deep_gemm._C.sm103_fp8_block128_dequantize(
        hidden_q, hidden_s
    )
    route_output = (
        torch.bmm(
            w2.index_select(0, flat_ids), hidden_dequantized_bf16.float().unsqueeze(-1)
        )
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    tokens, topk = ids.shape
    output_float = torch.zeros(tokens, 256, device=x.device, dtype=torch.float32)
    route_view = route_output.view(tokens, topk, 256)
    for route in range(topk):
        output_float = (
            output_float + route_view[:, route].float() * score_ref[:, route, None]
        )
    output = output_float.to(torch.bfloat16)

    upstream_routes = upstream.repeat_interleave(topk, dim=0).float()
    grad_score = (route_output.float() * upstream_routes).sum(dim=-1).view(tokens, topk)
    grad_route_float = upstream_routes * score_ref.flatten()[:, None]
    grad_route_blocks = grad_route_float.view(
        grad_route_float.shape[0], grad_route_float.shape[1] // 128, 128
    )
    grad_route_scales = grad_route_blocks.abs().amax(dim=-1) / 448.0
    grad_route_scales = torch.where(
        grad_route_scales == 0,
        torch.ones_like(grad_route_scales),
        grad_route_scales,
    )
    grad_route_q = (grad_route_blocks / grad_route_scales[..., None]).to(
        torch.float8_e4m3fn
    )
    grad_route = (
        (grad_route_q.float() * grad_route_scales[..., None])
        .reshape_as(grad_route_float)
        .to(torch.bfloat16)
    )
    grad_hidden = (
        torch.bmm(
            w2.index_select(0, flat_ids).transpose(1, 2),
            grad_route.float().unsqueeze(-1),
        )
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    sigmoid_gate = torch.sigmoid(gate)
    grad_gate = (
        grad_hidden.float() * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    ).to(torch.bfloat16)
    grad_up = (grad_hidden.float() * gate * sigmoid_gate).to(torch.bfloat16)
    grad_preactivation = torch.cat((grad_gate, grad_up), dim=-1)
    grad_pre_q, grad_pre_s = deep_gemm._C.sm103_fp8_block128_quantize(
        grad_preactivation
    )
    grad_pre_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(
        grad_pre_q, grad_pre_s
    ).float()
    canonical_w13 = torch.cat(
        (w13[:, w13.shape[1] // 2 :], w13[:, : w13.shape[1] // 2]), dim=1
    )
    grad_input_routes = (
        torch.bmm(
            canonical_w13.index_select(0, flat_ids).transpose(1, 2),
            grad_pre_dequantized.unsqueeze(-1),
        )
        .squeeze(-1)
        .to(torch.bfloat16)
    )
    grad_input_float = torch.zeros_like(x_ref, dtype=torch.float32)
    grad_input_view = grad_input_routes.view(tokens, topk, x_ref.shape[1])
    for route in range(topk):
        grad_input_float += grad_input_view[:, route].float()

    experts, hidden, model_dim = full_w2_q.shape[0], full_w2_q.shape[2], x.shape[1]
    grad_w1 = torch.zeros(
        experts, hidden, model_dim, device=x.device, dtype=torch.float32
    )
    grad_w2 = torch.zeros_like(w2, dtype=torch.float32)
    grad_w3 = torch.zeros_like(grad_w1)
    route_x_bf16 = route_x.to(torch.bfloat16)
    for expert in range(experts):
        mask = flat_ids == expert
        if not mask.any():
            continue
        grad_w1[expert] = (
            grad_gate[mask].float().transpose(0, 1) @ route_x_bf16[mask].float()
        )
        grad_w2[expert] = (
            grad_route[mask].float().transpose(0, 1)
            @ hidden_dequantized_bf16[mask].float()
        )
        grad_w3[expert] = (
            grad_up[mask].float().transpose(0, 1) @ route_x_bf16[mask].float()
        )
    return (
        output,
        grad_input_float.to(torch.bfloat16),
        grad_score,
        grad_w1,
        grad_w2,
        grad_w3,
    )


def _normalized_difference(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f, expected_f = actual.float().flatten(), expected.float().flatten()
    return float(
        1
        - 2
        * (actual_f @ expected_f)
        / (actual_f.square().sum() + expected_f.square().sum() + 1e-12)
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["DG_JIT_CACHE_DIR"] = f"/tmp/deepgemm-sm103-dist-rank{rank}"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    import torch.distributed as dist

    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        import deep_gemm
        from deep_gemm.mega.fp8_block128 import (
            _configure_fp8_block128_mega_moe_transport,
        )

        assert torch.cuda.get_device_capability(device) == (10, 3)
        _configure_fp8_block128_mega_moe_transport(
            dist.group.WORLD,
            context_tokens_per_rank=64,
        )
        experts, local_experts, model_dim, hidden = 4, 2, 256, 128
        generator = torch.Generator(device=device).manual_seed(8800)
        full_w13_master = (
            torch.randn(
                experts * 2,
                hidden,
                model_dim,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.05
        )
        full_w2_master = (
            torch.randn(
                experts,
                model_dim,
                hidden,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.05
        )
        full_w13_q_canonical, full_w13_s_canonical = _weight_quantize(full_w13_master)
        full_w13_q_active, full_w13_s_active = (
            deep_gemm.transform_glm_w13_for_fp8_block128_mega_moe(
                full_w13_q_canonical, full_w13_s_canonical
            )
        )
        full_w2_q, full_w2_s = _weight_quantize(full_w2_master)
        expert_start = rank * local_experts
        expert_end = expert_start + local_experts
        local_w13_q = full_w13_q_canonical[
            expert_start * 2 : expert_end * 2
        ].contiguous()
        local_w13_s = full_w13_s_canonical[
            expert_start * 2 : expert_end * 2
        ].contiguous()
        local_w2_q = full_w2_q[expert_start:expert_end].contiguous()
        local_w2_s = full_w2_s[expert_start:expert_end].contiguous()
        local_w1_master = (
            full_w13_master[expert_start * 2 : expert_end * 2 : 2]
            .clone()
            .detach()
            .requires_grad_()
        )
        local_w3_master = (
            full_w13_master[expert_start * 2 + 1 : expert_end * 2 : 2]
            .clone()
            .detach()
            .requires_grad_()
        )
        local_w2_master = (
            full_w2_master[expert_start:expert_end].clone().detach().requires_grad_()
        )
        x, ids, scores, upstream = _rank_inputs(rank, device)
        # Reproduce the production lifecycle: a small warmup constructs the
        # first arena, whose fixed target envelope must admit the real batch.
        with torch.no_grad():
            warmup_output = deep_gemm.fp8_block128_mega_moe(
                x[:1].contiguous(),
                ids[:1].contiguous(),
                scores[:1].contiguous(),
                local_w13_q,
                local_w13_s,
                local_w2_q,
                local_w2_s,
                local_w1_master,
                local_w2_master,
                local_w3_master,
                group=dist.group.WORLD,
            )
        assert warmup_output.shape == (1, model_dim)
        (
            expected_output,
            expected_x_grad,
            expected_score_grad,
            expected_w1_grad,
            expected_w2_grad,
            expected_w3_grad,
        ) = _reference(
            deep_gemm,
            x,
            ids,
            scores,
            full_w13_q_active,
            full_w13_s_active,
            full_w2_q,
            full_w2_s,
            upstream,
        )
        # Every sharded expert receives routes from every source rank.  Sum the
        # FP32 reference partials before comparing with the owning rank's one
        # grouped BF16-master wgrad result.
        for expected_weight_grad in (
            expected_w1_grad,
            expected_w2_grad,
            expected_w3_grad,
        ):
            dist.all_reduce(expected_weight_grad, group=dist.group.WORLD)
        expected_w1_grad = expected_w1_grad.to(torch.bfloat16)
        expected_w2_grad = expected_w2_grad.to(torch.bfloat16)
        expected_w3_grad = expected_w3_grad.to(torch.bfloat16)

        output = deep_gemm.fp8_block128_mega_moe(
            x,
            ids,
            scores,
            local_w13_q,
            local_w13_s,
            local_w2_q,
            local_w2_s,
            local_w1_master,
            local_w2_master,
            local_w3_master,
            group=dist.group.WORLD,
        )
        output.backward(upstream)
        torch.testing.assert_close(
            output.float(), expected_output.float(), rtol=0.08, atol=0.08
        )
        torch.testing.assert_close(
            scores.grad, expected_score_grad, rtol=3e-4, atol=3e-3
        )
        assert _normalized_difference(x.grad, expected_x_grad) < 0.12
        weight_differences = {
            "w1": _normalized_difference(
                local_w1_master.grad,
                expected_w1_grad[expert_start:expert_end],
            ),
            "w2": _normalized_difference(
                local_w2_master.grad,
                expected_w2_grad[expert_start:expert_end],
            ),
            "w3": _normalized_difference(
                local_w3_master.grad,
                expected_w3_grad[expert_start:expert_end],
            ),
        }
        assert weight_differences["w1"] < 0.15, weight_differences
        assert weight_differences["w2"] < 0.12, weight_differences
        assert weight_differences["w3"] < 0.15, weight_differences
        for gradient in (
            x.grad,
            local_w1_master.grad,
            local_w2_master.grad,
            local_w3_master.grad,
        ):
            assert gradient is not None
            assert gradient.dtype == torch.bfloat16
            assert torch.isfinite(gradient).all()
        if rank == 1:
            # Global expert 3 has no routes on either source rank.
            assert torch.count_nonzero(local_w1_master.grad[1]) == 0
            assert torch.count_nonzero(local_w3_master.grad[1]) == 0
            assert torch.count_nonzero(local_w2_master.grad[1]) == 0

        direct_output = output.detach()
        direct_x_grad = x.grad.detach().clone()
        direct_score_grad = scores.grad.detach().clone()
        direct_weight_grads = tuple(
            value.grad.detach().clone()
            for value in (local_w1_master, local_w2_master, local_w3_master)
        )
        for value in (local_w1_master, local_w2_master, local_w3_master):
            value.grad = None
        checkpoint_x = x.detach().clone().requires_grad_()
        checkpoint_scores = scores.detach().clone().requires_grad_()

        def checkpointed_megamoe(
            checkpoint_input: torch.Tensor,
            checkpoint_route_scores: torch.Tensor,
        ) -> torch.Tensor:
            return deep_gemm.fp8_block128_mega_moe(
                checkpoint_input,
                ids,
                checkpoint_route_scores,
                local_w13_q,
                local_w13_s,
                local_w2_q,
                local_w2_s,
                local_w1_master,
                local_w2_master,
                local_w3_master,
                group=dist.group.WORLD,
            )

        checkpoint_output = checkpoint(
            checkpointed_megamoe,
            checkpoint_x,
            checkpoint_scores,
            use_reentrant=False,
        )
        checkpoint_output.backward(upstream)
        torch.testing.assert_close(checkpoint_output, direct_output, rtol=0, atol=0)
        torch.testing.assert_close(checkpoint_x.grad, direct_x_grad, rtol=0, atol=0)
        torch.testing.assert_close(
            checkpoint_scores.grad, direct_score_grad, rtol=0, atol=0
        )
        for checkpoint_grad, direct_grad in zip(
            (local_w1_master.grad, local_w2_master.grad, local_w3_master.grad),
            direct_weight_grads,
        ):
            torch.testing.assert_close(checkpoint_grad, direct_grad, rtol=0, atol=0)

        # Exercise the production lifetime where multiple layer forwards reuse
        # one ElasticBuffer before their backwards consume distinct handles.
        # The second routing pattern activates expert 3 instead of expert 2 so
        # stale/overwritten handle metadata cannot accidentally pass.
        second_x = (x.detach() * 0.75).contiguous().requires_grad_()
        second_ids = ids.detach().clone()
        second_ids[:, 0] = 1 - second_ids[:, 0]
        second_ids[:, 1] = 3
        second_scores = scores.detach().flip(1).contiguous().requires_grad_()
        second_upstream = upstream.flip(0).contiguous()
        (
            second_expected_output,
            second_expected_x_grad,
            second_expected_score_grad,
            second_expected_w1_grad,
            second_expected_w2_grad,
            second_expected_w3_grad,
        ) = _reference(
            deep_gemm,
            second_x,
            second_ids,
            second_scores,
            full_w13_q_active,
            full_w13_s_active,
            full_w2_q,
            full_w2_s,
            second_upstream,
        )
        for second_expected_weight_grad in (
            second_expected_w1_grad,
            second_expected_w2_grad,
            second_expected_w3_grad,
        ):
            dist.all_reduce(second_expected_weight_grad, group=dist.group.WORLD)

        for value in (local_w1_master, local_w2_master, local_w3_master):
            value.grad = None
        first_x = x.detach().clone().requires_grad_()
        first_scores = scores.detach().clone().requires_grad_()
        first_output = deep_gemm.fp8_block128_mega_moe(
            first_x,
            ids,
            first_scores,
            local_w13_q,
            local_w13_s,
            local_w2_q,
            local_w2_s,
            local_w1_master,
            local_w2_master,
            local_w3_master,
            group=dist.group.WORLD,
        )
        second_output = deep_gemm.fp8_block128_mega_moe(
            second_x,
            second_ids,
            second_scores,
            local_w13_q,
            local_w13_s,
            local_w2_q,
            local_w2_s,
            local_w1_master,
            local_w2_master,
            local_w3_master,
            group=dist.group.WORLD,
        )
        torch.autograd.backward(
            (first_output, second_output), (upstream, second_upstream)
        )
        torch.testing.assert_close(first_output, direct_output, rtol=0, atol=0)
        torch.testing.assert_close(
            second_output.float(),
            second_expected_output.float(),
            rtol=0.08,
            atol=0.08,
        )
        torch.testing.assert_close(first_x.grad, direct_x_grad, rtol=0, atol=0)
        torch.testing.assert_close(first_scores.grad, direct_score_grad, rtol=0, atol=0)
        assert _normalized_difference(second_x.grad, second_expected_x_grad) < 0.12
        torch.testing.assert_close(
            second_scores.grad,
            second_expected_score_grad,
            rtol=3e-4,
            atol=3e-3,
        )
        expected_combined_weight_grads = (
            expected_w1_grad.float()
            + second_expected_w1_grad.to(torch.bfloat16).float(),
            expected_w2_grad.float()
            + second_expected_w2_grad.to(torch.bfloat16).float(),
            expected_w3_grad.float()
            + second_expected_w3_grad.to(torch.bfloat16).float(),
        )
        combined_weight_differences = {
            name: _normalized_difference(
                actual.grad,
                expected[expert_start:expert_end],
            )
            for name, actual, expected in zip(
                ("w1", "w2", "w3"),
                (local_w1_master, local_w2_master, local_w3_master),
                expected_combined_weight_grads,
            )
        }
        assert combined_weight_differences["w1"] < 0.15, combined_weight_differences
        assert combined_weight_differences["w2"] < 0.12, combined_weight_differences
        assert combined_weight_differences["w3"] < 0.15, combined_weight_differences
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two SM103 GPUs")
def test_two_rank_uneven_cross_rank_forward_backward() -> None:
    if any(torch.cuda.get_device_capability(index) != (10, 3) for index in range(2)):
        pytest.skip("requires two SM103 GPUs")
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
