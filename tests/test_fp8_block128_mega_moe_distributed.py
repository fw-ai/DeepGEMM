"""Two-rank distributed parity/adversarial coverage for FP8-block128 MegaMoE."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp


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
    tokens, topk, experts = 5 + rank * 2, 2, 4
    generator = torch.Generator(device=device).manual_seed(9000 + rank)
    x = (
        torch.randn(tokens, 256, generator=generator, device=device, dtype=torch.bfloat16)
        * 0.1
    ).requires_grad_()
    token_index = torch.arange(tokens, device=device, dtype=torch.int64)
    # Every rank sends remotely, expert 2 receives skewed traffic, and expert 3
    # is globally empty.  The unequal token counts stress asymmetric splits.
    first = torch.remainder(token_index + rank, 2)
    second = torch.full_like(first, 2)
    ids = torch.stack((first, second), dim=-1).contiguous()
    raw_scores = torch.sigmoid(
        torch.randn(tokens, topk, generator=generator, device=device, dtype=torch.float32)
    )
    scores = (raw_scores / raw_scores.sum(dim=-1, keepdim=True) * 2.5).detach().requires_grad_()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ref = x.detach().clone().requires_grad_()
    score_ref = scores.detach().clone().requires_grad_()
    x_q, x_s = deep_gemm._C.sm103_fp8_block128_quantize(x_ref.detach())
    x_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(x_q, x_s).float()
    x_effective = x_ref.float() + (x_dequantized - x_ref.float()).detach()
    w13 = _weight_dequantize(full_w13_q, full_w13_s)
    w2 = _weight_dequantize(full_w2_q, full_w2_s)
    flat_ids = ids.flatten()
    route_x = x_effective.repeat_interleave(ids.shape[1], dim=0)
    preactivation = torch.bmm(
        w13.index_select(0, flat_ids), route_x.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    up, gate = preactivation.float().chunk(2, dim=-1)
    hidden_raw = torch.nn.functional.silu(gate) * up
    hidden_q, hidden_s = deep_gemm._C.sm103_fp8_block128_swiglu_quantize(
        preactivation.detach()
    )
    hidden_dequantized = deep_gemm._C.sm103_fp8_block128_dequantize(
        hidden_q, hidden_s
    ).float()
    hidden_effective = hidden_raw + (hidden_dequantized - hidden_raw).detach()
    route_output = torch.bmm(
        w2.index_select(0, flat_ids), hidden_effective.unsqueeze(-1)
    ).squeeze(-1).to(torch.bfloat16)
    tokens, topk = ids.shape
    output_float = torch.zeros(tokens, 256, device=x.device, dtype=torch.float32)
    route_view = route_output.view(tokens, topk, 256)
    for route in range(topk):
        output_float = output_float + route_view[:, route].float() * score_ref[:, route, None]
    output = output_float.to(torch.bfloat16)
    output.backward(upstream)
    return output.detach(), x_ref.grad.detach(), score_ref.grad.detach()


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

        assert torch.cuda.get_device_capability(device) == (10, 3)
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
        full_w13_q_canonical, full_w13_s_canonical = _weight_quantize(
            full_w13_master
        )
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
            full_w2_master[expert_start:expert_end]
            .clone()
            .detach()
            .requires_grad_()
        )
        x, ids, scores, upstream = _rank_inputs(rank, device)
        expected_output, expected_x_grad, expected_score_grad = _reference(
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
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two SM103 GPUs")
def test_two_rank_uneven_cross_rank_forward_backward() -> None:
    if any(torch.cuda.get_device_capability(index) != (10, 3) for index in range(2)):
        pytest.skip("requires two SM103 GPUs")
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
