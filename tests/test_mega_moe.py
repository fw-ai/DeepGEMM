import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import (
    per_token_cast_to_fp4, per_token_cast_to_fp8,
    cast_back_from_fp4, unpack_ue8m0_from_int,
)
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


def _apply_gate_activation(gate: torch.Tensor, activation: str) -> torch.Tensor:
    # Both variants reduce to `gate * sigmoid(z)`:
    #  - SwiGLU: SiLU(gate)        => z = gate
    #  - GeGLU:  tanh-approx GELU  => z = alpha * (gate + beta * gate ** 3),
    #    since 0.5 * (1 + tanh(t)) == sigmoid(2 * t)
    if activation == 'swiglu':
        z = gate
    elif activation == 'geglu':
        alpha = 1.5957691216057308  # 2 * sqrt(2 / pi)
        beta = 0.044715
        z = alpha * (gate + beta * gate * gate * gate)
    else:
        raise ValueError(f'Unsupported activation: {activation}')
    return gate * torch.sigmoid(z)


def _dequant_x_fp8(x_fp8: torch.Tensor, sf_packed: torch.Tensor, gran_k: int = 32) -> torch.Tensor:
    # FP8 (E4M3) activations with packed UE8M0 per-`gran_k` scale factors
    m, n = x_fp8.shape
    sf = unpack_ue8m0_from_int(sf_packed)[:, :n // gran_k]
    return (x_fp8.float().view(m, n // gran_k, gran_k) * sf.unsqueeze(2)).view(m, n)


def _dequant_weight_fp4(w_bf16: torch.Tensor, gran_k: int = 32) -> torch.Tensor:
    # Emulate the exact FP4 (E2M1) values the kernel consumes by round-tripping
    # each expert's weight through the same quantizer used to build the inputs.
    num_experts = w_bf16.size(0)
    out = torch.empty_like(w_bf16, dtype=torch.float32)
    for e in range(num_experts):
        packed, sf = per_token_cast_to_fp4(w_bf16[e], use_ue8m0=True, gran_k=gran_k)
        out[e] = cast_back_from_fp4(packed, sf, gran_k=gran_k)
    return out


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    is_bf16xbf16 = args.mma_type == 'bf16xbf16'
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = (
        0 if args.zero_tokens or args.zero_rank == rank_idx else
        max(
            0,
            args.num_max_tokens_per_rank -
            random.randint(0, args.num_max_removed_tokens))
        if args.num_tokens == 0 else args.num_tokens
    )
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type=args.mma_type,
        activation=args.activation
    )

    # Cast weights into FP4
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global l1_weights_bf16, l2_weights_bf16
        global saved_l1_preact
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        # Keep BF16 originals for the self-contained numerical reference
        l1_weights_bf16, l2_weights_bf16 = l1_weights, l2_weights
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        if args.routing == 'balanced':
            token_ids = torch.arange(
                num_tokens, device='cuda', dtype=torch.long).unsqueeze(1)
            slots = torch.arange(
                num_topk, device='cuda', dtype=torch.long).unsqueeze(0)
            topk_idx = (token_ids * num_topk + slots) % num_experts
        elif args.routing == 'skew':
            token_ids = torch.arange(
                num_tokens, device='cuda', dtype=torch.long).unsqueeze(1)
            slots = torch.arange(
                num_topk, device='cuda', dtype=torch.long).unsqueeze(0)
            tail = token_ids >= max(1, (num_tokens * 7) // 8)
            topk_idx = torch.where(
                tail,
                (token_ids * num_topk + slots) % num_experts,
                slots % num_experts)
        elif args.routing == 'extreme':
            assert num_topk <= num_experts
            topk_idx = torch.arange(
                num_topk, device='cuda', dtype=torch.long
            ).expand(num_tokens, -1).clone()
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        if not is_bf16xbf16:
            # FP8 path: cast inputs to FP8/FP4 with per-32 UE8M0 SF
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0
            x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
            l1_weights = _cast_weights_to_fp4(l1_weights)
            l2_weights = _cast_weights_to_fp4(l2_weights)

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                l1_weights, l2_weights, activation=args.activation))
        saved_l1_preact = None
        if is_bf16xbf16 and (
            args.save_l1_preact or args.test_backward
        ):
            saved_l1_preact = torch.full(
                (buffer.token_src_metadata.size(0), 2 * intermediate_hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused():
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            saved_l1_preact=saved_l1_preact)
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    # Self-contained PyTorch reference. Distributed routing/combine is modeled
    # for BF16; the quantized reference remains single-rank.
    def gather_expert_weights(local_weights: torch.Tensor) -> torch.Tensor:
        if num_ranks == 1:
            return local_weights
        gathered = [torch.empty_like(local_weights) for _ in range(num_ranks)]
        dist.all_gather(gathered, local_weights.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    def gather_rank_padded(
        local_tensor: torch.Tensor, fill_value: float
    ) -> torch.Tensor:
        padded_shape = (
            buffer.num_max_tokens_per_rank, *local_tensor.shape[1:])
        padded = torch.full(
            padded_shape, fill_value,
            dtype=local_tensor.dtype, device=local_tensor.device)
        padded[:local_tensor.size(0)].copy_(local_tensor)
        if num_ranks == 1:
            return padded.unsqueeze(0)
        gathered = [torch.empty_like(padded) for _ in range(num_ranks)]
        dist.all_gather(gathered, padded, group=group)
        return torch.stack(gathered)

    def run_reference():
        assert is_bf16xbf16 or num_ranks == 1, (
            'Distributed numerical reference only supports BF16')
        clamp = float(args.activation_clamp)
        if is_bf16xbf16:
            x_deq = x[:num_tokens]
            l1_w = gather_expert_weights(l1_weights_bf16)
            l2_w = gather_expert_weights(l2_weights_bf16)
            num_reference_experts = num_experts
        else:
            x_deq = _dequant_x_fp8(
                x[0][:num_tokens], x[1][:num_tokens])
            l1_w = _dequant_weight_fp4(l1_weights_bf16)
            l2_w = _dequant_weight_fp4(l2_weights_bf16)
            num_reference_experts = num_experts_per_rank

        y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device='cuda')
        for slot in range(num_topk):
            expert_idx = topk_idx[:num_tokens, slot]
            weight = topk_weights[:num_tokens, slot].float()
            for e in range(num_reference_experts):
                mask = expert_idx == e
                if not bool(mask.any()):
                    continue
                xt = x_deq[mask]

                # L1 GEMM then split into gate/up (first/second half of the output)
                acc1 = xt @ l1_w[e].t()
                gate = acc1[:, :intermediate_hidden].to(torch.bfloat16)
                up = acc1[:, intermediate_hidden:].to(torch.bfloat16)
                if clamp != float('inf'):
                    gate = torch.clamp(gate, max=clamp)
                    up = torch.clamp(up, min=-clamp, max=clamp)

                # Gated activation with the per-token routing weight folded in
                act = _apply_gate_activation(gate.float(), args.activation) * up.float()
                act = act * weight[mask].unsqueeze(1)

                if is_bf16xbf16:
                    # Native BF16 contract: round the activation output before
                    # W2, then round W2 before the FP32 top-k reduction.
                    act_deq = act.to(torch.bfloat16)
                    l2_out = act_deq @ l2_w[e].t()
                else:
                    # Requantize to FP8 (per-32 UE8M0), matching the
                    # quantized kernel's L1 output.
                    act_fp8, act_sf = per_token_cast_to_fp8(
                        act, use_ue8m0=True, gran_k=32)
                    n_groups = intermediate_hidden // 32
                    act_deq = (
                        act_fp8.float().view(-1, n_groups, 32) *
                        act_sf[:, :n_groups].unsqueeze(2)
                    ).view(-1, intermediate_hidden)
                    l2_out = act_deq @ l2_w[e].t()

                # L2 GEMM, accumulate across the top-k experts
                y[mask] += l2_out.float()
        return y.to(torch.bfloat16)

    def check_bf16_parity(fused_y: torch.Tensor, ref_y: torch.Tensor):
        assert torch.isfinite(fused_y.float()).all()
        assert torch.isfinite(ref_y.float()).all()
        if fused_y.numel() > 0 and torch.count_nonzero(ref_y):
            cosine = torch.nn.functional.cosine_similarity(
                fused_y.float().flatten(),
                ref_y.float().flatten(),
                dim=0).item()
            assert cosine >= 0.999999, f'BF16 cosine too small: {cosine}'
        torch.testing.assert_close(
            fused_y, ref_y, rtol=2 ** -7, atol=2 ** -7)
        if num_ranks == 1 and num_tokens == num_topk == num_experts == 1:
            assert torch.equal(
                fused_y, ref_y), 'deterministic 1x1 route must be bitwise equal'

        expected_stats = cumulative_local_expert_recv_stats_baseline.clone()
        all_topk_idx = gather_rank_padded(topk_idx, -1)
        local_expert_start = rank_idx * num_experts_per_rank
        valid_experts = all_topk_idx[
            (all_topk_idx >= local_expert_start) &
            (all_topk_idx < local_expert_start + num_experts_per_rank)
        ] - local_expert_start
        if valid_experts.numel():
            expected_stats += torch.bincount(
                valid_experts, minlength=num_experts_per_rank).to(torch.int)
        assert torch.equal(
            cumulative_local_expert_recv_stats_fused, expected_stats)

        if saved_l1_preact is None:
            return

        finite = torch.isfinite(saved_l1_preact.float())
        valid_rows = finite.all(dim=1)
        assert torch.equal(valid_rows, finite.any(dim=1)), (
            'saved pre-clamp rows must be either complete or untouched')
        assert valid_rows.sum().item() == valid_experts.numel()
        metadata = buffer.token_src_metadata[valid_rows].long()
        all_x = gather_rank_padded(x, 0)
        global_l1_weights = gather_expert_weights(l1_weights_bf16)
        if metadata.numel() == 0:
            return
        assert (
            (metadata[:, 0] >= 0) &
            (metadata[:, 0] < num_ranks)
        ).all()
        source_ranks = metadata[:, 0]
        token_ids, topk_slots = metadata[:, 1], metadata[:, 2]
        expert_ids = all_topk_idx[source_ranks, token_ids, topk_slots]
        expected_preact = torch.empty(
            (metadata.size(0), 2 * intermediate_hidden),
            dtype=torch.bfloat16, device='cuda')
        for expert_idx in range(num_experts):
            mask = expert_ids == expert_idx
            if bool(mask.any()):
                expected_preact[mask] = (
                    all_x[source_ranks[mask], token_ids[mask]] @
                    global_l1_weights[expert_idx].t())
        actual_preact = saved_l1_preact[valid_rows]
        assert torch.isfinite(actual_preact.float()).all()
        preact_cosine = torch.nn.functional.cosine_similarity(
            actual_preact.float().flatten(),
            expected_preact.float().flatten(),
            dim=0).item()
        assert preact_cosine >= 0.999999, (
            f'pre-clamp cosine too small: {preact_cosine}')
        torch.testing.assert_close(
            actual_preact, expected_preact,
            rtol=2 ** -7, atol=2 ** -7)

    def run_bf16_backward_test():
        expected_tokens_per_expert = (
            num_tokens * num_ranks * num_topk / num_experts)
        if expected_tokens_per_expert <= 8.5:
            block_m = 16
        elif expected_tokens_per_expert <= 16.5:
            block_m = 32
        elif expected_tokens_per_expert <= 32.5:
            block_m = 64
        elif expected_tokens_per_expert <= 64.5:
            block_m = 96
        elif expected_tokens_per_expert <= 96.5:
            block_m = 128
        else:
            block_m = 192

        all_topk_idx = gather_rank_padded(topk_idx, -1)
        all_topk_weights = gather_rank_padded(topk_weights, 0)
        local_expert_start = rank_idx * num_experts_per_rank
        local_route_mask = (
            (all_topk_idx >= local_expert_start) &
            (all_topk_idx <
             local_expert_start + num_experts_per_rank))
        local_expert_ids = (
            all_topk_idx[local_route_mask] - local_expert_start)
        expert_counts = torch.bincount(
            local_expert_ids,
            minlength=num_experts_per_rank).to(torch.int)
        padded_expert_counts = (
            (expert_counts + block_m - 1) // block_m * block_m)

        pool_rows = saved_l1_preact.size(0)
        grad_y = torch.randn(
            (num_tokens, hidden),
            dtype=torch.bfloat16, device='cuda')
        all_grad_y = gather_rank_padded(grad_y, 0)
        all_x = gather_rank_padded(x, 0)

        grad_ye = torch.zeros(
            (pool_rows, hidden),
            dtype=torch.bfloat16, device='cuda')
        grad_h = torch.zeros(
            (pool_rows, intermediate_hidden),
            dtype=torch.bfloat16, device='cuda')
        grad_gate_up = torch.zeros(
            (pool_rows, 2 * intermediate_hidden),
            dtype=torch.bfloat16, device='cuda')
        h_act = torch.zeros_like(grad_h)
        h_weighted = torch.zeros_like(grad_h)
        x_pool = torch.zeros_like(grad_ye)
        grad_x_pool = torch.zeros_like(grad_ye)
        route_weights_pool = torch.zeros(
            pool_rows, dtype=torch.float, device='cuda')
        grad_route_pool = torch.zeros_like(route_weights_pool)
        num_grid_states = (
            num_experts_per_rank *
            ((hidden // 64) *
             (intermediate_hidden // 128) +
             ((2 * intermediate_hidden) // 64) *
             (hidden // 128)) +
            2)
        grid_sync_counter = torch.zeros(
            num_grid_states, dtype=torch.int, device='cuda')

        deep_gemm.bf16_mega_moe_backward_dgrad(
            gate_up_output=saved_l1_preact,
            grad_h_output=grad_h,
            grad_gate_up_output=grad_gate_up,
            h_act_output=h_act,
            h_weighted_output=h_weighted,
            x_pool_output=x_pool,
            grad_x_pool_output=grad_x_pool,
            grad_route_output=grad_route_pool,
            grad_ye=grad_ye,
            route_weights=route_weights_pool,
            w2_weights=l2_weights_bf16,
            w13_weights=l1_weights_bf16,
            expert_counts=expert_counts,
            grid_sync_counter=grid_sync_counter,
            grad_y=grad_y,
            sym_buffer=buffer,
            activation_limit=float(args.activation_clamp),
            block_m=block_m,
            activation=args.activation,
            fast_math=bool(args.fast_math),
            direct_remote_grad_x=num_ranks > 1,
            write_grad_x_pool=True,
            clear_wgrad_padding=True)
        valid_rows = torch.isfinite(
            saved_l1_preact.float()).all(dim=1)
        metadata = buffer.token_src_metadata[valid_rows].long()
        pool_indices = valid_rows.nonzero().flatten()
        source_ranks = metadata[:, 0]
        source_tokens = metadata[:, 1]
        source_slots = metadata[:, 2]
        global_experts = all_topk_idx[
            source_ranks, source_tokens, source_slots]
        local_experts = global_experts - local_expert_start

        ref_grad_ye = torch.zeros_like(grad_ye)
        ref_grad_h = torch.zeros_like(grad_h)
        ref_grad_gate_up = torch.zeros_like(grad_gate_up)
        ref_h_act = torch.zeros_like(h_act)
        ref_h_weighted = torch.zeros_like(h_weighted)
        ref_x_pool = torch.zeros_like(x_pool)
        ref_grad_x_pool = torch.zeros_like(grad_x_pool)
        ref_route_weights = torch.zeros_like(route_weights_pool)
        ref_grad_route = torch.zeros_like(grad_route_pool)

        for expert_idx in range(num_experts_per_rank):
            selected = local_experts == expert_idx
            if not bool(selected.any()):
                continue
            rows = pool_indices[selected]
            source_rank = source_ranks[selected]
            source_token = source_tokens[selected]
            source_slot = source_slots[selected]
            xe = all_x[source_rank, source_token]
            gye = all_grad_y[source_rank, source_token]
            route = all_topk_weights[
                source_rank, source_token, source_slot]
            preact = saved_l1_preact[rows]
            gate = preact[:, :intermediate_hidden]
            up = preact[:, intermediate_hidden:]
            clamp = float(args.activation_clamp)
            gate_clamped = torch.clamp(gate, max=clamp)
            up_clamped = torch.clamp(
                up, min=-clamp, max=clamp)
            gate_fp32 = gate_clamped.float()
            up_fp32 = up_clamped.float()
            if args.activation == 'geglu':
                alpha = 1.5957691216057308
                beta = 0.044715
                z = alpha * (
                    gate_fp32 +
                    beta * gate_fp32 ** 3)
                dz = alpha * (
                    1 + 3 * beta * gate_fp32 ** 2)
            else:
                z = gate_fp32
                dz = torch.ones_like(gate_fp32)
            sig = torch.sigmoid(z)
            activated_gate = gate_fp32 * sig
            h = (activated_gate * up_fp32).to(
                torch.bfloat16)
            hw = (
                h.float() * route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            gh_unweighted = (
                gye @ l2_weights_bf16[expert_idx]
            )
            gh = (
                gh_unweighted.float() *
                route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            activation_grad = (
                sig +
                gate_fp32 * sig * (1 - sig) * dz)
            grad_gate = (
                gh.float() * up_fp32 *
                activation_grad)
            grad_gate = torch.where(
                gate.float() <= clamp,
                grad_gate,
                torch.zeros_like(grad_gate))
            grad_up = (
                gh.float() * activated_gate)
            grad_up = torch.where(
                (up.float() >= -clamp) &
                (up.float() <= clamp),
                grad_up,
                torch.zeros_like(grad_up))
            grad_gu = torch.cat(
                [grad_gate, grad_up], dim=1
            ).to(torch.bfloat16)
            grad_xe = (
                grad_gu @ l1_weights_bf16[expert_idx]
            )

            ref_grad_ye[rows] = gye
            ref_grad_h[rows] = gh_unweighted
            ref_grad_gate_up[rows] = grad_gu
            ref_h_act[rows] = h
            ref_h_weighted[rows] = hw
            ref_x_pool[rows] = xe
            ref_grad_x_pool[rows] = grad_xe
            ref_route_weights[rows] = route
            ref_grad_route[rows] = (
                gh_unweighted.float() * h.float()
            ).sum(dim=1)

        def assert_gradient_close(
            actual: torch.Tensor,
            expected: torch.Tensor,
            name: str,
        ):
            assert torch.isfinite(actual.float()).all(), name
            if (
                num_ranks == 1 and
                num_tokens == 1 and
                num_experts == 1 and
                num_topk == 1
            ):
                assert torch.equal(actual, expected), (
                    f'{name} must be bitwise equal in the '
                    'single-token/single-expert case')
                return
            if torch.count_nonzero(expected):
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(),
                    expected.float().flatten(),
                    dim=0).item()
                assert cosine >= 0.999999, (
                    f'{name} cosine too small: {cosine}')
            # Allow two BF16 ULPs for valid FP32 tensor-core reduction-order
            # differences before the required BF16 output rounding.
            torch.testing.assert_close(
                actual, expected,
                rtol=2 ** -6, atol=2 ** -6,
                msg=lambda msg: f'{name}: {msg}')

        assert_gradient_close(grad_ye, ref_grad_ye, 'grad_ye')
        assert_gradient_close(grad_h, ref_grad_h, 'grad_h')
        assert_gradient_close(
            grad_gate_up, ref_grad_gate_up,
            'grad_gate_up')
        assert_gradient_close(h_act, ref_h_act, 'h_act')
        assert_gradient_close(
            h_weighted, ref_h_weighted,
            'h_weighted')
        assert_gradient_close(x_pool, ref_x_pool, 'x_pool')
        assert_gradient_close(
            grad_x_pool, ref_grad_x_pool,
            'grad_x_pool')
        assert_gradient_close(
            route_weights_pool, ref_route_weights,
            'route_weights_pool')
        assert_gradient_close(
            grad_route_pool, ref_grad_route,
            'grad_route_pool')

        grad_w2 = torch.zeros_like(l2_weights_bf16)
        grad_w13 = torch.zeros_like(l1_weights_bf16)
        grad_x_combined = None
        if num_ranks > 1:
            grad_x_combined = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.bfloat16, device='cuda')
            deep_gemm.bf16_mega_moe_backward_w2_combine(
                grad_w2, grad_ye, h_weighted,
                padded_expert_counts, grad_x_combined,
                buffer)
            deep_gemm.bf16_mega_moe_backward_w13_combine(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts, grad_x_combined,
                buffer)
        else:
            deep_gemm.bf16_mega_moe_backward_w2(
                grad_w2, grad_ye, h_weighted,
                padded_expert_counts)
            deep_gemm.bf16_mega_moe_backward_w13(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts)
        ref_grad_w2 = torch.zeros_like(grad_w2)
        ref_grad_w13 = torch.zeros_like(grad_w13)
        pool_offset = 0
        for expert_idx in range(num_experts_per_rank):
            count = int(expert_counts[expert_idx].item())
            rows = slice(pool_offset, pool_offset + count)
            if count:
                ref_grad_w2[expert_idx] = (
                    ref_grad_ye[rows].t() @
                    ref_h_weighted[rows])
                ref_grad_w13[expert_idx] = (
                    ref_grad_gate_up[rows].t() @
                    ref_x_pool[rows])
            pool_offset += int(
                padded_expert_counts[expert_idx].item())
        assert_gradient_close(
            grad_w2, ref_grad_w2, 'grad_w2')
        assert_gradient_close(
            grad_w13, ref_grad_w13, 'grad_w13')

        if num_ranks > 1:
            ref_grad_x_planes = torch.zeros(
                (num_ranks, num_max_tokens_per_rank,
                 num_topk, hidden),
                dtype=torch.bfloat16, device='cuda')
            actual_route_planes = torch.zeros(
                (num_ranks, num_max_tokens_per_rank,
                 num_topk),
                dtype=torch.float, device='cuda')
            ref_route_planes = torch.zeros_like(
                actual_route_planes)
            ref_grad_x_planes[
                source_ranks, source_tokens, source_slots
            ] = ref_grad_x_pool[pool_indices]
            actual_route_planes[
                source_ranks, source_tokens, source_slots
            ] = grad_route_pool[pool_indices]
            ref_route_planes[
                source_ranks, source_tokens, source_slots
            ] = ref_grad_route[pool_indices]
            dist.all_reduce(ref_grad_x_planes, group=group)
            dist.all_reduce(actual_route_planes, group=group)
            dist.all_reduce(ref_route_planes, group=group)
            ref_grad_x = ref_grad_x_planes[
                rank_idx, :num_tokens
            ].float().sum(dim=1).to(torch.bfloat16)
            assert_gradient_close(
                grad_x_combined, ref_grad_x,
                'grad_x_combined')
            assert_gradient_close(
                actual_route_planes[rank_idx, :num_tokens],
                ref_route_planes[rank_idx, :num_tokens],
                'grad_route_source')

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    try:
        ep_buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
            num_topk=num_topk, use_fp8_dispatch=True,
            explicitly_destroy=True,
            allow_multiple_reduction=False,
            num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
        ) if is_legacy_loaded else None
    except Exception as ex:
        dist_print(f'Failed to create legacy EP buffer: {ex}, skip baseline', once_in_node=True)
        ep_buffer, is_legacy_loaded = None, False

    # Baseline params differ by mma type
    run_baseline = None
    if is_legacy_loaded:
        if is_bf16xbf16:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False, 'do_expand': True}
            gemm_fn = deep_gemm.m_grouped_bf16_gemm_nt_contiguous
            gemm_kwargs = {'compiled_dims': '', 'use_psum_layout': True}
            swiglu_kwargs = {'round_scale': False, 'ue8m0_scale': False, 'output_bf16': True}
            get_num_tokens = lambda recv_x: recv_x.size(0)
        else:
            dispatch_kwargs = {'do_cpu_sync': False, 'do_handle_copy': False,
                               'do_expand': True, 'use_tma_aligned_col_major_sf': True}
            gemm_fn = deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous
            gemm_kwargs = {'use_psum_layout': True, 'recipe': (1, 1, 32)}
            swiglu_kwargs = {'round_scale': True, 'ue8m0_scale': True, 'output_bf16': False}
            get_num_tokens = lambda recv_x: recv_x[0].size(0)

        def run_baseline():
            # Dispatch
            recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
                cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
                num_experts=num_experts, expert_alignment=alignment,
                **dispatch_kwargs)
            num_recv_tokens = get_num_tokens(recv_x)

            # L1 GEMM
            l1_y = torch.empty((num_recv_tokens, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
            gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # SwiGLU
            swiglu_result = tilelang_ops.swiglu_apply_weight_to_fp8(
                x=l1_y, topk_weights=recv_topk_weights,
                avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
                num_per_channels=32, use_col_major_scales=True,
                clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                **swiglu_kwargs)
            l1_y = swiglu_result[-1] if is_bf16xbf16 else swiglu_result

            # L2 GEMM
            l2_y = torch.empty((num_recv_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, **gemm_kwargs)

            # Combine
            return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # The legacy tilelang baseline is bitwise-identical but only implements SwiGLU
    use_legacy_baseline = is_legacy_loaded and args.activation == 'swiglu'
    # The self-contained reference models the native BF16 boundaries and the
    # quantized FP8/FP4 boundaries, but only local routing/combine.
    use_numerical_reference = is_bf16xbf16 or num_ranks == 1
    ran_correctness = False

    # noinspection PyBroadException
    if use_legacy_baseline and num_correctness_tests > 0:
        dist_print('Running correctness tests (bitwise vs legacy baseline):', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            for fused_result, baseline_result in zip(run_fused(), run_baseline()):
                assert torch.equal(fused_result, baseline_result)
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
        ran_correctness = True

    if use_numerical_reference and num_correctness_tests > 0:
        # FP8/FP4 quantization across two GEMMs bounds the achievable similarity
        max_diff = 0.05
        dist_print(f'Running correctness tests ({args.activation}, numerical reference):', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, _ = run_fused()
            ref_y = run_reference()
            if is_bf16xbf16:
                check_bf16_parity(fused_y, ref_y)
                if args.test_backward:
                    run_bf16_backward_test()
                diff = calc_diff(fused_y, ref_y)
            else:
                diff = calc_diff(fused_y, ref_y)
                assert diff < max_diff, f'{args.activation} diff too large: {diff:.5f} >= {max_diff}'
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed (diff {diff:.5f})', once_in_node=True)
        dist_print(once_in_node=True)
        ran_correctness = True

    if not ran_correctness:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size, weight_elem_size = (2, 2) if is_bf16xbf16 else (1, 0.5)
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * weight_elem_size   # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * weight_elem_size     # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                  # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                     # L2 acts read
        + num_recv_tokens * hidden * 2                                              # L2 output write (always BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=8192, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--zero-tokens', action='store_true', help='Launch a rank with exactly zero tokens')
    parser.add_argument('--zero-rank', type=int, default=-1, help='Rank that launches with zero tokens')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation', type=str, default='swiglu', choices=['swiglu', 'geglu'], help='Gated activation type')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--routing', choices=['random', 'balanced', 'skew', 'extreme'], default='random')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', help='MMA type: fp8xfp4 or bf16xbf16')
    parser.add_argument('--save-l1-preact', action='store_true', help='Validate optional exact BF16 W13 output')
    parser.add_argument('--test-backward', action='store_true', help='Validate BF16 dgrad and grouped wgrad')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
