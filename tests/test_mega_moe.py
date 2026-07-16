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
        gate_sq = gate * gate
        z = (alpha * gate) * (1.0 + beta * gate_sq)
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
        global saved_l1_preact, saved_h_unweighted
        global saved_h_weighted, saved_down_unweighted
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
        saved_h_unweighted = None
        saved_h_weighted = None
        saved_down_unweighted = None
        if is_bf16xbf16 and (
            args.save_l1_preact or args.test_backward
        ):
            saved_l1_preact = torch.full(
                (buffer.token_src_metadata.size(0), 2 * intermediate_hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')
        if is_bf16xbf16 and (
            args.save_l1_preact or args.test_backward
        ):
            saved_h_unweighted = torch.full(
                (buffer.token_src_metadata.size(0), intermediate_hidden),
                float('nan'), dtype=torch.bfloat16, device='cuda')
            saved_h_weighted = torch.full_like(
                saved_h_unweighted, float('nan'))
        if (
            is_bf16xbf16 and
            (
                args.save_l1_preact or args.test_backward
            )
        ):
            saved_down_unweighted = torch.full(
                (buffer.token_src_metadata.size(0), hidden),
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
        if is_bf16xbf16:
            kernel_kwargs.update(
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode),
                saved_h_unweighted=saved_h_unweighted,
                saved_h_weighted=saved_h_weighted,
                saved_down_unweighted=saved_down_unweighted)
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    # Self-contained PyTorch reference. Distributed routing/combine is modeled
    # for BF16; the quantized reference remains single-rank.
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

    reference_stages = {}
    native_route_layout = {}

    def native_bf16_grouped_mm(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        offsets: torch.Tensor,
        *,
        rhs_needs_transpose: bool = True,
        lhs_needs_contiguous: bool = True,
    ) -> torch.Tensor:
        mat2 = rhs.transpose(-2, -1) if rhs_needs_transpose else rhs
        mat1 = lhs.contiguous() if lhs_needs_contiguous else lhs
        if not offsets.numel() or int(offsets[-1].item()) == 0:
            if lhs_needs_contiguous:
                output_columns = (
                    rhs.size(-2)
                    if rhs_needs_transpose
                    else rhs.size(-1))
                return lhs.new_zeros((0, output_columns))
            return lhs.new_zeros(
                (offsets.numel(), lhs.size(0), rhs.size(1)))
        return torch._grouped_mm(mat1, mat2, offs=offsets)

    def build_native_route_layout(
        all_topk_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_expert_start = rank_idx * num_experts_per_rank
        flat_experts = all_topk_idx.reshape(-1)
        valid = (
            (flat_experts >= local_expert_start) &
            (flat_experts < local_expert_start + num_experts_per_rank))
        flat_positions = valid.nonzero().flatten()
        local_experts = (
            flat_experts[flat_positions] - local_expert_start)
        # FireTitan DeepEP's permutation is expert-major and stable within an
        # expert: source token order first, then top-k slot order.
        order = torch.argsort(local_experts, stable=True)
        flat_positions = flat_positions[order]
        local_experts = local_experts[order]
        routes_per_rank = all_topk_idx.size(1) * num_topk
        source_ranks = flat_positions // routes_per_rank
        rank_positions = flat_positions % routes_per_rank
        source_tokens = rank_positions // num_topk
        source_slots = rank_positions % num_topk
        counts = torch.bincount(
            local_experts, minlength=num_experts_per_rank).to(torch.int32)
        return {
            'local_experts': local_experts,
            'source_ranks': source_ranks,
            'source_tokens': source_tokens,
            'source_slots': source_slots,
            'counts': counts,
            'offsets': counts.cumsum(0).to(torch.int32),
        }

    def native_to_pool_permutation(
        metadata: torch.Tensor,
        layout: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        routes_per_rank = (
            buffer.num_max_tokens_per_rank * num_topk)
        actual_keys = (
            metadata[:, 0] * routes_per_rank +
            metadata[:, 1] * num_topk +
            metadata[:, 2])
        native_keys = (
            layout['source_ranks'] * routes_per_rank +
            layout['source_tokens'] * num_topk +
            layout['source_slots'])
        positions_by_key = torch.full(
            (num_ranks * routes_per_rank,),
            -1, dtype=torch.long, device='cuda')
        positions_by_key[actual_keys] = torch.arange(
            actual_keys.numel(), dtype=torch.long, device='cuda')
        permutation = positions_by_key[native_keys]
        assert (permutation >= 0).all(), (
            'MegaMoE pool is missing a native DeepEP route')
        assert torch.unique(permutation).numel() == permutation.numel(), (
            'MegaMoE pool contains duplicate route metadata')
        return permutation

    def run_native_bf16_reference() -> torch.Tensor:
        all_topk_idx = gather_rank_padded(topk_idx, -1)
        all_topk_weights = gather_rank_padded(topk_weights, 0)
        all_x = gather_rank_padded(x, 0)
        layout = build_native_route_layout(all_topk_idx)
        native_route_layout.clear()
        native_route_layout.update(layout)

        source_ranks = layout['source_ranks']
        source_tokens = layout['source_tokens']
        source_slots = layout['source_slots']
        counts = layout['counts']
        offsets = layout['offsets']
        if source_ranks.numel():
            assert int(source_ranks.min().item()) >= 0
            assert int(source_ranks.max().item()) < all_x.size(0), (
                f'{source_ranks.max().item()=} {all_x.shape=}')
            assert int(source_tokens.min().item()) >= 0
            assert int(source_tokens.max().item()) < all_x.size(1), (
                f'{source_tokens.max().item()=} {all_x.shape=}')
        x_pool = all_x[source_ranks, source_tokens]
        route_weights = all_topk_weights[
            source_ranks, source_tokens, source_slots].float()

        w1_weights = l1_weights_bf16[
            :, :intermediate_hidden].contiguous()
        w3_weights = l1_weights_bf16[
            :, intermediate_hidden:].contiguous()
        gate = native_bf16_grouped_mm(
            x_pool, w1_weights, offsets)
        up = native_bf16_grouped_mm(
            x_pool, w3_weights, offsets)
        w13 = torch.cat((gate, up), dim=1)
        clamp = float(args.activation_clamp)
        gate_clamped = torch.clamp(gate, max=clamp)
        up_clamped = torch.clamp(up, min=-clamp, max=clamp)
        h = (
            _apply_gate_activation(
                gate_clamped.float(), args.activation) *
            up_clamped.float()
        ).to(torch.bfloat16)
        h_weighted = (
            h.float() * route_weights.unsqueeze(1)
        ).to(torch.bfloat16)
        w2_input = (
            h_weighted
            if args.route_weight_mode == 'pre_down'
            else h)
        down = native_bf16_grouped_mm(
            w2_input, l2_weights_bf16, offsets)
        route_output = (
            down
            if args.route_weight_mode == 'pre_down'
            else (
                down.float() * route_weights.unsqueeze(1)
            ).to(torch.bfloat16))

        reference_stages.clear()
        reference_stages.update({
            'w13': w13,
            'h': h,
            'h_weighted': h_weighted,
            'w2_input': w2_input,
            'down': down,
            'route_output': route_output,
            'route_weights': route_weights,
        })

        # Match the native slot reducer: each top-k slot is accumulated in
        # ascending slot order with FP32 accumulation, then rounded to BF16.
        route_planes = torch.zeros(
            (
                num_ranks,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
            ),
            dtype=torch.float32,
            device='cuda')
        route_planes[
            source_ranks, source_tokens, source_slots
        ] = route_output.float()
        if num_ranks > 1:
            dist.all_reduce(route_planes, group=group)
        slot_combined = torch.zeros(
            (num_tokens, hidden), dtype=torch.float32, device='cuda')
        for slot in range(num_topk):
            slot_combined += route_planes[
                rank_idx, :num_tokens, slot]
        slot_combined = slot_combined.to(torch.bfloat16)

        # The non-EP DSV4 bridge accumulates already expert-sorted output in
        # increasing expert-id order. This is intentionally only local: EP
        # uses DeepEP's combine rather than this eager expert loop.
        expert_combined = None
        if num_ranks == 1:
            expert_combined_fp32 = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.float32,
                device='cuda')
            start = 0
            for count in counts.tolist():
                end = start + int(count)
                if end > start:
                    expert_combined_fp32[
                        source_tokens[start:end]
                    ] += route_output[start:end].float()
                start = end
            expert_combined = expert_combined_fp32.to(torch.bfloat16)

        reference_stages['final_slot'] = slot_combined
        reference_stages['final_expert'] = expert_combined
        return slot_combined

    def run_reference():
        assert is_bf16xbf16 or num_ranks == 1, (
            'Distributed numerical reference only supports BF16')
        if is_bf16xbf16:
            return run_native_bf16_reference()
        clamp = float(args.activation_clamp)
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

                # Unweighted gated activation.
                act = (
                    _apply_gate_activation(
                        gate.float(), args.activation) *
                    up.float())

                # Requantize to FP8 (per-32 UE8M0), matching the
                # quantized kernel's L1 output.
                act = act * weight[mask].unsqueeze(1)
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

    def bf16_ulp_distance(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        def ordered_bits(tensor: torch.Tensor) -> torch.Tensor:
            bits = tensor.view(torch.int16).to(torch.int32) & 0xffff
            magnitude = bits & 0x7fff
            return torch.where(
                (bits & 0x8000) != 0,
                0x8000 - magnitude,
                0x8000 + magnitude)

        return (
            ordered_bits(lhs) - ordered_bits(rhs)
        ).abs()

    def report_bf16_parity(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> int:
        assert actual.shape == expected.shape, (
            f'{name}: {actual.shape=} != {expected.shape=}')
        mismatch_count = int((actual != expected).sum().item())
        if actual.numel():
            abs_error = (actual.float() - expected.float()).abs()
            max_abs = float(abs_error.max().item())
            denominator = expected.float().abs().clamp_min(
                torch.finfo(torch.float32).tiny)
            max_relative = float(
                (abs_error / denominator).max().item())
            max_ulp = int(
                bf16_ulp_distance(actual, expected).max().item())
        else:
            max_abs = max_relative = 0.0
            max_ulp = 0
        mismatch_detail = ''
        if mismatch_count:
            mismatch = actual != expected
            mismatch_ulps = bf16_ulp_distance(
                actual[mismatch], expected[mismatch])
            ulps, ulp_counts = torch.unique(
                mismatch_ulps, return_counts=True)
            top_count = min(12, ulp_counts.numel())
            top_indices = torch.topk(
                ulp_counts, top_count).indices
            ulp_histogram = ','.join(
                f'{int(ulp)}:{int(count)}'
                for ulp, count in zip(
                    ulps[top_indices].cpu().tolist(),
                    ulp_counts[top_indices].cpu().tolist()))
            first_flat = int(
                mismatch.flatten().nonzero()[0].item())
            first_actual = float(actual.flatten()[first_flat].float())
            first_expected = float(
                expected.flatten()[first_flat].float())
            mismatch_detail = (
                f', top_ulp_hist={{{ulp_histogram}}}, '
                f'unique_ulps={ulps.numel()}, '
                f'first_flat={first_flat}, '
                f'first_actual={first_actual:.9g}, '
                f'first_expected={first_expected:.9g}')
        dist_print(
            f' > {name}: mismatches={mismatch_count}/'
            f'{actual.numel()}, max_abs={max_abs:.9g}, '
            f'max_rel={max_relative:.9g}, max_ulp={max_ulp}'
            f'{mismatch_detail}',
            once_in_node=True)
        return mismatch_count

    def report_forward_stage_parity(
        fused_y: torch.Tensor,
        ref_y: torch.Tensor,
    ) -> None:
        final_mismatches = report_bf16_parity(
            'Stage final', fused_y, ref_y)
        if reference_stages['final_expert'] is not None:
            report_bf16_parity(
                'Stage final (native expert order)',
                fused_y,
                reference_stages['final_expert'])
        if saved_l1_preact is None:
            assert final_mismatches == 0, (
                'native grouped-MM final output must be bitwise equal')
            return

        valid_rows = torch.isfinite(
            saved_l1_preact.float()).all(dim=1)
        pool_indices = valid_rows.nonzero().flatten()
        metadata = buffer.token_src_metadata[pool_indices].long()
        native_to_pool = native_to_pool_permutation(
            metadata, native_route_layout)
        native_pool_indices = pool_indices[native_to_pool]
        pool_order_mismatches = int(
            (native_to_pool != torch.arange(
                native_to_pool.numel(),
                dtype=torch.long,
                device=native_to_pool.device)).sum().item())
        dist_print(
            f' > Pool-vs-DeepEP order mismatches='
            f'{pool_order_mismatches}/{native_to_pool.numel()}',
            once_in_node=True)
        activation_stage = (
            'h_weighted'
            if args.route_weight_mode == 'pre_down'
            else 'w2_input')
        actual_stages = {
            'w13': saved_l1_preact[native_pool_indices],
            'h': saved_h_unweighted[native_pool_indices],
            activation_stage: (
                saved_h_weighted[native_pool_indices]
                if args.route_weight_mode == 'pre_down'
                else saved_h_unweighted[native_pool_indices]
            ),
            'down': saved_down_unweighted[native_pool_indices],
        }
        first_difference = (
            'final' if final_mismatches else None)
        for stage in ('w13', 'h', activation_stage, 'down'):
            mismatch_count = report_bf16_parity(
                f'Stage {stage}',
                actual_stages[stage],
                reference_stages[stage])
            if stage == 'h' and mismatch_count:
                mismatch = (
                    actual_stages[stage] != reference_stages[stage])
                first_flat = int(
                    mismatch.flatten().nonzero()[0].item())
                route_row = first_flat // intermediate_hidden
                hidden_col = first_flat % intermediate_hidden
                gate = reference_stages['w13'][
                    route_row, hidden_col].float()
                up = reference_stages['w13'][
                    route_row,
                    intermediate_hidden + hidden_col].float()
                alpha = 1.5957691216057308
                beta = 0.044715
                gate_sq = gate * gate
                z = (alpha * gate) * (1.0 + beta * gate_sq)
                explicit = (
                    gate / (1.0 + torch.exp(-z)) * up
                ).to(torch.bfloat16)
                dist_print(
                    f' > First h mismatch inputs: '
                    f'row={route_row}, col={hidden_col}, '
                    f'gate={float(gate):.9g}, up={float(up):.9g}, '
                    f'z={float(z):.9g}, '
                    f'explicit={float(explicit.float()):.9g}',
                    once_in_node=True)
            if stage == 'h_weighted' and mismatch_count:
                mismatch = (
                    actual_stages[stage] != reference_stages[stage])
                first_flat = int(
                    mismatch.flatten().nonzero()[0].item())
                route_row = first_flat // intermediate_hidden
                hidden_col = first_flat % intermediate_hidden
                physical_row = int(
                    native_pool_indices[route_row].item())
                source = buffer.token_src_metadata[
                    physical_row].tolist()
                actual_h = float(
                    actual_stages['h'][
                        route_row, hidden_col].float())
                actual_weighted = float(
                    actual_stages['h_weighted'][
                        route_row, hidden_col].float())
                expected_weighted = float(
                    reference_stages['h_weighted'][
                        route_row, hidden_col].float())
                expected_weight = float(
                    reference_stages['route_weights'][
                        route_row].float())
                observed_weight = (
                    actual_weighted / actual_h
                    if actual_h else float('nan'))
                dist_print(
                    f' > First h_weighted mismatch inputs: '
                    f'native_row={route_row}, '
                    f'physical_row={physical_row}, col={hidden_col}, '
                    f'source={source}, h={actual_h:.9g}, '
                    f'expected_weight={expected_weight:.9g}, '
                    f'observed_weight={observed_weight:.9g}, '
                    f'actual={actual_weighted:.9g}, '
                    f'expected={expected_weighted:.9g}',
                    once_in_node=True)
            if mismatch_count and first_difference is None:
                first_difference = stage
        dist_print(
            f' > First differing stage: {first_difference}',
            once_in_node=True)
        assert first_difference is None, (
            f'first native grouped-MM difference: {first_difference}')

    def check_bf16_parity(fused_y: torch.Tensor, ref_y: torch.Tensor):
        report_forward_stage_parity(fused_y, ref_y)
        assert torch.isfinite(fused_y.float()).all()
        assert torch.isfinite(ref_y.float()).all()

    def check_native_forward_repeatability(
        ref_y: torch.Tensor,
    ) -> None:
        first_stages = dict(reference_stages)
        repeated_y = run_native_bf16_reference()
        mismatch_count = report_bf16_parity(
            'Native repeat final', repeated_y, ref_y)
        assert mismatch_count == 0, (
            'native final output varied across identical runs')
        for stage in (
            'w13', 'h', 'h_weighted', 'w2_input',
            'down', 'route_output'
        ):
            mismatch_count = report_bf16_parity(
                f'Native repeat {stage}',
                reference_stages[stage],
                first_stages[stage])
            assert mismatch_count == 0, (
                f'native {stage} varied across identical runs')

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

        if saved_l1_preact is not None:
            finite = torch.isfinite(saved_l1_preact.float())
            valid_rows = finite.all(dim=1)
            assert torch.equal(valid_rows, finite.any(dim=1)), (
                'saved pre-clamp rows must be either complete or untouched')
            assert valid_rows.sum().item() == valid_experts.numel()

    def run_bf16_backward_test():
        config_num_tokens = torch.tensor(
            num_tokens, dtype=torch.int32, device='cuda')
        if num_ranks > 1:
            dist.all_reduce(
                config_num_tokens,
                op=dist.ReduceOp.MAX,
                group=group)
        expected_tokens_per_expert = (
            int(config_num_tokens.item()) *
            num_ranks * num_topk / num_experts)
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
        grad_y_unweighted = torch.zeros_like(grad_ye)
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
        metadata_before_backward = buffer.token_src_metadata.clone()

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
            route_weight_mode=deep_gemm.RouteWeightMode(
                args.route_weight_mode),
            grad_y_unweighted_output=grad_y_unweighted,
            down_unweighted_output=saved_down_unweighted,
            direct_remote_grad_x=num_ranks > 1,
            write_grad_x_pool=True,
            clear_wgrad_padding=True)
        actual_direct_grad_x_planes = None
        if num_ranks > 1:
            actual_direct_grad_x_planes = torch.as_strided(
                buffer.backward_grad_y,
                size=(
                    num_topk,
                    buffer.num_max_tokens_per_rank,
                    hidden,
                ),
                stride=(
                    buffer.num_max_tokens_per_rank * hidden,
                    hidden,
                    1,
                ),
            ).clone()
        assert torch.equal(
            buffer.token_src_metadata,
            metadata_before_backward), (
                'backward must preserve forward source-route metadata')
        valid_rows = torch.isfinite(
            saved_l1_preact.float()).all(dim=1)
        metadata = buffer.token_src_metadata[valid_rows].long()
        actual_pool_indices = valid_rows.nonzero().flatten()

        ref_grad_ye = torch.zeros_like(grad_ye)
        ref_grad_y_unweighted = torch.zeros_like(
            grad_y_unweighted)
        ref_grad_h = torch.zeros_like(grad_h)
        ref_grad_gate_up = torch.zeros_like(grad_gate_up)
        ref_h_act = torch.zeros_like(h_act)
        ref_h_weighted = torch.zeros_like(h_weighted)
        ref_x_pool = torch.zeros_like(x_pool)
        ref_grad_x_pool = torch.zeros_like(grad_x_pool)
        ref_route_weights = torch.zeros_like(route_weights_pool)
        ref_grad_route = torch.zeros_like(grad_route_pool)

        layout = native_route_layout
        assert torch.equal(layout['counts'], expert_counts)
        native_to_pool = native_to_pool_permutation(metadata, layout)
        pool_indices = actual_pool_indices[native_to_pool]
        source_ranks = layout['source_ranks']
        source_tokens = layout['source_tokens']
        source_slots = layout['source_slots']
        offsets = layout['offsets']
        xe = all_x[source_ranks, source_tokens]
        gye = all_grad_y[source_ranks, source_tokens]
        route = all_topk_weights[
            source_ranks, source_tokens, source_slots]
        preact = saved_l1_preact[pool_indices]
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
            gate_sq = gate_fp32 * gate_fp32
            z = (
                alpha * gate_fp32 *
                (1.0 + beta * gate_sq))
            dz = alpha * (
                1.0 + (3.0 * beta) * gate_sq)
        else:
            z = gate_fp32
            dz = torch.ones_like(gate_fp32)
        sig = torch.sigmoid(z)
        activated_gate = gate_fp32 * sig
        h = (activated_gate * up_fp32).to(torch.bfloat16)
        if args.route_weight_mode == 'pre_down':
            w2_input = (
                h.float() * route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_w2_input = gye
            grad_h_w2 = native_bf16_grouped_mm(
                gye,
                l2_weights_bf16,
                offsets,
                rhs_needs_transpose=False)
            gh = (
                grad_h_w2.float() *
                route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_route = (
                grad_h_w2.float() * h.float()
            ).sum(dim=1)
        else:
            w2_input = h
            grad_w2_input = (
                gye.float() * route.float().unsqueeze(1)
            ).to(torch.bfloat16)
            grad_h_w2 = native_bf16_grouped_mm(
                grad_w2_input,
                l2_weights_bf16,
                offsets,
                rhs_needs_transpose=False)
            gh = grad_h_w2
            down = native_bf16_grouped_mm(
                h, l2_weights_bf16, offsets)
            grad_route = (
                gye.float() * down.float()
            ).sum(dim=1)
            assert torch.equal(
                saved_down_unweighted[pool_indices], down)
        activation_grad = (
            sig + gate_fp32 * sig * (1 - sig) * dz)
        grad_gate = (
            gh.float() * up_fp32 * activation_grad)
        grad_gate = torch.where(
            gate.float() <= clamp,
            grad_gate,
            torch.zeros_like(grad_gate))
        grad_up = gh.float() * activated_gate
        grad_up = torch.where(
            (up.float() >= -clamp) &
            (up.float() <= clamp),
            grad_up,
            torch.zeros_like(grad_up))
        grad_gate_bf16 = grad_gate.to(torch.bfloat16)
        grad_up_bf16 = grad_up.to(torch.bfloat16)
        grad_gu = torch.cat(
            (grad_gate_bf16, grad_up_bf16), dim=1)
        w1_weights = l1_weights_bf16[
            :, :intermediate_hidden].contiguous()
        w3_weights = l1_weights_bf16[
            :, intermediate_hidden:].contiguous()
        grad_xe = native_bf16_grouped_mm(
            grad_gate_bf16,
            w1_weights,
            offsets,
            rhs_needs_transpose=False)
        grad_xe.add_(native_bf16_grouped_mm(
            grad_up_bf16,
            w3_weights,
            offsets,
            rhs_needs_transpose=False))

        ref_grad_ye[actual_pool_indices] = grad_w2_input
        ref_grad_y_unweighted[pool_indices] = gye
        ref_grad_h[pool_indices] = grad_h_w2
        ref_grad_gate_up[actual_pool_indices] = grad_gu
        ref_h_act[pool_indices] = h
        ref_h_weighted[actual_pool_indices] = w2_input
        ref_x_pool[actual_pool_indices] = xe
        ref_grad_x_pool[pool_indices] = grad_xe
        ref_route_weights[pool_indices] = route
        ref_grad_route[pool_indices] = grad_route

        gradient_mismatches = {}

        def assert_gradient_close(
            actual: torch.Tensor,
            expected: torch.Tensor,
            name: str,
        ):
            assert torch.isfinite(actual.float()).all(), name
            if actual.dtype == torch.bfloat16:
                mismatch_count = report_bf16_parity(
                    name, actual, expected)
            else:
                mismatch_count = int(
                    (actual != expected).sum().item())
                max_abs = (
                    float(
                        (actual.float() - expected.float())
                        .abs().max().item())
                    if actual.numel() else 0.0)
                dist_print(
                    f' > {name}: mismatches={mismatch_count}/'
                    f'{actual.numel()}, max_abs={max_abs:.9g}',
                    once_in_node=True)
            if mismatch_count:
                gradient_mismatches[name] = mismatch_count

        assert_gradient_close(grad_ye, ref_grad_ye, 'grad_ye')
        assert_gradient_close(
            grad_y_unweighted, ref_grad_y_unweighted,
            'grad_y_unweighted')
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
        repeated_grad_h = native_bf16_grouped_mm(
            grad_w2_input,
            l2_weights_bf16,
            offsets,
            rhs_needs_transpose=False)
        repeated_grad_x = native_bf16_grouped_mm(
            grad_gate_bf16,
            w1_weights,
            offsets,
            rhs_needs_transpose=False)
        repeated_grad_x.add_(native_bf16_grouped_mm(
            grad_up_bf16,
            w3_weights,
            offsets,
            rhs_needs_transpose=False))
        if args.route_weight_mode == 'pre_down':
            repeated_grad_route = (
                repeated_grad_h.float() * h.float()
            ).sum(dim=1)
        else:
            repeated_down = native_bf16_grouped_mm(
                h, l2_weights_bf16, offsets)
            repeated_grad_route = (
                gye.float() * repeated_down.float()
            ).sum(dim=1)
        assert_gradient_close(
            repeated_grad_h, grad_h_w2,
            'native_repeat_grad_h')
        assert_gradient_close(
            repeated_grad_x, grad_xe,
            'native_repeat_grad_x')
        assert_gradient_close(
            repeated_grad_route, grad_route,
            'native_repeat_grad_route')

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
                buffer,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13_combine(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts, grad_x_combined,
                buffer)
        else:
            deep_gemm.bf16_mega_moe_backward_w2(
                grad_w2, grad_ye, h_weighted,
                padded_expert_counts,
                route_weight_mode=deep_gemm.RouteWeightMode(
                    args.route_weight_mode))
            deep_gemm.bf16_mega_moe_backward_w13(
                grad_w13, grad_gate_up, x_pool,
                padded_expert_counts)
        compact_grad_ye = ref_grad_ye[actual_pool_indices]
        compact_h_weighted = ref_h_weighted[actual_pool_indices]
        compact_grad_gate = ref_grad_gate_up[
            actual_pool_indices, :intermediate_hidden]
        compact_grad_up = ref_grad_gate_up[
            actual_pool_indices, intermediate_hidden:]
        compact_x = ref_x_pool[actual_pool_indices]
        # These are the exact native FireTitan transposed-LHS grouped-MM
        # calls. They preserve the native K-group boundaries and BF16 output
        # rounding; no per-expert Python matmul participates in parity.
        ref_grad_w2 = native_bf16_grouped_mm(
            compact_grad_ye.t(),
            compact_h_weighted,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        ref_grad_w1 = native_bf16_grouped_mm(
            compact_grad_gate.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        ref_grad_w3 = native_bf16_grouped_mm(
            compact_grad_up.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        inactive = (expert_counts == 0).view(-1, 1, 1)
        ref_grad_w2.masked_fill_(inactive, 0)
        ref_grad_w1.masked_fill_(inactive, 0)
        ref_grad_w3.masked_fill_(inactive, 0)
        ref_grad_w13 = torch.cat(
            (ref_grad_w1, ref_grad_w3), dim=1)
        repeated_grad_w2 = native_bf16_grouped_mm(
            compact_grad_ye.t(),
            compact_h_weighted,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w1 = native_bf16_grouped_mm(
            compact_grad_gate.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w3 = native_bf16_grouped_mm(
            compact_grad_up.t(),
            compact_x,
            offsets,
            rhs_needs_transpose=False,
            lhs_needs_contiguous=False)
        repeated_grad_w2.masked_fill_(inactive, 0)
        repeated_grad_w1.masked_fill_(inactive, 0)
        repeated_grad_w3.masked_fill_(inactive, 0)
        assert_gradient_close(
            repeated_grad_w2, ref_grad_w2,
            'native_repeat_grad_w2')
        assert_gradient_close(
            repeated_grad_w1, ref_grad_w1,
            'native_repeat_grad_w1')
        assert_gradient_close(
            repeated_grad_w3, ref_grad_w3,
            'native_repeat_grad_w3')
        assert_gradient_close(
            grad_w2, ref_grad_w2, 'grad_w2')
        assert_gradient_close(
            grad_w13, ref_grad_w13, 'grad_w13')

        if num_ranks > 1:
            ref_grad_x_planes = torch.zeros(
                (num_ranks, buffer.num_max_tokens_per_rank,
                 num_topk, hidden),
                dtype=torch.bfloat16, device='cuda')
            actual_route_planes = torch.zeros(
                (num_ranks, buffer.num_max_tokens_per_rank,
                 num_topk),
                dtype=torch.float, device='cuda')
            ref_route_planes = torch.zeros_like(
                actual_route_planes)
            ref_grad_x_planes[
                source_ranks, source_tokens, source_slots
            ] = ref_grad_x_pool[pool_indices]
            actual_route_planes[
                metadata[:, 0], metadata[:, 1], metadata[:, 2]
            ] = grad_route_pool[actual_pool_indices]
            ref_route_planes[
                source_ranks, source_tokens, source_slots
            ] = ref_grad_route[pool_indices]
            dist.all_reduce(ref_grad_x_planes, group=group)
            dist.all_reduce(actual_route_planes, group=group)
            dist.all_reduce(ref_route_planes, group=group)
            assert actual_direct_grad_x_planes is not None
            direct_expected = ref_grad_x_planes[
                rank_idx, :num_tokens
            ].permute(1, 0, 2)
            if not torch.equal(
                actual_direct_grad_x_planes[:, :num_tokens],
                direct_expected,
            ):
                cross_slot_mismatches = [
                    [
                        int((
                            actual_direct_grad_x_planes[
                                actual_slot, :num_tokens
                            ] != direct_expected[expected_slot]
                        ).sum().item())
                        for expected_slot in range(num_topk)
                    ]
                    for actual_slot in range(num_topk)
                ]
                dist_print(
                    f' > grad_x source slot cross-mismatches: '
                    f'{cross_slot_mismatches}',
                    once_in_node=True)
            assert_gradient_close(
                actual_direct_grad_x_planes[
                    :, :num_tokens
                ].permute(1, 0, 2),
                ref_grad_x_planes[
                    rank_idx, :num_tokens
                ],
                'grad_x_source_planes')
            ref_grad_x_fp32 = torch.zeros(
                (num_tokens, hidden),
                dtype=torch.float32,
                device='cuda')
            for slot in range(num_topk):
                ref_grad_x_fp32 += ref_grad_x_planes[
                    rank_idx, :num_tokens, slot].float()
            ref_grad_x = ref_grad_x_fp32.to(torch.bfloat16)
            assert_gradient_close(
                grad_x_combined, ref_grad_x,
                'grad_x_combined')
            assert_gradient_close(
                actual_route_planes[rank_idx, :num_tokens],
                ref_route_planes[rank_idx, :num_tokens],
                'grad_route_source')

        assert not gradient_mismatches, (
            'native FireTitan gradient mismatches: '
            f'{gradient_mismatches}')

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
                check_native_forward_repeatability(ref_y)
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

    if args.correctness_only:
        dist.barrier()
        buffer.destroy()
        ep_buffer.destroy() if ep_buffer else None
        dist.destroy_process_group()
        return

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
    parser.add_argument('--route-weight-mode', type=str, default='pre_down', choices=['pre_down', 'post_down'], help='Location of the BF16 route-weight boundary')
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
    parser.add_argument('--correctness-only', action='store_true', help='Exit after correctness validation')
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
