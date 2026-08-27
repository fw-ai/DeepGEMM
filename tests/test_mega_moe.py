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
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    use_lora_payload = args.lora_mode != 'disabled'
    num_lora_slots = (
        num_tokens + 1 if args.lora_mode == 'payload_only'
        else args.num_lora_slots
        if args.lora_mode in ('fc1', 'fc1_down')
        else 0)
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        mma_type=args.mma_type,
        activation=args.activation,
        lora_rank=128 if use_lora_payload else 0,
        num_lora_slots=num_lora_slots,
        enable_lora_down=args.lora_mode == 'fc1_down'
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
        global lora_gate_up_acts, lora_adapter_slots, lora_gate_b, lora_up_b
        global lora_down_a, lora_scaling
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
        if args.expert_pattern == 'single_expert':
            topk_idx.zero_()
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if use_lora_payload:
            lora_gate_up_acts = torch.randn(
                (num_tokens, 2, 128), dtype=torch.bfloat16, device='cuda')
            if args.lora_mode == 'payload_only':
                # Unique IDs make the expert-major dispatch permutation directly
                # observable; the final token exercises the reserved sentinel slot.
                lora_adapter_slots = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
                if num_tokens:
                    lora_adapter_slots[-1] = buffer.lora_sentinel_slot
            else:
                # Deterministically mix every active slot and the sentinel
                # across source tokens; routing then distributes those rows
                # independently into expert-major ring blocks.
                if args.lora_slot_pattern == 'all_active':
                    lora_adapter_slots = torch.zeros(
                        num_tokens, dtype=torch.int32, device='cuda')
                else:
                    lora_adapter_slots = (
                        torch.arange(
                            num_tokens, dtype=torch.int32, device='cuda') *
                        7) % num_lora_slots
                lora_gate_b = torch.zeros(
                    (num_lora_slots, num_experts_per_rank,
                     intermediate_hidden, 128),
                    dtype=torch.bfloat16, device='cuda')
                lora_up_b = torch.zeros_like(lora_gate_b)
                lora_gate_b[:-1].normal_(std=0.05)
                lora_up_b[:-1].normal_(std=0.05)
                if args.lora_mode == 'fc1_down':
                    lora_down_a = torch.zeros(
                        (num_lora_slots, num_experts_per_rank,
                         128, intermediate_hidden),
                        dtype=torch.bfloat16, device='cuda')
                    lora_down_a[:-1].normal_(std=0.05)
                    lora_scaling = 0.25
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
            deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def run_fused(lora_mode=None, scaling=None):
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        if use_lora_payload:
            buffer.set_lora_payload(lora_gate_up_acts, lora_adapter_slots)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation=args.activation,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math))
        if not is_bf16xbf16:
            kernel_kwargs['lora_mode'] = args.lora_mode if lora_mode is None else lora_mode
            if kernel_kwargs['lora_mode'] in ('fc1', 'fc1_down'):
                kernel_kwargs['lora_gate_b'] = lora_gate_b
                kernel_kwargs['lora_up_b'] = lora_up_b
            if kernel_kwargs['lora_mode'] == 'fc1_down':
                kernel_kwargs['lora_down_a'] = lora_down_a
                kernel_kwargs['lora_scaling'] = (
                    lora_scaling if scaling is None else scaling)
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    def assert_fc1_subgroup_layout():
        """Check device prefixes and physical expert/adapter ring ordering."""
        assert args.lora_mode in ('fc1', 'fc1_down')
        gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
        gathered_slots = uneven_all_gather(lora_adapter_slots, group=group)
        routed_slots = gathered_slots[:, None].expand(-1, num_topk)
        if not hasattr(buffer, 'lora_subgroup_offsets'):
            raise AssertionError(
                f'missing subgroup view on {type(buffer)}: '
                f'{sorted(buffer.__dict__)}')
        subgroup_offsets = buffer.lora_subgroup_offsets.cpu()

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

        pool_block_offset = 0
        for local_expert_idx in range(num_experts_per_rank):
            global_expert_idx = rank_idx * num_experts_per_rank + local_expert_idx
            slots = routed_slots[gathered_topk_idx == global_expert_idx]
            slots = slots.clamp(0, buffer.lora_sentinel_slot)
            counts = torch.bincount(
                slots.long(), minlength=num_lora_slots).cpu()
            expected_offsets = torch.cat((
                torch.zeros(1, dtype=torch.long, device='cpu'),
                counts.cumsum(0)))
            assert torch.equal(
                subgroup_offsets[local_expert_idx].long(),
                expected_offsets), (
                    f'subgroup offsets mismatch for local expert '
                    f'{local_expert_idx}')

            num_rows = int(expected_offsets[-1])
            if num_rows:
                ring_row = (
                    pool_block_offset * block_m +
                    torch.arange(num_rows, device='cuda')
                ) % buffer.num_ring_tokens
                actual_slots = buffer.dispatched_lora_adapter_slots[ring_row]
                expected_slots = torch.repeat_interleave(
                    torch.arange(
                        num_lora_slots, dtype=torch.int32, device='cuda'),
                    counts.to(device='cuda'))
                assert torch.equal(actual_slots, expected_slots), (
                    f'non-contiguous adapter rows for local expert '
                    f'{local_expert_idx}')
            pool_block_offset += (num_rows + block_m - 1) // block_m

    # Self-contained PyTorch reference.  Gather expert-owned parameters across
    # ranks while keeping source-token activations local, matching routed
    # execution followed by the return-to-source combine.
    # Mirrors the fused pipeline: L1 GEMM -> gated activation (* topk weight) ->
    # per-32 UE8M0 FP8 requantization -> L2 GEMM -> top-k combine (plain sum).
    def run_reference():
        def gather_expert_tensor(tensor):
            gathered = [torch.empty_like(tensor) for _ in range(num_ranks)]
            dist.all_gather(gathered, tensor, group=group)
            return torch.cat(gathered, dim=0)

        def gather_lora_tensor(tensor):
            gathered = [torch.empty_like(tensor) for _ in range(num_ranks)]
            dist.all_gather(gathered, tensor, group=group)
            # [rank, slot, local_expert, ...] ->
            # [slot, global_expert, ...]
            ranked = torch.stack(gathered, dim=0)
            trailing_dims = tuple(ranked.shape[3:])
            return ranked.permute(
                1, 0, 2, *range(3, ranked.dim())
            ).reshape(num_lora_slots, num_experts, *trailing_dims)

        clamp = float(args.activation_clamp)
        x_deq = _dequant_x_fp8(x[0][:num_tokens], x[1][:num_tokens])
        l1_w = _dequant_weight_fp4(
            gather_expert_tensor(l1_weights_bf16))
        l2_w = _dequant_weight_fp4(
            gather_expert_tensor(l2_weights_bf16))
        if args.lora_mode in ('fc1', 'fc1_down'):
            global_lora_gate_b = gather_lora_tensor(lora_gate_b)
            global_lora_up_b = gather_lora_tensor(lora_up_b)
        if args.lora_mode == 'fc1_down':
            global_lora_down_a = gather_lora_tensor(lora_down_a)

        y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device='cuda')
        rank_y = torch.zeros(
            (num_tokens, 128), dtype=torch.float32, device='cuda')
        for slot in range(num_topk):
            expert_idx = topk_idx[:num_tokens, slot]
            weight = topk_weights[:num_tokens, slot].float()
            for e in range(num_experts):
                mask = expert_idx == e
                if not bool(mask.any()):
                    continue
                xt = x_deq[mask]

                # L1 GEMM then split into gate/up (first/second half of the output)
                acc1 = xt @ l1_w[e].t()
                gate = acc1[:, :intermediate_hidden]
                up = acc1[:, intermediate_hidden:]
                if args.lora_mode in ('fc1', 'fc1_down'):
                    adapter_slots = lora_adapter_slots[mask].long()
                    gate_delta = torch.einsum(
                        'tr,tir->ti',
                        lora_gate_up_acts[mask, 0].float(),
                        global_lora_gate_b[adapter_slots, e].float())
                    up_delta = torch.einsum(
                        'tr,tir->ti',
                        lora_gate_up_acts[mask, 1].float(),
                        global_lora_up_b[adapter_slots, e].float())
                    gate = gate + gate_delta
                    up = up + up_delta
                gate = gate.to(torch.bfloat16)
                up = up.to(torch.bfloat16)
                if clamp != float('inf'):
                    gate = torch.clamp(gate, max=clamp)
                    up = torch.clamp(up, min=-clamp, max=clamp)

                # Gated activation with the per-token routing weight folded in
                act = _apply_gate_activation(gate.float(), args.activation) * up.float()
                act = act * weight[mask].unsqueeze(1)

                # Requantize to FP8 (per-32 UE8M0), matching the kernel's L1 output
                act_fp8, act_sf = per_token_cast_to_fp8(act, use_ue8m0=True, gran_k=32)
                n_groups = intermediate_hidden // 32
                act_deq = (act_fp8.float().view(-1, n_groups, 32) * act_sf[:, :n_groups].unsqueeze(2)
                           ).view(-1, intermediate_hidden)

                # L2 GEMM, accumulate across the top-k experts
                y[mask] += act_deq @ l2_w[e].t()
                if args.lora_mode == 'fc1_down':
                    adapter_slots = lora_adapter_slots[mask].long()
                    rank_y[mask] += torch.einsum(
                        'ti,tri->tr',
                        act_deq,
                        global_lora_down_a[adapter_slots, e].float())
        if args.lora_mode == 'fc1_down':
            rank_y *= lora_scaling
        return y.to(torch.bfloat16), rank_y.to(torch.bfloat16)

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    if args.lora_mode in ('fc1', 'fc1_down'):
        dist_print(
            f' > LoRA slots: {num_lora_slots - 1} active + sentinel',
            once_in_node=True)
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
    # The self-contained numerical reference covers any activation but only the
    # single-rank FP8 path (it models the kernel's FP8 L1-output requantization)
    use_numerical_reference = num_ranks == 1 and not is_bf16xbf16
    ran_correctness = False

    if use_lora_payload and num_correctness_tests > 0:
        assert not is_bf16xbf16, 'LoRA payload transport is only supported by fp8xfp4'
        if args.lora_mode == 'payload_only':
            dist_print('Running LoRA payload-only transport test:', once_in_node=True)
            create_inputs()
            baseline_y, _ = run_fused(lora_mode='disabled')
            payload_y, _ = run_fused(lora_mode='payload_only')
            assert torch.equal(payload_y, baseline_y), 'payload-only mode changed MegaMoE output'

            # A single rank/expert/top-k has one contiguous expert-major segment.
            # Sort by the unique transported slots to undo dispatch's atomic order.
            if num_ranks == num_experts == num_topk == 1:
                dispatched_slots = buffer.dispatched_lora_adapter_slots[:num_tokens]
                dispatched_acts = buffer.dispatched_lora_gate_up_acts[:num_tokens]
                dispatched_order = torch.argsort(dispatched_slots)
                source_order = torch.argsort(lora_adapter_slots)
                assert torch.equal(dispatched_slots[dispatched_order], lora_adapter_slots[source_order])
                assert torch.equal(dispatched_acts[dispatched_order], lora_gate_up_acts[source_order])
            dist_print(' > Payload transport and exact no-add output passed', once_in_node=True)
            dist_print(once_in_node=True)
        elif args.lora_mode == 'fc1':
            dist_print('Running FC1 zero-delta exactness test:', once_in_node=True)
            create_inputs()
            baseline_y, _ = run_fused(lora_mode='disabled')
            lora_gate_b.zero_()
            lora_up_b.zero_()
            zero_delta_y, _ = run_fused(lora_mode='fc1')
            assert torch.equal(zero_delta_y, baseline_y), 'zero FC1-B changed MegaMoE output'

            create_inputs()
            baseline_y, _ = run_fused(lora_mode='disabled')
            mixed_y, _ = run_fused(lora_mode='fc1')
            assert not torch.equal(
                mixed_y, baseline_y), 'mixed active adapters produced no output delta'
            assert_fc1_subgroup_layout()
            lora_adapter_slots.fill_(buffer.lora_sentinel_slot)
            sentinel_y, _ = run_fused(lora_mode='fc1')
            assert torch.equal(
                sentinel_y, baseline_y), 'all-sentinel FC1 payload changed output'
            assert_fc1_subgroup_layout()
            dist_print(
                ' > Zero-B and all-sentinel paths preserved exact output; '
                'mixed active slots changed output; subgroup layout passed',
                once_in_node=True)
            dist_print(once_in_node=True)
        else:
            dist_print(
                'Running FC1 + down-A rank-sideband tests:',
                once_in_node=True)
            create_inputs()
            fc1_y, _ = run_fused(lora_mode='fc1')

            lora_down_a.zero_()
            zero_down_y, _ = run_fused(lora_mode='fc1_down')
            zero_rank = buffer.combined_lora_rank_acts[:num_tokens].clone()
            zero_base_diff = calc_diff(zero_down_y, fc1_y)
            assert zero_base_diff < 1e-3, (
                    'zero down-A changed the FC1/base MegaMoE output: '
                    f'diff={zero_base_diff:.5f}, '
                    f'max_abs={(zero_down_y.float() - fc1_y.float()).abs().max().item():.5f}, '
                    f'count={torch.count_nonzero(zero_down_y != fc1_y).item()}')
            assert torch.count_nonzero(zero_rank).item() == 0, (
                'zero down-A produced a nonzero combined rank')

            create_inputs()
            fc1_y, _ = run_fused(lora_mode='fc1')
            down_y, _ = run_fused(lora_mode='fc1_down')
            actual_rank = buffer.combined_lora_rank_acts[:num_tokens].clone()
            ref_y, ref_rank = run_reference()
            down_base_diff = calc_diff(down_y, fc1_y)
            assert down_base_diff < 1e-3, (
                    'down-A sideband changed the base FC1 output: '
                    f'diff={down_base_diff:.5f}')
            rank_diff = calc_diff(actual_rank, ref_rank)
            assert rank_diff < 5e-3, (
                f'down rank intermediate mismatch: diff={rank_diff:.5f}, '
                f'max_abs={(actual_rank.float() - ref_rank.float()).abs().max().item():.5f}')
            assert_fc1_subgroup_layout()

            half_y, _ = run_fused(
                lora_mode='fc1_down', scaling=lora_scaling * 0.5)
            half_rank = buffer.combined_lora_rank_acts[:num_tokens].clone()
            half_base_diff = calc_diff(half_y, fc1_y)
            assert half_base_diff < 1e-3, (
                'scaled down sideband changed the base FC1 output: '
                f'diff={half_base_diff:.5f}')
            half_scale_diff = calc_diff(
                half_rank, actual_rank * 0.5)
            assert half_scale_diff < 5e-3, (
                'LoRA scaling was not applied exactly once: '
                f'diff={half_scale_diff:.5f}')

            lora_adapter_slots.fill_(buffer.lora_sentinel_slot)
            sentinel_y, _ = run_fused(lora_mode='fc1_down')
            sentinel_rank = buffer.combined_lora_rank_acts[:num_tokens]
            baseline_y, _ = run_fused(lora_mode='disabled')
            assert torch.equal(
                sentinel_y, baseline_y), (
                    'all-sentinel FC1+down payload changed base output')
            assert torch.count_nonzero(sentinel_rank).item() == 0, (
                'all-sentinel routes produced a nonzero combined rank')
            assert_fc1_subgroup_layout()
            dist_print(
                ' > Zero-A/sentinel exactness, mixed adapters, subgroup '
                f'boundaries, rank oracle (diff {rank_diff:.5f}), and '
                'single scaling passed',
                once_in_node=True)
            dist_print(once_in_node=True)

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
            ref_y, _ = run_reference()
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

    # Benchmark before inspecting the dispatched payload so the physical ring
    # contains the current route/slot assignment.
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    if args.lora_mode in ('fc1', 'fc1_down'):
        expected_tokens_per_expert = (
            num_tokens * num_ranks * num_topk / num_experts)
        if expected_tokens_per_expert <= 8.5:
            lora_block_m = 16
        elif expected_tokens_per_expert <= 16.5:
            lora_block_m = 32
        elif expected_tokens_per_expert <= 32.5:
            lora_block_m = 64
        elif expected_tokens_per_expert <= 64.5:
            lora_block_m = 96
        elif expected_tokens_per_expert <= 96.5:
            lora_block_m = 128
        else:
            lora_block_m = 192

        local_expert_idx = (
            gathered_topk_idx[gathered_topk_idx >= 0] -
            rank_idx * num_experts_per_rank)
        local_counts = torch.bincount(
            local_expert_idx, minlength=num_experts_per_rank).cpu().tolist()
        masked_mma_rows = compact_mma_rows = routed_mma_rows = 0
        useful_mma_rows = 0
        pool_block_offset = 0
        num_ring_blocks = buffer.num_ring_tokens // lora_block_m
        for expert_count in local_counts:
            for block_start in range(0, expert_count, lora_block_m):
                valid = min(lora_block_m, expert_count - block_start)
                ring_block = (
                    pool_block_offset + block_start // lora_block_m
                ) % num_ring_blocks
                slots = buffer.dispatched_lora_adapter_slots[
                    ring_block * lora_block_m:
                    ring_block * lora_block_m + valid]
                active = slots[slots < buffer.lora_sentinel_slot]
                distinct = torch.unique(active)
                # The replaced masked implementation always ran slot zero to
                # initialize its full-M TMEM delta tile.
                masked_slots = (
                    int(distinct.numel()) +
                    int(not bool((distinct == 0).any())))
                masked_mma_rows += masked_slots * lora_block_m
                for slot in distinct.tolist():
                    slot_rows = (slots == slot).nonzero().flatten()
                    count = int(slot_rows.numel())
                    compact_mma_rows += ((count + 15) // 16) * 16
                    segment_start = int(slot_rows[0])
                    routed_mma_rows += (
                        (segment_start % 16 + count + 15) // 16) * 16
                    useful_mma_rows += count
            pool_block_offset += (
                (expert_count + lora_block_m - 1) // lora_block_m)
        dist_print(
            f' > LoRA MMA M rows/N-tile: masked={masked_mma_rows}, '
            f'compact={compact_mma_rows}, routed={routed_mma_rows}, '
            f'useful={useful_mma_rows}, '
            f'routed_padding={routed_mma_rows - useful_mma_rows}',
            once_in_node=False)

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
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation', type=str, default='swiglu', choices=['swiglu', 'geglu'], help='Gated activation type')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', help='MMA type: fp8xfp4 or bf16xbf16')
    parser.add_argument('--lora-mode', type=str, default='disabled',
                        choices=['disabled', 'payload_only', 'fc1', 'fc1_down'],
                        help='Optional rank-128 MegaMoE LoRA transport mode')
    parser.add_argument('--num-lora-slots', type=int, default=2,
                        help='FC1 LoRA slots including the final sentinel (2..32)')
    parser.add_argument('--lora-slot-pattern', type=str, default='round_robin',
                        choices=['round_robin', 'all_active'],
                        help='FC1 adapter-slot test pattern')
    parser.add_argument('--expert-pattern', type=str, default='random',
                        choices=['random', 'single_expert'],
                        help='Expert route pattern for dispatch stress tests')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()
    if (args.lora_mode in ('fc1', 'fc1_down') and
            not 2 <= args.num_lora_slots <= 32):
        parser.error(
            '--num-lora-slots must be in [2, 32] for fc1/fc1_down mode')
    if args.expert_pattern == 'single_expert' and args.num_topk != 1:
        parser.error('--expert-pattern single_expert requires --num-topk 1')

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
