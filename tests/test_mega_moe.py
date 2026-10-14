import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Optional, Tuple

import deep_gemm
from deep_gemm.utils import align, cast_back_from_fp4, cast_back_from_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major, test_filter

# `test_filter` is an imported decorator, not a test: keep pytest from collecting it
test_filter.__test__ = False


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


def _to_shared_mega_moe_sf_layout(sf: torch.Tensor, block_m: int, num_max_sf_tokens: int) -> torch.Tensor:
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = align(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype, device=sf.device)
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(sf[block_idx * block_m + m_idx])
    return result


def _cast_fp8_for_mega_moe(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, align(mn, 4)), dtype=x_sf.dtype, device=x_sf.device)
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


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
    num_shared_experts = args.num_shared_experts
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type=args.mma_type
    )

    # Cast routed weights into FP4 or FP8
    def _cast_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    def _cast_weights_to_fp8(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_groups, n, k = bf16_weights.shape
        w = torch.empty((num_groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)
        w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            w[i], w_sf[i] = per_token_cast_to_fp8(bf16_weights[i], use_ue8m0=True, gran_k=32)
        w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
        return w, w_sf

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, shared_x, shared_l1_x_sf, topk_idx, topk_weights, l1_weights, l2_weights
        global transformed_l1_weights, transformed_l2_weights
        global shared_l1_weights, shared_l2_weights, transformed_shared_l1_weights, transformed_shared_l2_weights
        global cumulative_local_expert_recv_stats_fused, cumulative_local_expert_recv_stats_baseline
        global initial_cumulative_local_expert_recv_stats_fused, initial_cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_fused = cumulative_local_expert_recv_stats_fused.clone()
        initial_cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_baseline.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        if num_shared_experts > 0:
            shared_l1_weights = torch.randn(
                (shared_intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
            shared_l2_weights = torch.randn(
                (hidden, shared_intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        else:
            shared_l1_weights = shared_l2_weights = None

        if not is_bf16xbf16:
            # FP8 path: cast inputs and weights with per-32 UE8M0 SF
            assert hidden % 128 == 0 and intermediate_hidden % 128 == 0 and shared_intermediate_hidden % 128 == 0
            block_m = deep_gemm.get_block_m_for_mega_moe(
                num_ranks, num_experts, buffer.num_max_tokens_per_rank, num_tokens, num_topk, args.mma_type)
            x_fp8, x_sf, x_sf_tma = _cast_fp8_for_mega_moe(x)
            x = (x_fp8, x_sf)
            shared_x = (x_fp8, x_sf_tma)
            if num_shared_experts > 0:
                shared_l1_x_sf = _to_shared_mega_moe_sf_layout(x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])
            cast_weights = _cast_weights_to_fp8 if args.mma_type == 'fp8xfp8' else _cast_weights_to_fp4
            l1_weights = cast_weights(l1_weights)
            l2_weights = cast_weights(l2_weights)
            if num_shared_experts > 0:
                shared_l1_weights = _cast_fp8_for_mega_moe(shared_l1_weights)[0::2]
                shared_l2_weights = _cast_fp8_for_mega_moe(shared_l2_weights)[0::2]

        transformed_l1_weights, transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))
        if num_shared_experts > 0:
            transformed_shared_l1_weights, transformed_shared_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe(shared_l1_weights, shared_l2_weights))
        else:
            transformed_shared_l1_weights = transformed_shared_l2_weights = None

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    def copy_inputs_to_buffer():
        if is_bf16xbf16:
            buffer.x[:num_tokens].copy_(x)
        else:
            buffer.x[:num_tokens].copy_(x[0])
            buffer.x_sf[:num_tokens].copy_(x[1])
            if num_shared_experts > 0:
                _copy_fp8_sf(buffer.shared_l1_acts_sf, shared_l1_x_sf, num_tokens)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused():
        cumulative_local_expert_recv_stats_fused.copy_(initial_cumulative_local_expert_recv_stats_fused)
        copy_inputs_to_buffer()

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        kernel_kwargs = dict(
            y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
            sym_buffer=buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math))
        if num_shared_experts > 0:
            kernel_kwargs.update(
                shared_l1_weights=transformed_shared_l1_weights,
                shared_l2_weights=transformed_shared_l2_weights
            )
        (deep_gemm.bf16_mega_moe if is_bf16xbf16 else deep_gemm.fp8_fp4_mega_moe)(**kernel_kwargs)
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > MMA: {args.mma_type}', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Shared experts: {num_shared_experts}', once_in_node=True)
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
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=not is_bf16xbf16,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
        num_gpu_timeout_secs=10, num_cpu_timeout_secs=30
    ) if is_legacy_loaded else None

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

        def get_baseline_shared_bias() -> Optional[torch.Tensor]:
            if num_shared_experts == 0:
                return None

            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            if is_bf16xbf16:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.bf16_gemm_nt(x, shared_l1_weights, l1_out)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=128, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=False, ue8m0_scale=False, output_bf16=True)[-1]
                deep_gemm.bf16_gemm_nt(l2_in, shared_l2_weights, y)
            else:
                l1_out = torch.empty((num_tokens, shared_intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
                deep_gemm.fp8_gemm_nt(shared_x, shared_l1_weights, l1_out, recipe=(1, 1, 32), disable_ue8m0_cast=True)
                l2_in = tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=l1_out, topk_weights=None,
                    avail_tokens=None,
                    num_per_channels=32, use_col_major_scales=True,
                    clamp_value=args.activation_clamp, fast_math=bool(args.fast_math),
                    round_scale=True, ue8m0_scale=True, output_bf16=False)
                deep_gemm.fp8_gemm_nt(l2_in, shared_l2_weights, y, recipe=(1, 1, 32), disable_ue8m0_cast=True)
            return y

        def run_baseline():
            cumulative_local_expert_recv_stats_baseline.copy_(initial_cumulative_local_expert_recv_stats_baseline)
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
            return (
                ep_buffer.combine(l2_y, handle=handle, bias=get_baseline_shared_bias())[0],
                cumulative_local_expert_recv_stats_baseline
            )

    # Check correctness
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_y, fused_stats = run_fused()
            baseline_y, baseline_stats = run_baseline()
            assert torch.equal(fused_stats, baseline_stats)
            if num_shared_experts == 0:
                assert torch.equal(fused_y, baseline_y)
            else:
                assert calc_diff(fused_y, baseline_y) < 1e-8
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    barrier_fn = lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.all_reduce(torch.empty(1, device='cuda'))
    trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json'
    t_fused = bench_kineto(run_fused, 'mega_moe', barrier=barrier_fn, trace_path=trace_path)
    t_baseline = tilelang_bench(
        run_baseline, _n_warmup=5, _n_repeat=1,
        backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: routed + shared L1/L2, each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3
    num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3
    num_total_flops = num_routed_flops + num_shared_flops

    # HBM bytes: weights + activations + output
    num_touched_experts = torch.unique(gathered_topk_idx[gathered_topk_idx >= 0]).numel()
    act_elem_size = 2 if is_bf16xbf16 else 1
    routed_weight_elem_size = 2 if is_bf16xbf16 else (1 if args.mma_type == 'fp8xfp8' else 0.5)
    shared_weight_elem_size = 2 if is_bf16xbf16 else 1
    num_routed_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden * routed_weight_elem_size      # L1 weights
        + num_touched_experts * hidden * intermediate_hidden * routed_weight_elem_size        # L2 weights
        + num_recv_tokens * hidden * act_elem_size                                            # L1 acts read
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L1 output write
        + num_recv_tokens * intermediate_hidden * act_elem_size                               # L2 acts read
        + num_recv_tokens * hidden * 2                                                        # L2 output write (always BF16)
    )
    num_shared_hbm_bytes = 0 if num_shared_experts == 0 else (
        shared_intermediate_hidden * 2 * hidden * shared_weight_elem_size      # Shared L1 weights
        + hidden * shared_intermediate_hidden * shared_weight_elem_size        # Shared L2 weights
        + num_tokens * hidden * act_elem_size                                  # Shared L1 acts read
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L1 output write
        + num_tokens * shared_intermediate_hidden * act_elem_size              # Shared L2 acts read
        + num_tokens * hidden * 2                                              # Shared L2 output write
    )
    num_hbm_bytes = num_routed_hbm_bytes + num_shared_hbm_bytes

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    def print_perf(elapsed: float, ref_time: float, ref_label: str):
        tflops = safe_div(num_total_flops / 1e12, elapsed)
        hbm_gbs = safe_div(num_hbm_bytes / 1e9, elapsed)
        nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, elapsed)
        approx_factor = safe_div(elapsed, elapsed - t_reduction)
        dist_print(f' > EP {rank_idx:2}/{num_ranks} | '
                   f'{tflops:4.0f} TFLOPS | '
                   f'overlap: '
                   f'{tflops * approx_factor:4.0f} TFLOPS, '
                   f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
                   f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
                   f'{elapsed * 1e6:4.0f} us, '
                   f'reduction: {t_reduction * 1e6:4.1f} us | '
                   f'{safe_div(ref_time, elapsed):.2f}x {ref_label}')

    dist_print(f'Performance (w/{"" if num_shared_experts else "o"} shared):', once_in_node=True)
    print_perf(t_fused, t_baseline, f'legacy{"+shared" if num_shared_experts else ""}')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


# The benchmark/correctness entry point above takes spawn arguments; keep pytest from collecting it.
test.__test__ = False


# ---------------------------------------------------------------------------------------------------------------
# Gated activations (SwiGLU / GeGLU / SiTU) vs. a dequantized PyTorch reference.
#
# SiTU (Kimi K3) replaces SiLU's numerator and optionally squashes `up`:
#   act = situ_beta * tanh(gate / situ_beta) * sigmoid(gate) * (situ_linear_beta * tanh(up / situ_linear_beta))
# with `up` left as is when `situ_linear_beta` is unset (+inf). The reference follows the kernel's numerical
# boundaries: exact FP8/FP4 dequantization, FP32 GEMMs, BF16 rounding of the L1 output before the activation,
# the route weight applied before the L2 input (`pre_down`), per-32 UE8M0 FP8 requantization of the L2 input,
# BF16 L2 output per (token, expert) and an FP32 top-k sum rounded to BF16.
# ---------------------------------------------------------------------------------------------------------------

_ACT_HIDDEN, _ACT_INTERMEDIATE, _ACT_EXPERTS, _ACT_TOPK, _ACT_TOKENS = 4096, 1536, 16, 4, 96


def _gated_activation_reference(gate: torch.Tensor, up: torch.Tensor, activation: str,
                                situ_beta: Optional[float] = None,
                                situ_linear_beta: Optional[float] = None) -> torch.Tensor:
    if activation == 'swiglu':
        return gate * torch.sigmoid(gate) * up
    if activation == 'geglu':
        return 0.5 * gate * (1 + torch.tanh(0.7978845608028654 * (gate + 0.044715 * gate ** 3))) * up
    assert activation == 'situ'
    beta = 1.0 if situ_beta is None else situ_beta
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if situ_linear_beta is not None:
        up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
    return act * up


def _mega_moe_reference(x_deq, topk_idx, topk_weights, l1_deq, l2_deq, activation, **situ_kwargs):
    num_tokens, hidden = x_deq.shape
    inter = l2_deq.shape[2]
    y = torch.zeros((num_tokens, hidden), dtype=torch.float, device='cuda')
    for expert in range(l1_deq.shape[0]):
        rows, slots = (topk_idx == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        l1 = (x_deq[rows] @ l1_deq[expert].T).bfloat16().float()
        act = _gated_activation_reference(l1[:, :inter], l1[:, inter:], activation, **situ_kwargs)
        act = act * topk_weights[rows, slots].unsqueeze(1)
        act_fp8, act_sf = per_token_cast_to_fp8(act, use_ue8m0=True, gran_k=32)
        l2 = (cast_back_from_fp8(act_fp8, act_sf, gran_k=32) @ l2_deq[expert].T).bfloat16().float()
        y.index_add_(0, rows, l2)
    return y.bfloat16()


def _activation_worker(local_rank: int, num_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_ranks)
    hidden, inter, num_experts, num_topk, num_tokens = (
        _ACT_HIDDEN, _ACT_INTERMEDIATE, _ACT_EXPERTS, _ACT_TOPK, _ACT_TOKENS)
    num_experts_per_rank = num_experts // num_ranks
    expert_lo = rank_idx * num_experts_per_rank

    # All ranks draw every expert from one seed, so each rank can build the full reference for its own tokens.
    # Weights are scaled so `|gate|` reaches several units and SiTU's tanh saturation is visible.
    torch.manual_seed(1234)
    l1_all = torch.randn((num_experts, 2 * inter, hidden), dtype=torch.bfloat16, device='cuda') * (2 / hidden ** 0.5)
    l2_all = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda') * (1 / inter ** 0.5)

    def cast_fp4(w):
        packed = torch.empty((w.shape[0], w.shape[1], w.shape[2] // 2), dtype=torch.int8, device='cuda')
        sf = torch.empty((w.shape[0], w.shape[1], w.shape[2] // 32), dtype=torch.float, device='cuda')
        for i in range(w.shape[0]):
            packed[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        deq = torch.stack([cast_back_from_fp4(packed[i], sf[i], gran_k=32) for i in range(w.shape[0])])
        return packed, sf, deq

    l1_q, l1_sf, l1_deq = cast_fp4(l1_all)
    l2_q, l2_sf, l2_deq = cast_fp4(l2_all)
    del l1_all, l2_all
    local = slice(expert_lo, expert_lo + num_experts_per_rank)
    l1_local = (l1_q[local].contiguous(), deep_gemm.transform_sf_into_required_layout(
        l1_sf[local].contiguous(), 2 * inter, hidden, (1, 32), num_experts_per_rank))
    l2_local = (l2_q[local].contiguous(), deep_gemm.transform_sf_into_required_layout(
        l2_sf[local].contiguous(), hidden, inter, (1, 32), num_experts_per_rank))

    torch.manual_seed(100 + rank_idx)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores.softmax(dim=-1), num_topk, dim=-1, sorted=False)
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    x_deq = cast_back_from_fp8(x_fp8, x_sf, gran_k=32, use_packed_ue8m0=True)

    def run(activation, **kwargs):
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, num_tokens, num_topk, hidden, inter,
            use_fp8_dispatch=True, activation=activation)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_local, l2_local, activation=activation)
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer, recipe=(1, 1, 32), activation=activation,
                                   activation_clamp=None, fast_math=True, **kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        buffer.destroy()
        return y

    def reference(activation, **situ_kwargs):
        return _mega_moe_reference(x_deq, topk_idx, topk_weights, l1_deq, l2_deq, activation, **situ_kwargs)

    tolerance = 2e-4
    swiglu_ref = reference('swiglu')
    for activation in ('swiglu', 'geglu'):
        # Unset SiTU parameters leave SwiGLU / GeGLU exactly as before; SiTU parameters are ignored by them
        y = run(activation)
        diff = calc_diff(y, reference(activation))
        assert diff < tolerance, f'{activation}: {diff=}'
        assert torch.equal(y, run(activation, situ_beta=None, situ_linear_beta=None)), activation
        assert torch.equal(y, run(activation, situ_beta=4.0, situ_linear_beta=25.0)), activation
        dist_print(f' > {activation}: diff vs reference {diff:.2e}', once_in_node=True)

    for situ_beta, situ_linear_beta in ((4.0, 25.0), (4.0, None), (2.0, 3.0)):
        y = run('situ', situ_beta=situ_beta, situ_linear_beta=situ_linear_beta)
        diff = calc_diff(y, reference('situ', situ_beta=situ_beta, situ_linear_beta=situ_linear_beta))
        swiglu_diff = calc_diff(y, swiglu_ref)
        assert diff < tolerance, f'situ({situ_beta}, {situ_linear_beta}): {diff=}'
        # Teeth: the SiTU output must be far from SwiGLU's, so the check above is not vacuous
        assert swiglu_diff > 50 * tolerance, f'situ({situ_beta}, {situ_linear_beta}) ~ swiglu: {swiglu_diff=}'
        dist_print(f' > situ(beta={situ_beta}, linear_beta={situ_linear_beta}): diff vs reference {diff:.2e}, '
                   f'vs swiglu {swiglu_diff:.2e}', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()


def _spawn_activation_test(num_ranks: int) -> None:
    os.environ.setdefault('MASTER_PORT', str(29500 + random.randint(0, 999)))
    torch.multiprocessing.spawn(_activation_worker, args=(num_ranks,), nprocs=num_ranks)


@test_filter(lambda: get_arch_major() == 10)
def test_mega_moe_gated_activations() -> None:
    _spawn_activation_test(1)


@test_filter(lambda: get_arch_major() == 10 and torch.cuda.device_count() >= 2)
def test_mega_moe_gated_activations_ep2() -> None:
    _spawn_activation_test(2)


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
    parser.add_argument('--num-shared-experts', type=int, default=1, help='Number of shared experts (use 0 to disable)')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--mma-type', type=str, default='fp8xfp4', choices=('fp8xfp4', 'fp8xfp8', 'bf16xbf16'))

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
