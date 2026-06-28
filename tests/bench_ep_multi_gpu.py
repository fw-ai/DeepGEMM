"""Multi-GPU expert-parallel (EP) MoE benchmark: DeepGEMM mega-MoE vs FlashInfer NVFP4.

Layout: total_experts=512 sharded across `num_processes` GPUs (512/N experts per
shard); each shard holds `num_tokens` local tokens; top_k routing.

  * DeepGEMM `nvfp4_nvfp4_mega_moe`: native EP kernel (dispatch + grouped GEMM +
    combine fused via symmetric memory). Timed end-to-end (comm is internal).

  * FlashInfer `cute_dsl_fused_moe_nvfp4`: single-device kernel, benchmarked with
    the user-specified recipe per iteration:
        1) barrier + cuda sync across all devices
        2) start timer
        3) moe (local shard: 512/N experts, num_tokens tokens, top_k)
        4) all_reduce the output across all devices (combine proxy)
        5) stop timer
    so its number includes the cross-device communication cost.

We report the worst-rank average latency (max over ranks).

Run:
    python tests/bench_ep_multi_gpu.py --num-processes 4
    python tests/bench_ep_multi_gpu.py --num-processes 8
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

import deep_gemm
from deep_gemm.testing import bench_kineto
from deep_gemm.utils.dist import init_dist, dist_print
from deep_gemm.utils.math import per_token_cast_to_nvfp4, nvfp4_global_scale, per_token_cast_to_fp8

sys.path.insert(0, os.path.dirname(__file__))
from test_nvfp4_mega_moe import _cast_l1_w, _cast_l2_w, GRAN_K
from bench_flashinfer_vs_deepgemm import build_flashinfer, build_flashinfer_cutlass
from fi_trtllm import build_flashinfer_trtllm
from bench_packed_fp4 import _cast_w_mxfp4

HIDDEN, INTER = 4608, 2560
TOTAL_EXPERTS, TOP_K = 512, 16
CLAMP = 10.0


def rank_avg_ms(per_iter_ms, group):
    """Aggregate a per-rank per-iter device time (ms) across GPUs by AVERAGING
    (mean over ranks). Each rank's value is its own pure device time from the
    trace; we just collect and average them."""
    world = dist.get_world_size(group)
    t = torch.tensor([per_iter_ms], device='cuda', dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return t.item() / world


def _self_dev_us(evt):
    return getattr(evt, 'self_device_time_total', None) or getattr(evt, 'self_cuda_time_total', 0.0)


def kernel_rows(fn, group, warmup=10, iters=50):
    """Return per-kernel (name, launches/iter, us/iter) device-time rows for `fn`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        dev = getattr(evt, 'self_device_time_total', None) or getattr(evt, 'self_cuda_time_total', 0.0)
        if dev and dev > 0:
            rows.append((evt.key, evt.count / iters, dev / iters))
    rows.sort(key=lambda r: -r[2])
    return rows


def allreduce_time(buf, group, warmup=10, iters=50):
    """Device time of the all_reduce NCCL kernel, with ~10ms sleep alignment so the
    collective isn't inflated by cross-rank launch skew. Returns worst-rank avg ms."""
    fn = lambda: dist.all_reduce(buf, group=group)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            torch.cuda._sleep(int(2e7))  # ~10ms: align launches across ranks
            fn()
        torch.cuda.synchronize()
    us, name = 0.0, ''
    for evt in prof.key_averages():
        if 'allreduce' in evt.key.lower() or 'nccl' in evt.key.lower():
            us += (getattr(evt, 'self_device_time_total', None) or getattr(evt, 'self_cuda_time_total', 0.0))
            name = evt.key
    return rank_avg_ms(us / iters / 1e3, group), name


def ep_total_device_time(step_fn, group, warmup=10, iters=50):
    """Profile the full EP path in ONE window (steps 2->6: dispatch + slice + moe +
    combine + all_reduce, all inside step_fn) and SUM the device time of every GPU
    kernel. A dist sync is done once at the beginning to align ranks. Returns the
    avg-over-ranks per-iter device time (ms)."""
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    dist.barrier(group)  # sync at the beginning
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step_fn()
        torch.cuda.synchronize()
    total_us = sum((getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0.0))
                   for e in prof.key_averages())
    return rank_avg_ms(total_us / iters / 1e3, group)


def device_time_sum(fn, group, warmup=10, iters=50):
    """Sum the self device time of every GPU kernel in the trace (per iter).
    Used for the local FlashInfer MoE + its short combine all_reduce (no hard
    cross-rank data dependency, so no launch alignment needed). NCCL/all_reduce
    kernels (the combine) ARE counted. Returns worst-rank avg in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = sum(_self_dev_us(evt) for evt in prof.key_averages())
    return rank_avg_ms(total_us / iters / 1e3, group)


def build_deepgemm_ep(group, x_bf16, sel, rw, local_experts):
    num_tokens = x_bf16.shape[0]
    num_max = 128
    w1 = torch.randn(local_experts, INTER * 2, HIDDEN, dtype=torch.bfloat16, device='cuda') / 10
    w2 = torch.randn(local_experts, HIDDEN, INTER, dtype=torch.bfloat16, device='cuda') / 10
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, TOTAL_EXPERTS, num_max, TOP_K,
                                                 HIDDEN, INTER, mma_type='nvfp4xnvfp4')
    gs_x = nvfp4_global_scale(x_bf16)
    l1, gate_gs, up_gs = _cast_l1_w(w1)
    l2, down_gs = _cast_l2_w(w2)
    l2act_gs = torch.ones(local_experts, device='cuda', dtype=torch.float32)
    gate_alpha = (gs_x * gate_gs).contiguous()
    up_alpha = (gs_x * up_gs).contiguous()
    down_alpha = (l2act_gs * down_gs).contiguous()
    l2_input_gs = (1.0 / l2act_gs).contiguous()
    nl1, nl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    xpn, xsfn = per_token_cast_to_nvfp4(x_bf16, gs_x, gran_k=GRAN_K)
    buf.x[:num_tokens].copy_(xpn)
    buf.x_sf[:num_tokens].copy_(xsfn.contiguous().view(torch.int32))
    buf.topk_idx[:num_tokens].copy_(sel)
    buf.topk_weights[:num_tokens].copy_(rw)
    y = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=nl1, l2_weights=nl2, sym_buffer=buf,
                                       gate_alpha=gate_alpha, up_alpha=up_alpha,
                                       l2_input_global_scale=l2_input_gs, down_alpha=down_alpha,
                                       activation_clamp=CLAMP, fast_math=True)
    return run, buf


def build_deepgemm_fp8_ep(group, x_bf16, sel, rw, local_experts):
    num_tokens = x_bf16.shape[0]
    num_max = 128
    w1 = torch.randn(local_experts, INTER * 2, HIDDEN, dtype=torch.bfloat16, device='cuda') / 10
    w2 = torch.randn(local_experts, HIDDEN, INTER, dtype=torch.bfloat16, device='cuda') / 10
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, TOTAL_EXPERTS, num_max, TOP_K,
                                                 HIDDEN, INTER, mma_type='fp8xfp4')
    xp8, xsf8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(_cast_w_mxfp4(w1), _cast_w_mxfp4(w2))
    buf.x[:num_tokens].copy_(xp8)
    buf.x_sf[:num_tokens].copy_(xsf8)
    buf.topk_idx[:num_tokens].copy_(sel)
    buf.topk_weights[:num_tokens].copy_(rw)
    y = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                   activation_clamp=CLAMP, fast_math=True)
    return run, buf


def worker(local_rank, num_local_ranks, args):
    global INTER
    INTER = args.inter
    rank, world, group = init_dist(local_rank, num_local_ranks)
    local_experts = TOTAL_EXPERTS // world
    local_offset = rank * local_experts
    # Global batch = num_tokens (per-rank for DeepGEMM dispatch) x world.
    dg_tokens = args.num_tokens                 # DeepGEMM: tokens per rank (dispatch model)
    global_tokens = dg_tokens * world           # FlashInfer: full batch replicated on every GPU
    tk = min(TOP_K, local_experts)

    results = {}

    def mega_kineto(run):
        dg_s = bench_kineto(run, 'mega_moe', suppress_kineto_output=True,
                            barrier=lambda: dist.barrier(group))
        return rank_avg_ms(dg_s * 1e3, group)

    # ---- DeepGEMM native EP mega kernel (dispatch+GEMM+combine fused), dg_tokens/rank ----
    torch.manual_seed(rank)
    x_dg = torch.randn(dg_tokens, HIDDEN, dtype=torch.bfloat16, device='cuda') / 10
    rw_dg, sel_dg = torch.topk(F.softmax(torch.randn(dg_tokens, TOTAL_EXPERTS, device='cuda'), -1), TOP_K, dim=-1)
    rw_dg = (rw_dg / rw_dg.sum(-1, keepdim=True)).float()
    sel_dg = sel_dg.to(torch.int64)
    for key, builder in (('dg_nvfp4', build_deepgemm_ep), ('dg_fp8', build_deepgemm_fp8_ep)):
        try:
            run, buf = builder(group, x_dg, sel_dg, rw_dg, local_experts)
            results[key] = mega_kineto(run)
            buf.destroy()
        except Exception as ex:
            results[key] = None
            if rank == 0:
                print(f'  {key} err: {str(ex).splitlines()[-1][:120]}')

    # ---- FlashInfer: replicated input + expert-shard + all_reduce combine ----
    # Every GPU holds the SAME full batch (replicated) and the SAME global routing;
    # each GPU owns experts [local_offset : local_offset+local_experts] and computes
    # ONLY their contributions (via local_expert_offset / ep_rank). all_reduce then
    # sums the partial [global_tokens, hidden] outputs into the full result.
    torch.manual_seed(12345)  # identical inputs/routing on all ranks (replicated)
    x_rep = torch.randn(global_tokens, HIDDEN, dtype=torch.bfloat16, device='cuda') / 10
    rw_g, sel_g = torch.topk(F.softmax(torch.randn(global_tokens, TOTAL_EXPERTS, device='cuda'), -1), TOP_K, dim=-1)
    rw_g = (rw_g / rw_g.sum(-1, keepdim=True)).float()
    sel_g = sel_g.to(torch.int64)
    # Local expert weight slice (random values; only shapes/positions matter for timing).
    torch.manual_seed(1000 + rank)
    w1 = torch.randn(local_experts, INTER * 2, HIDDEN, dtype=torch.bfloat16, device='cuda') / 10
    w2 = torch.randn(local_experts, HIDDEN, INTER, dtype=torch.bfloat16, device='cuda') / 10

    # all_reduce-only device time (combine of the full [global_tokens, hidden] output)
    # Allocate the combine buffer from NCCL's registered (multicast) allocator when
    # available, so NCCL can dispatch the NVLS all_reduce kernel (plain torch tensors
    # fall back to RING+LL even when NVLS is forced).
    ar_buf, ar_pool, ar_be = None, None, None
    dev_t = torch.device('cuda', torch.cuda.current_device())
    try:
        ar_be = group._get_backend(dev_t)
        ar_pool = torch.cuda.MemPool(ar_be.mem_allocator)
        with torch.cuda.use_mem_pool(ar_pool):
            ar_buf = torch.empty(global_tokens, HIDDEN, dtype=torch.bfloat16, device=dev_t)
        ar_be.register_mem_pool(ar_pool, symm=True)  # symm=True -> NVLS multicast registration
    except Exception as ex:
        if rank == 0:
            print(f'  (NVLS-registered buffer failed, using plain tensor: {str(ex).splitlines()[-1][:80]})')
        ar_buf, ar_pool = torch.empty(global_tokens, HIDDEN, dtype=torch.bfloat16, device='cuda'), None
    results['allreduce'], ar_kname = allreduce_time(ar_buf, group)

    # GeGLU (gelu-gated) where supported; cute_dsl and DeepGEMM are SwiGLU-only.
    import flashinfer
    geglu = flashinfer.ActivationType.Geglu

    def run_fi(name):
        if name == 'fi_cutedsl':  # SwiGLU-only (no activation param)
            return build_flashinfer(x_rep, w1, w2, sel_g, rw_g, TOTAL_EXPERTS, tk, HIDDEN, INTER,
                                    local_expert_offset=local_offset)
        if name == 'fi_cutlass':
            return build_flashinfer_cutlass(x_rep, w1, w2, sel_g, rw_g, local_experts, tk, HIDDEN, INTER,
                                            ep_size=world, ep_rank=rank, activation_type=geglu)
        return build_flashinfer_trtllm(x_rep, w1, w2, TOTAL_EXPERTS, tk, HIDDEN, INTER,
                                       local_expert_offset=local_offset, activation_type=geglu)

    for name in ('fi_cutedsl', 'fi_cutlass', 'fi_trtllm'):
        try:
            fi_run = run_fi(name)
            results[name + '_moe'] = device_time_sum(fi_run, group)  # moe only (breakdown)
            # Full path (steps 2->6) in one profiling window: moe + NVLS all_reduce combine.
            def step(fi_run=fi_run):
                fi_run()
                dist.all_reduce(ar_buf, group=group)
            results[name + '_total'] = ep_total_device_time(step, group)
        except Exception as ex:
            results[name + '_moe'] = results[name + '_total'] = None
            if rank == 0:
                print(f'  {name} err: {str(ex).splitlines()[-1][:120]}')

    if rank == 0:
        def us(v):
            return f'{v*1e3:7.1f}' if v is not None else f'{"n/a":>7}'
        ar = results.get('allreduce')
        m_e = global_tokens * TOP_K / TOTAL_EXPERTS
        ar_kib = global_tokens * HIDDEN * 2 / 1024
        print(f'\n=== EP MoE (device time from trace): {world} GPUs | total_experts={TOTAL_EXPERTS} | '
              f'{local_experts} experts/gpu | global batch={global_tokens} tokens '
              f'(DeepGEMM {dg_tokens}/rank dispatch; FlashInfer replicated+expert-sharded) | '
              f'~{m_e:g} tokens/expert | hidden={HIDDEN} inter={INTER} top_k={TOP_K} ===')
        print(f'  {"kernel":<34} {"moe":>8} {"allreduce":>10} {"sum":>8} {"total(2-6)":>11}  (us)')
        print(f'  {"DeepGEMM fp8xfp4 mega (fused EP)":<34} {us(results.get("dg_fp8")):>8} '
              f'{"fused":>10} {us(results.get("dg_fp8")):>8} {us(results.get("dg_fp8")):>11}')
        print(f'  {"DeepGEMM nvfp4   mega (fused EP)":<34} {us(results.get("dg_nvfp4")):>8} '
              f'{"fused":>10} {us(results.get("dg_nvfp4")):>8} {us(results.get("dg_nvfp4")):>11}')
        for name, lbl in (('fi_cutedsl', 'FlashInfer nvfp4 cute_dsl (SwiGLU)'),
                          ('fi_cutlass', 'FlashInfer nvfp4 cutlass (GeGLU)'),
                          ('fi_trtllm', 'FlashInfer nvfp4 trtllm-gen (GeGLU)')):
            moe = results.get(name + '_moe')
            tot_sum = (moe + ar) if (moe is not None and ar is not None) else None  # separate-measure sum
            tot_win = results.get(name + '_total')                                  # one-window sum (2-6)
            print(f'  {lbl:<34} {us(moe):>8} {us(ar):>10} {us(tot_sum):>8} {us(tot_win):>11}')
        print(f'\n  all_reduce combine ([{global_tokens}, {HIDDEN}] bf16, {ar_kib:.0f} KiB, sleep-aligned, '
              f'NCCL_ALGO={os.environ.get("NCCL_ALGO", "auto")}): {us(ar)} us  {ar_kname[:72]}')
        print('  total(2-6) = one profiling window: dispatch+slice+moe+combine+all_reduce, summed device time')

    # Clean up the NVLS-registered combine pool before teardown (all ranks).
    if ar_pool is not None:
        try:
            ar_be.deregister_mem_pool(ar_pool)
        except Exception:
            pass
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-processes', type=int, default=4)
    p.add_argument('--num-tokens', type=int, default=32)
    p.add_argument('--inter', type=int, default=2560,
                   help='intermediate size; fp8xfp4 needs a multiple of 512 (2304 is nvfp4-only)')
    p.add_argument('--nccl-algo', type=str, default='',
                   help='force NCCL_ALGO for the combine all_reduce, e.g. NVLS, Ring, Tree')
    args = p.parse_args()
    if args.nccl_algo:
        os.environ['NCCL_ALGO'] = args.nccl_algo
        if 'NVLS' in args.nccl_algo.upper():
            os.environ.setdefault('NCCL_NVLS_ENABLE', '1')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12577')
    torch.multiprocessing.spawn(worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    main()
