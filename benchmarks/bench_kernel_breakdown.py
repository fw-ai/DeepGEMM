"""Per-kernel device-time breakdown for each MoE backend (single device).

Profiles each backend's run closure and lists every GPU kernel with its
per-iteration count and device time, so you can see how many kernels each
backend launches and where the time goes.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

import deep_gemm

sys.path.insert(0, os.path.dirname(__file__))
from bench_flashinfer_vs_deepgemm import (
    build_deepgemm, build_deepgemm_nvfp4, build_flashinfer, build_flashinfer_cutlass,
)
from fi_trtllm import build_flashinfer_trtllm


def kernel_breakdown(run, warmup=10, iters=50):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
        dev = getattr(evt, 'self_device_time_total', None) or getattr(evt, 'self_cuda_time_total', 0.0)
        if dev and dev > 0:
            rows.append((evt.key, evt.count / iters, dev / iters))  # name, launches/iter, us/iter
    rows.sort(key=lambda r: -r[2])
    return rows


def show(title, build, *, is_tuple=False):
    print(f'\n===== {title} =====')
    try:
        out = build()
        run = out[0] if is_tuple else out
        rows = kernel_breakdown(run)
        if is_tuple:
            out[1].destroy()
    except Exception as ex:
        print(f'  ERROR: {str(ex).splitlines()[-1][:120]}')
        return
    total = sum(r[2] for r in rows)
    n_launches = sum(r[1] for r in rows)
    print(f'  {len(rows)} distinct kernels, {n_launches:.0f} launches/iter, total device {total:.1f} us/iter')
    print(f'  {"#/it":>5} {"us/it":>8}  kernel')
    for name, cnt, us in rows:
        print(f'  {cnt:>5.0f} {us:>8.1f}  {name[:88]}')


def main():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12599')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    # 4-GPU-shard view of the previous shape: 128 experts, 32 tokens, top-16, m_e=4.
    num_tokens, hidden, inter, num_experts, top_k = 32, 4608, 2304, 128, 16
    print(f'torch {torch.__version__}, flashinfer {__import__("flashinfer").__version__}, '
          f'{torch.cuda.get_device_name(0)}')
    print(f'shape: tokens={num_tokens} experts={num_experts} top_k={top_k} hidden={hidden} inter={inter}')

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda') / 10
    w1 = torch.randn(num_experts, inter * 2, hidden, dtype=torch.bfloat16, device='cuda') / 10
    w2 = torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device='cuda') / 10
    rw, sel = torch.topk(F.softmax(torch.randn(num_tokens, num_experts, device='cuda'), -1), top_k, dim=-1)
    rw = (rw / rw.sum(-1, keepdim=True)).float()
    sel = sel.to(torch.int64)

    show('DeepGEMM fp8xfp4 mega (fused EP)',
         lambda: build_deepgemm(group, x, w1, w2, sel, rw, num_experts, top_k, hidden, inter), is_tuple=True)
    show('DeepGEMM nvfp4 mega (fused EP)',
         lambda: build_deepgemm_nvfp4(group, x, w1, w2, sel, rw, num_experts, top_k, hidden, inter), is_tuple=True)
    show('FlashInfer nvfp4 (cute_dsl)',
         lambda: build_flashinfer(x, w1, w2, sel, rw, num_experts, top_k, hidden, inter))
    show('FlashInfer nvfp4 (cutlass)',
         lambda: build_flashinfer_cutlass(x, w1, w2, sel, rw, num_experts, top_k, hidden, inter))
    show('FlashInfer nvfp4 (trtllm-gen)',
         lambda: build_flashinfer_trtllm(x, w1, w2, num_experts, top_k, hidden, inter))

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
