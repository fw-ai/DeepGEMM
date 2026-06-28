"""Single-device perf comparison:

  DeepGEMM `fp8_fp4_mega_moe` (FP8 E4M3 activations x FP4 weights)
      vs
  FlashInfer `cute_dsl_fused_moe_nvfp4` (NVFP4 activations x NVFP4 weights)

Both run the *same* MoE problem (identical shapes + identical routing) so the
comparison is apples-to-apples at the op level. We report end-to-end device
time (CUDA events), which includes all sub-kernels (FlashInfer dispatches
moe_sort + gemm1 + gemm2/finalize; DeepGEMM is a single mega kernel).
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F

import deep_gemm
from deep_gemm.utils.math import (
    per_token_cast_to_fp8, per_token_cast_to_nvfp4, nvfp4_global_scale,
)

sys.path.insert(0, os.path.dirname(__file__))
from bench_packed_fp4 import _cast_w_mxfp4
from test_nvfp4_mega_moe import _cast_l1_w, _cast_l2_w, GRAN_K

import flashinfer
from flashinfer import fused_moe
from flashinfer.fused_moe import cute_dsl_fused_moe_nvfp4
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

from fi_trtllm import build_flashinfer_trtllm

SF_VEC = 16
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
_round_up = lambda x, y: (x + y - 1) // y * y


def build_flashinfer_cutlass(x_bf16, w1_bf16, w2_bf16, sel, rw, num_experts, top_k, hidden, inter,
                             ep_size=1, ep_rank=0, activation_type=None):
    """FlashInfer CUTLASS NVFP4 fused MoE (`cutlass_fused_moe`), prep per the
    reference test. `num_experts` here is the LOCAL expert count (w1 groups);
    EP slicing is handled via ep_size/ep_rank with global `sel` in [0, local*ep_size)."""
    dev = x_bf16.device
    e, n, k = num_experts, inter, hidden
    w1_n = 2 * n
    w1_q = torch.empty((e, w1_n, k // 2), device=dev, dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device=dev, dtype=torch.uint8)
    w1_bs = torch.empty((e, _round_up(w1_n, 128), _round_up(k // 16, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w2_bs = torch.empty((e, _round_up(k, 128), _round_up(n // 16, 4)), device=dev, dtype=torch.float8_e4m3fn)
    w1_gs = torch.empty(e, device=dev, dtype=torch.float32)
    w2_gs = torch.empty(e, device=dev, dtype=torch.float32)
    for ex in range(e):
        w1_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_bf16[ex].abs().max().float()
        w2_gs[ex] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_bf16[ex].abs().max().float()
        w1_q[ex], w1_bs[ex] = fp4_quantize(w1_bf16[ex], w1_gs[ex])
        w2_q[ex], w2_bs[ex] = fp4_quantize(w2_bf16[ex], w2_gs[ex])
    a1_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device=dev, dtype=torch.float32)
    quant_scales = [a1_gs, w1_bs.view(torch.int32), 1.0 / (a1_gs * w1_gs),
                    a2_gs, w2_bs.view(torch.int32), 1.0 / (a2_gs * w2_gs)]
    hidden_states, input_sf = fp4_quantize(x_bf16, a1_gs)
    out = torch.zeros_like(x_bf16)
    w1l = w1_q.contiguous().view(torch.long)
    w2l = w2_q.contiguous().view(torch.long)
    seli = sel.to(torch.int32)
    act = activation_type if activation_type is not None else flashinfer.ActivationType.Swiglu

    def run():
        return fused_moe.cutlass_fused_moe(hidden_states, seli, rw, w1l, w2l,
                                           torch.bfloat16, quant_scales=quant_scales,
                                           input_sf=input_sf, output=out,
                                           ep_size=ep_size, ep_rank=ep_rank,
                                           activation_type=act)
    return run


def _time_events(launch, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds


def cuda_time(fn, warmup=10, iters=50):
    """Pure device time via CUDA-graph replay (eliminates CPU dispatch/autotuner
    overhead). Falls back to eager event timing if capture is unsupported."""
    # Warm up first so any JIT/autotuning happens before capture.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(warmup):
            g.replay()
        return _time_events(g.replay, iters), 'graph'
    except Exception:
        for _ in range(warmup):
            fn()
        return _time_events(fn, iters), 'eager'


def _interleave_gate(x, group_size=64, dim=1):
    sizes = x.size()
    dim = dim % x.dim()
    x = x.view(*sizes[:dim], 2, sizes[dim] // (group_size * 2), group_size, *sizes[dim + 1:])
    return x.transpose(dim, dim + 1).contiguous().view(*sizes)


def build_flashinfer(x_bf16, w1_bf16, w2_bf16, sel, rw, num_experts, top_k, hidden, inter,
                     local_expert_offset=0):
    """cute_dsl NVFP4 MoE. `num_experts` is the GLOBAL expert count (routing space);
    `w1_bf16`/`w2_bf16` hold only this rank's local expert slice. For single device,
    num_experts == w1.shape[0] and local_expert_offset == 0."""
    dev = x_bf16.device
    g = torch.tensor([1.0], device=dev, dtype=torch.float32)
    e = w1_bf16.shape[0]  # local experts

    xq, xsf = fp4_quantize(x_bf16, global_scale=g, sf_vec_size=SF_VEC, is_sf_swizzled_layout=False)
    xsf = xsf.unsqueeze(-1)

    fc1_rows = 2 * inter
    w1_il = _interleave_gate(w1_bf16, group_size=64, dim=1)
    w1q, w1sf = fp4_quantize(w1_il.reshape(e * fc1_rows, hidden), global_scale=g,
                             sf_vec_size=SF_VEC, is_sf_swizzled_layout=True)
    w1q = w1q.view(e, fc1_rows, hidden // 2)
    w1sf = convert_sf_to_mma_layout(w1sf, m=fc1_rows, k=hidden, num_groups=e, sf_vec_size=SF_VEC)

    w2q, w2sf = fp4_quantize(w2_bf16.view(e * hidden, inter), global_scale=g,
                             sf_vec_size=SF_VEC, is_sf_swizzled_layout=True)
    w2q = w2q.view(e, hidden, inter // 2)
    w2sf = convert_sf_to_mma_layout(w2sf, m=hidden, k=inter, num_groups=e, sf_vec_size=SF_VEC)

    ones = torch.ones(e, device=dev, dtype=torch.float32)
    fc2_in = torch.tensor([1.0], device=dev, dtype=torch.float32)

    def run():
        return cute_dsl_fused_moe_nvfp4(
            x=xq, x_sf=xsf, token_selected_experts=sel.to(torch.int32),
            token_final_scales=rw, w1_weight=w1q, w1_weight_sf=w1sf, w1_alpha=ones,
            fc2_input_scale=fc2_in, w2_weight=w2q, w2_weight_sf=w2sf, w2_alpha=ones,
            num_experts=num_experts, top_k=top_k,
            num_local_experts=e, local_expert_offset=local_expert_offset)
    return run


def build_deepgemm(group, x_bf16, w1_bf16, w2_bf16, sel, rw, num_experts, top_k, hidden, inter):
    num_tokens = x_bf16.shape[0]
    num_max = max(128, num_tokens)
    clamp = 10.0
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max, top_k,
                                                 hidden, inter, mma_type='fp8xfp4')
    xp8, xsf8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(_cast_w_mxfp4(w1_bf16), _cast_w_mxfp4(w2_bf16))
    buf.x[:num_tokens].copy_(xp8)
    buf.x_sf[:num_tokens].copy_(xsf8)
    buf.topk_idx[:num_tokens].copy_(sel)
    buf.topk_weights[:num_tokens].copy_(rw)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.fp8_fp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                   activation_clamp=clamp, fast_math=True)
    return run, buf


def build_deepgemm_nvfp4(group, x_bf16, w1_bf16, w2_bf16, sel, rw, num_experts, top_k, hidden, inter):
    num_tokens = x_bf16.shape[0]
    num_max = max(128, num_tokens)
    clamp = 10.0
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max, top_k,
                                                 hidden, inter, mma_type='nvfp4xnvfp4')
    gs_x = nvfp4_global_scale(x_bf16)
    l1, gate_gs, up_gs = _cast_l1_w(w1_bf16)
    l2, down_gs = _cast_l2_w(w2_bf16)
    # Per-expert global scales: values don't affect timing, so skip the costly
    # per-token L2-activation amax estimation and use unit L2-input scale.
    l2act_gs = torch.ones(num_experts, device='cuda', dtype=torch.float32)
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
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run():
        deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=nl1, l2_weights=nl2, sym_buffer=buf,
                                       gate_alpha=gate_alpha, up_alpha=up_alpha,
                                       l2_input_global_scale=l2_input_gs, down_alpha=down_alpha,
                                       activation_clamp=clamp, fast_math=True)
    return run, buf


def bench_one(group, num_tokens, hidden, inter, num_experts, top_k):
    torch.manual_seed(0)
    dev = 'cuda'
    x_bf16 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev) / 10
    w1_bf16 = torch.randn(num_experts, inter * 2, hidden, dtype=torch.bfloat16, device=dev) / 10
    w2_bf16 = torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16, device=dev) / 10

    logits = torch.randn(num_tokens, num_experts, device=dev)
    probs = F.softmax(logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(probs, top_k, dim=-1)
    rw = (rw / rw.sum(-1, keepdim=True)).float()
    sel = sel.to(torch.int64)

    res = {}

    def try_run(name, builder):
        try:
            out = builder()
            run = out[0] if isinstance(out, tuple) else out
            res[name], res[name + '_m'] = cuda_time(run)
            if isinstance(out, tuple):
                out[1].destroy()
        except Exception as ex:
            res[name] = None
            res[name + '_e'] = str(ex).splitlines()[-1][:90]

    try_run('dg8', lambda: build_deepgemm(group, x_bf16, w1_bf16, w2_bf16, sel, rw,
                                          num_experts, top_k, hidden, inter))
    try_run('dg4', lambda: build_deepgemm_nvfp4(group, x_bf16, w1_bf16, w2_bf16, sel, rw,
                                                num_experts, top_k, hidden, inter))
    try_run('fi_cutedsl', lambda: build_flashinfer(x_bf16, w1_bf16, w2_bf16, sel, rw,
                                                   num_experts, top_k, hidden, inter))
    try_run('fi_cutlass', lambda: build_flashinfer_cutlass(x_bf16, w1_bf16, w2_bf16, sel, rw,
                                                           num_experts, top_k, hidden, inter))
    try_run('fi_trtllm', lambda: build_flashinfer_trtllm(x_bf16, w1_bf16, w2_bf16,
                                                         num_experts, top_k, hidden, inter))

    def f(k):
        if res.get(k) is None:
            return f'{"n/a":>9}'
        return f'{res[k]*1e6:>8.1f}{res.get(k + "_m", "")[0]}'  # suffix g/e

    print(f'{num_tokens:>5} {num_experts:>4} {top_k:>3} {hidden:>5} {inter:>5} | '
          f'{f("dg8")} | {f("dg4")} | {f("fi_cutedsl")} | {f("fi_cutlass")} | {f("fi_trtllm")}')
    for k in ('dg8', 'dg4', 'fi_cutedsl', 'fi_cutlass', 'fi_trtllm'):
        if res.get(k + '_e'):
            print(f'      {k} err: {res[k + "_e"]}')


def main():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12566')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    print(f'torch {torch.__version__}, flashinfer {flashinfer.__version__}, {torch.cuda.get_device_name(0)}')
    print('DeepGEMM mega-MoE (fp8xfp4 / nvfp4)  vs  FlashInfer NVFP4 MoE backends  (device us, CUDA graph; g=graph e=eager)')
    print(f'{"tok":>5} {"exp":>4} {"tk":>3} {"hid":>5} {"int":>5} | {"dg_fp8":>9} | {"dg_nvfp4":>9} | '
          f'{"fi_cutedsl":>9} | {"fi_cutlass":>9} | {"fi_trtllm":>9}')
    for cfg in (
        (128, 2048, 2048, 32, 4),
        (512, 2048, 2048, 32, 4),
        (1024, 7168, 2560, 256, 8),
        # inter=2304 is nvfp4-only (fp8xfp4 needs inter % 512 == 0)
        (32, 4608, 2304, 256, 16),
    ):
        bench_one(group, *cfg)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
