"""Performance benchmarks for the packed-FP4 paths (MXFP4 vs NVFP4).

Covers the standalone GEMM (`mxfp4_gemm_nt` / `nvfp4_gemm_nt`) and the
mega-MoE (`mxfp4_mxfp4_mega_moe` / `nvfp4_nvfp4_mega_moe`).
"""
import os
import sys
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import bench_kineto
from deep_gemm.utils.math import (
    per_token_cast_to_fp4, per_token_cast_to_nvfp4, nvfp4_global_scale,
)

sys.path.insert(0, os.path.dirname(__file__))
from test_mxfp4_gemm import _prepare as _prep_mxfp4_gemm
from test_nvfp4_gemm import _prepare as _prep_nvfp4_gemm
from test_nvfp4_mega_moe import _cast_l1_w, _cast_l2_w, _estimate_l2act_gs, GRAN_K


def bench_gemm():
    print('=== Standalone packed-FP4 GEMM (2-CTA de-risk kernel) ===')
    print(f'{"M":>6} {"N":>6} {"K":>6} | {"mxfp4 us":>9} {"TFLOPS":>8} | {"nvfp4 us":>9} {"TFLOPS":>8} | nv/mx')
    for m, n, k in ((4096, 4096, 4096), (4096, 4096, 8192), (8192, 8192, 8192), (2048, 4096, 16384)):
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        flops = 2.0 * m * n * k

        ap, asf, _ = _prep_mxfp4_gemm(a)
        bp, bsf, _ = _prep_mxfp4_gemm(b)
        t_mx = bench_kineto(lambda: deep_gemm.mxfp4_gemm_nt((ap, asf), (bp, bsf), d),
                            'mxfp4_gemm', suppress_kineto_output=True)

        gsa, gsb = nvfp4_global_scale(a), nvfp4_global_scale(b)
        ap2, asf2, _ = _prep_nvfp4_gemm(a, gsa)
        bp2, bsf2, _ = _prep_nvfp4_gemm(b, gsb)
        t_nv = bench_kineto(lambda: deep_gemm.nvfp4_gemm_nt((ap2, asf2), (bp2, bsf2), d, a_global_scale=gsa, b_global_scale=gsb),
                            'mxfp4_gemm', suppress_kineto_output=True)

        print(f'{m:>6} {n:>6} {k:>6} | {t_mx*1e6:>9.1f} {flops/t_mx/1e12:>8.1f} | '
              f'{t_nv*1e6:>9.1f} {flops/t_nv/1e12:>8.1f} | {t_nv/t_mx:>4.2f}x')
    print()


def _cast_w_mxfp4(w):
    g, n, k = w.shape
    wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(g):
        wp[i], wsf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    return wp, deep_gemm.transform_sf_into_required_layout(wsf, n, k, (1, 32), g)


def bench_mega():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12533')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD
    clamp = 10.0

    print('=== Packed-FP4 mega-MoE (single rank) ===')
    print(f'{"tok":>5} {"exp":>4} {"topk":>4} {"hid":>5} {"int":>5} | {"mxfp4 us":>9} | {"nvfp4 us":>9} | nv/mx')
    torch.manual_seed(0)
    for num_tokens, num_experts, num_topk, hidden, inter in (
        (128, 8, 2, 2048, 2048),
        (512, 8, 2, 2048, 2048),
        (1024, 32, 4, 4096, 1536),
    ):
        num_max_tokens = max(128, num_tokens)
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1w = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / (hidden ** 0.5)
        l2w = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda') / (inter ** 0.5)
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

        # MXFP4
        buf_mx = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                        hidden, inter, mma_type='mxfp4xmxfp4')
        xp, xsf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(_cast_w_mxfp4(l1w), _cast_w_mxfp4(l2w))
        buf_mx.x[:num_tokens].copy_(xp); buf_mx.x_sf[:num_tokens].copy_(xsf)
        buf_mx.topk_idx[:num_tokens].copy_(topk_idx); buf_mx.topk_weights[:num_tokens].copy_(topk_weights)
        t_mx = bench_kineto(lambda: deep_gemm.mxfp4_mxfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf_mx,
                                                                  activation_clamp=clamp, fast_math=True),
                            'mega_moe', suppress_kineto_output=True)
        buf_mx.destroy()

        # NVFP4 (per-expert global scales, TRT-LLM convention)
        buf_nv = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                        hidden, inter, mma_type='nvfp4xnvfp4')
        gs_x = nvfp4_global_scale(x)
        l1, gate_gs, up_gs = _cast_l1_w(l1w)
        l2, down_gs = _cast_l2_w(l2w)
        l2act_gs = _estimate_l2act_gs(x, l1w, topk_idx, topk_weights, inter, gs_x, gate_gs, up_gs, clamp, num_experts)
        gate_alpha = (gs_x * gate_gs).contiguous(); up_alpha = (gs_x * up_gs).contiguous()
        down_alpha = (l2act_gs * down_gs).contiguous(); l2_input_gs = (1.0 / l2act_gs).contiguous()
        xpn, xsfn = per_token_cast_to_nvfp4(x, gs_x, gran_k=GRAN_K)
        nl1, nl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
        buf_nv.x[:num_tokens].copy_(xpn); buf_nv.x_sf[:num_tokens].copy_(xsfn.contiguous().view(torch.int32))
        buf_nv.topk_idx[:num_tokens].copy_(topk_idx); buf_nv.topk_weights[:num_tokens].copy_(topk_weights)
        t_nv = bench_kineto(lambda: deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=nl1, l2_weights=nl2, sym_buffer=buf_nv,
                                                                  gate_alpha=gate_alpha, up_alpha=up_alpha,
                                                                  l2_input_global_scale=l2_input_gs, down_alpha=down_alpha,
                                                                  activation_clamp=clamp, fast_math=True),
                            'mega_moe', suppress_kineto_output=True)
        buf_nv.destroy()

        print(f'{num_tokens:>5} {num_experts:>4} {num_topk:>4} {hidden:>5} {inter:>5} | '
              f'{t_mx*1e6:>9.1f} | {t_nv*1e6:>9.1f} | {t_nv/t_mx:>4.2f}x')
    dist.destroy_process_group()
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    bench_gemm()
    bench_mega()
