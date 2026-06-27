import os
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import (
    align, per_token_cast_to_nvfp4, cast_back_from_nvfp4, nvfp4_global_scale,
)
from deep_gemm.utils.layout import get_tma_aligned_size

GRAN_K = 16


def _rt(x, gs):
    p, s = per_token_cast_to_nvfp4(x, gs, gran_k=GRAN_K)
    return cast_back_from_nvfp4(p, s, gs, gran_k=GRAN_K).to(x.dtype)


def _mn_major_packed_e4m3(sf_bytes):
    g, mn, k = sf_bytes.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((g, aligned_mn, aligned_k), dtype=torch.uint8, device=sf_bytes.device)
    padded[:, :mn, :k] = sf_bytes
    padded = padded.reshape(-1).view(torch.int32).view(g, aligned_mn, aligned_k // 4)
    out = torch.empty_strided((g, aligned_mn, aligned_k // 4),
                              (aligned_mn * (aligned_k // 4), 1, aligned_mn),
                              dtype=torch.int32, device=sf_bytes.device)
    return out.copy_(padded)[:, :mn]


def _cast_l1_w(l1w):
    # l1w: [E, inter*2, hidden]. gate = rows [:inter], up = rows [inter:]; per-expert global scales.
    g, n2, k = l1w.shape
    inter = n2 // 2
    wp = torch.empty((g, n2, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n2, k // GRAN_K), device='cuda', dtype=torch.uint8)
    gate_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    up_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    for e in range(g):
        ggs = nvfp4_global_scale(l1w[e][:inter]); ugs = nvfp4_global_scale(l1w[e][inter:])
        gate_gs[e], up_gs[e] = ggs, ugs
        wp[e][:inter], wsf[e][:inter] = per_token_cast_to_nvfp4(l1w[e][:inter], ggs, gran_k=GRAN_K)
        wp[e][inter:], wsf[e][inter:] = per_token_cast_to_nvfp4(l1w[e][inter:], ugs, gran_k=GRAN_K)
    return (wp, _mn_major_packed_e4m3(wsf)), gate_gs, up_gs


def _cast_l2_w(l2w):
    g, n, k = l2w.shape
    wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n, k // GRAN_K), device='cuda', dtype=torch.uint8)
    down_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    for e in range(g):
        dgs = nvfp4_global_scale(l2w[e]); down_gs[e] = dgs
        wp[e], wsf[e] = per_token_cast_to_nvfp4(l2w[e], dgs, gran_k=GRAN_K)
    return (wp, _mn_major_packed_e4m3(wsf)), down_gs


def _swiglu(l1, inter, weight, clamp):
    gate = l1[:inter].clamp(max=clamp)
    up = l1[inter:].clamp(min=-clamp, max=clamp)
    return (gate * torch.sigmoid(gate)) * up * weight


def _estimate_l2act_gs(x, l1w, topk_idx, topk_weights, inter, gs_x, gate_gs, up_gs, clamp, num_experts):
    # Per-expert L2-input (intermediate) global scale (my convention: amax/(6*448)).
    x_deq = _rt(x, gs_x)
    w1_deq = [torch.cat([_rt(l1w[e][:inter], gate_gs[e].item()), _rt(l1w[e][inter:], up_gs[e].item())])
              for e in range(num_experts)]
    amax = torch.full((num_experts,), 1e-6, device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            act = _swiglu(l1, inter, float(topk_weights[t, k].item()), clamp)
            amax[e] = torch.maximum(amax[e], act.abs().amax())
    return amax / (6.0 * 448.0)


def reference_nvfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter,
                        gs_x, gate_gs, up_gs, down_gs, l2act_gs, clamp):
    num_tokens, num_experts = x.shape[0], l1w.shape[0]
    x_deq = _rt(x, gs_x)
    w1_deq = [torch.cat([_rt(l1w[e][:inter], gate_gs[e].item()), _rt(l1w[e][inter:], up_gs[e].item())])
              for e in range(num_experts)]
    w2_deq = [_rt(l2w[e], down_gs[e].item()) for e in range(num_experts)]
    y = torch.zeros((num_tokens, hidden), dtype=torch.float, device=x.device)
    for t in range(num_tokens):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            act = _swiglu(l1, inter, float(topk_weights[t, k].item()), clamp).to(torch.bfloat16)
            act_deq = _rt(act.unsqueeze(0), l2act_gs[e].item()).squeeze(0)
            y[t] += act_deq.float() @ w2_deq[e].float().T
    return y.to(torch.bfloat16)


def test_nvfp4_mega_moe():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12400')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    num_max_tokens, num_tokens = 128, 128
    hidden, inter = 512, 512
    num_experts, num_topk = 8, 2
    clamp = 10.0

    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                 hidden, inter, mma_type='nvfp4xnvfp4')

    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1w = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / (hidden ** 0.5)
    l2w = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda') / (inter ** 0.5)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)

    # Per-tensor input global scale (my convention amax/(6*448)); per-expert weight global scales
    gs_x = nvfp4_global_scale(x)
    l1, gate_gs, up_gs = _cast_l1_w(l1w)
    l2, down_gs = _cast_l2_w(l2w)
    l2act_gs = _estimate_l2act_gs(x, l1w, topk_idx, topk_weights, inter, gs_x, gate_gs, up_gs, clamp, num_experts)

    # Per-expert kernel params (TRT-LLM convention):
    #   alpha = 1/(input_gs_trt * weight_gs_trt) = my_gs_input * my_gs_weight
    #   l2_input_global_scale (TRT) = 1 / my_gs
    gate_alpha = (gs_x * gate_gs).contiguous()
    up_alpha = (gs_x * up_gs).contiguous()
    down_alpha = (l2act_gs * down_gs).contiguous()
    l2_input_global_scale = (1.0 / l2act_gs).contiguous()

    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    xp, xsf = per_token_cast_to_nvfp4(x, gs_x, gran_k=GRAN_K)

    buf.x[:num_tokens].copy_(xp)
    buf.x_sf[:num_tokens].copy_(xsf.contiguous().view(torch.int32))
    buf.topk_idx[:num_tokens].copy_(topk_idx)
    buf.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                   gate_alpha=gate_alpha, up_alpha=up_alpha,
                                   l2_input_global_scale=l2_input_global_scale, down_alpha=down_alpha,
                                   activation_clamp=clamp, fast_math=True)
    torch.cuda.synchronize()

    ref = reference_nvfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter,
                              gs_x, gate_gs, up_gs, down_gs, l2act_gs, clamp)
    diff = calc_diff(y, ref)
    print(f'diff = {diff:.5f}  (y~{y.float().abs().mean():.3f}, ref~{ref.float().abs().mean():.3f})')
    assert diff < 0.05, f'{diff=}'
    print('NVFP4 mega MoE passed.')
    dist.destroy_process_group()


if __name__ == '__main__':
    test_nvfp4_mega_moe()
