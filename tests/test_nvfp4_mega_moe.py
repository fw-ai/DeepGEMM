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


def _nvfp4_roundtrip(x: torch.Tensor, gs: float) -> torch.Tensor:
    """Quantize bf16 -> NVFP4 (E4M3 SF gran-16 + global scale) and dequantize back to the
    exact values the kernel operates on."""
    packed, sf = per_token_cast_to_nvfp4(x, gs, gran_k=GRAN_K)
    return cast_back_from_nvfp4(packed, sf, gs, gran_k=GRAN_K).to(x.dtype)


def _mn_major_packed_e4m3(sf_bytes: torch.Tensor) -> torch.Tensor:
    """Replicate `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch` for raw E4M3 bytes:
    pad to TMA-aligned MN and K%4, pack 4 SF bytes -> int32, transpose to MN-major."""
    g, mn, k = sf_bytes.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((g, aligned_mn, aligned_k), dtype=torch.uint8, device=sf_bytes.device)
    padded[:, :mn, :k] = sf_bytes
    padded = padded.reshape(-1).view(torch.int32).view(g, aligned_mn, aligned_k // 4)
    out = torch.empty_strided((g, aligned_mn, aligned_k // 4),
                              (aligned_mn * (aligned_k // 4), 1, aligned_mn),
                              dtype=torch.int32, device=sf_bytes.device)
    out = out.copy_(padded)[:, :mn]
    return out


def _cast_w_nvfp4(w: torch.Tensor, gs: float):
    g, n, k = w.shape
    wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n, k // GRAN_K), device='cuda', dtype=torch.uint8)
    for i in range(g):
        wp[i], wsf[i] = per_token_cast_to_nvfp4(w[i], gs, gran_k=GRAN_K)
    return wp, _mn_major_packed_e4m3(wsf)


def reference_nvfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter,
                        gs_x, gs_w1, gs_w2, gs_l2, activation_clamp: float = 10.0):
    num_tokens = x.shape[0]
    x_deq = _nvfp4_roundtrip(x, gs_x)
    w1_deq = torch.stack([_nvfp4_roundtrip(l1w[e], gs_w1) for e in range(l1w.shape[0])])
    w2_deq = torch.stack([_nvfp4_roundtrip(l2w[e], gs_w2) for e in range(l2w.shape[0])])

    y = torch.zeros((num_tokens, hidden), dtype=torch.float, device=x.device)
    for t in range(num_tokens):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            if e < 0:
                continue
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            gate, up = l1[:inter], l1[inter:]
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(min=-activation_clamp, max=activation_clamp)
            act = (gate * torch.sigmoid(gate)) * up * float(topk_weights[t, k].item())
            act_deq = _nvfp4_roundtrip(act.to(torch.bfloat16).unsqueeze(0), gs_l2).squeeze(0)
            l2 = act_deq.float() @ w2_deq[e].float().T
            y[t] += l2
    return y.to(torch.bfloat16)


def _estimate_l2_global_scale(x, l1w, topk_idx, topk_weights, inter, gs_x, gs_w1, clamp):
    # The L1-output (L2-input) global scale is a CPU param; estimate it from the activation amax.
    x_deq = _nvfp4_roundtrip(x, gs_x)
    w1_deq = torch.stack([_nvfp4_roundtrip(l1w[e], gs_w1) for e in range(l1w.shape[0])])
    amax = torch.tensor(1e-6, device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            gate, up = l1[:inter].clamp(max=clamp), l1[inter:].clamp(min=-clamp, max=clamp)
            act = (gate * torch.sigmoid(gate)) * up * float(topk_weights[t, k].item())
            amax = torch.maximum(amax, act.abs().amax())
    return float(amax / (6.0 * 448.0))


def test_nvfp4_mega_moe():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12400')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    num_max_tokens, num_tokens = 128, 128
    hidden, inter = 512, 512
    num_experts, num_topk = 8, 2
    ne_per_rank = num_experts
    clamp = 10.0

    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                 hidden, inter, mma_type='nvfp4xnvfp4')

    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1w = torch.randn((ne_per_rank, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / (hidden ** 0.5)
    l2w = torch.randn((ne_per_rank, hidden, inter), dtype=torch.bfloat16, device='cuda') / (inter ** 0.5)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)

    # Per-tensor global scales (CPU scalars)
    gs_x = nvfp4_global_scale(x)
    gs_w1 = nvfp4_global_scale(l1w)
    gs_w2 = nvfp4_global_scale(l2w)
    gs_l2 = _estimate_l2_global_scale(x, l1w, topk_idx, topk_weights, inter, gs_x, gs_w1, clamp)

    # Quantize inputs
    xp, xsf = per_token_cast_to_nvfp4(x, gs_x, gran_k=GRAN_K)
    xsf_packed = xsf.contiguous().view(torch.int32)  # K-major int32 (4 E4M3 bytes / int32)
    l1 = _cast_w_nvfp4(l1w, gs_w1)
    l2 = _cast_w_nvfp4(l2w, gs_w2)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

    buf.x[:num_tokens].copy_(xp)
    buf.x_sf[:num_tokens].copy_(xsf_packed)
    buf.topk_idx[:num_tokens].copy_(topk_idx)
    buf.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                   l1_act_global_scale=gs_x, l2_act_global_scale=gs_l2,
                                   l1_weight_global_scale=gs_w1, l2_weight_global_scale=gs_w2,
                                   activation_clamp=clamp, fast_math=True)
    torch.cuda.synchronize()

    ref = reference_nvfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter,
                              gs_x, gs_w1, gs_w2, gs_l2, clamp)
    diff = calc_diff(y, ref)
    print(f'diff = {diff:.5f}  (y~{y.float().abs().mean():.3f}, ref~{ref.float().abs().mean():.3f})')
    assert diff < 0.05, f'{diff=}'
    print('NVFP4 mega MoE passed.')
    dist.destroy_process_group()


if __name__ == '__main__':
    test_nvfp4_mega_moe()
