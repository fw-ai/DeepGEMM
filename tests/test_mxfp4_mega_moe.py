import os
import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import per_token_cast_to_fp4, cast_back_from_fp4


def _fp4_roundtrip(x: torch.Tensor, gran_k: int = 32) -> torch.Tensor:
    """Quantize bf16 -> packed MXFP4 (UE8M0, gran-32) and dequantize back to the
    exact values the kernel operates on."""
    packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k)
    return cast_back_from_fp4(packed, sf, gran_k=gran_k).to(x.dtype)


def reference_mxfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter,
                        activation_clamp: float = 10.0):
    """Single-rank MXFP4 MoE reference matching the fused kernel's math:
       FP4(x) @ FP4(W1).T -> SwiGLU*weight -> FP4 requant -> @ FP4(W2).T -> combine (sum over top-k)."""
    num_tokens = x.shape[0]
    x_deq = _fp4_roundtrip(x)
    w1_deq = torch.stack([_fp4_roundtrip(l1w[e]) for e in range(l1w.shape[0])])
    w2_deq = torch.stack([_fp4_roundtrip(l2w[e]) for e in range(l2w.shape[0])])

    y = torch.zeros((num_tokens, hidden), dtype=torch.float, device=x.device)
    for t in range(num_tokens):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            if e < 0:
                continue
            l1 = x_deq[t].float() @ w1_deq[e].float().T          # [inter*2]
            gate, up = l1[:inter], l1[inter:]
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(min=-activation_clamp, max=activation_clamp)
            act = (gate * torch.sigmoid(gate)) * up * float(topk_weights[t, k].item())
            act_deq = _fp4_roundtrip(act.to(torch.bfloat16).unsqueeze(0)).squeeze(0)
            l2 = act_deq.float() @ w2_deq[e].float().T            # [hidden]
            y[t] += l2
    return y.to(torch.bfloat16)


def test_mxfp4_mega_moe():
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '12399')
    dist.init_process_group('nccl', rank=0, world_size=1)
    group = dist.group.WORLD

    num_max_tokens, num_tokens = 128, 128
    hidden, inter = 512, 512
    num_experts, num_topk = 8, 2
    ne_per_rank = num_experts

    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                 hidden, inter, mma_type='mxfp4xmxfp4')

    def cast_w(w):
        g, n, k = w.shape
        wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
        wsf = torch.empty((g, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(g):
            wp[i], wsf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        return wp, deep_gemm.transform_sf_into_required_layout(wsf, n, k, (1, 32), g)

    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1w = torch.randn((ne_per_rank, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / (hidden ** 0.5)
    l2w = torch.randn((ne_per_rank, hidden, inter), dtype=torch.bfloat16, device='cuda') / (inter ** 0.5)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)

    xp, xsf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    l1 = cast_w(l1w)
    l2 = cast_w(l2w)
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

    buf.x[:num_tokens].copy_(xp)
    buf.x_sf[:num_tokens].copy_(xsf)
    buf.topk_idx[:num_tokens].copy_(topk_idx)
    buf.topk_weights[:num_tokens].copy_(topk_weights)

    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.mxfp4_mxfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=buf,
                                   activation_clamp=10.0, fast_math=True)
    torch.cuda.synchronize()

    ref = reference_mxfp4_moe(x, l1w, l2w, topk_idx, topk_weights, hidden, inter, 10.0)
    diff = calc_diff(y, ref)
    print(f'diff = {diff:.5f}  (y~{y.float().abs().mean():.3f}, ref~{ref.float().abs().mean():.3f})')
    assert diff < 0.05, f'{diff=}'
    print('MXFP4 mega MoE passed.')
    dist.destroy_process_group()


if __name__ == '__main__':
    test_mxfp4_mega_moe()
