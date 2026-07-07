import argparse
import os
import re
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.dist import init_dist, dist_print
from deep_gemm.utils.layout import get_tma_aligned_size
from deep_gemm.utils.math import (
    align,
    per_token_cast_to_fp4,
    cast_back_from_fp4,
    per_token_cast_to_nvfp4,
    cast_back_from_nvfp4,
    nvfp4_global_scale,
)

DIFF_TOL = 0.05


def capture_num_cycles(run_fn):
    """Run ``run_fn`` (which launches the mega-MoE kernel), capture the kernel's
    ``[mega_moe] rank=%u num_cycles=%u ...`` printf, and return ``(num_cycles, result, raw)``.

    CUDA printf writes to the C-level stdout (fd 1), not Python's ``sys.stdout``, so we
    redirect fd 1 to a temp file around the launch + ``torch.cuda.synchronize`` (the CUDA
    printf fifo is flushed at kernel exit / sync). ``result`` is ``run_fn``'s return value.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mega_moe.log')
    os.close(tmp_fd)
    saved_stdout = os.dup(1)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_TRUNC)
    os.dup2(fd, 1)
    result = None
    try:
        result = run_fn()
        torch.cuda.synchronize()
    finally:
        os.dup2(saved_stdout, 1)
        os.close(fd)
        os.close(saved_stdout)
    with open(tmp_path) as f:
        raw = f.read()
    os.unlink(tmp_path)
    m = re.search(r'num_cycles=(\d+)', raw)
    return (int(m.group(1)) if m else None), result, raw

MXFP4_GRAN_K = 32
NVFP4_GRAN_K = 16
CLAMP = 10.0

# Kernel constraints (verified in csrc/jit_kernels/impls/sm100_mxfp4_mxfp4_mega_moe.hpp
# and deep_gemm/include/deep_gemm/layout/mega_moe.cuh):
#   - block_n = 128  -> hidden % 128 == 0 and inter % 128 == 0
#   - num_max_tokens_per_rank is aligned up to kLCMCandidateBlockM = 384 internally;
#     num_tokens may be any value <= num_max_tokens_per_rank (partial M-blocks ok)
#   - num_experts % num_ranks == 0  (expert-parallel sharding)
#   - topk_idx entries may be -1 (masked / dropped selections)
assert_msg = 'hidden/inter must be multiples of 128 (block_n=128)'


def _swiglu(l1: torch.Tensor, inter: int, weight: float, clamp: float) -> torch.Tensor:
    gate = l1[:inter].clamp(max=clamp)
    up = l1[inter:].clamp(min=-clamp, max=clamp)
    return (gate * torch.sigmoid(gate)) * up * weight


def _mn_major_packed_e4m3_3d(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[g, mn, k_sf] uint8 (E4M3 bytes) -> MN-major TMA-aligned int32 (4 SF bytes/int32)."""
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


# --------------------------------------------------------------------------- #
# MXFP4 strategy
# --------------------------------------------------------------------------- #

def _fp4_roundtrip_mxfp4(x: torch.Tensor, gran_k: int = MXFP4_GRAN_K) -> torch.Tensor:
    packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k)
    return cast_back_from_fp4(packed, sf, gran_k=gran_k).to(x.dtype)


def _cast_w_mxfp4(w: torch.Tensor):
    """w: [g, n, k] (a local expert shard) -> (packed_e2m1, UE8M0 SF in kernel layout)."""
    g, n, k = w.shape
    wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n, k // MXFP4_GRAN_K), device='cuda', dtype=torch.float)
    for i in range(g):
        wp[i], wsf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=MXFP4_GRAN_K)
    return wp, deep_gemm.transform_sf_into_required_layout(wsf, n, k, (1, MXFP4_GRAN_K), g)


def _run_mxfp4(s) -> torch.Tensor:
    lo, le = s.local_offset, s.local_experts
    xp, xsf = per_token_cast_to_fp4(s.x, use_ue8m0=True, gran_k=MXFP4_GRAN_K, use_packed_ue8m0=True)
    l1 = _cast_w_mxfp4(s.l1w[lo:lo + le])
    l2 = _cast_w_mxfp4(s.l2w[lo:lo + le])
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

    s.buf.x[:s.num_tokens].copy_(xp)
    s.buf.x_sf[:s.num_tokens].copy_(xsf)
    s.buf.topk_idx[:s.num_tokens].copy_(s.topk_idx)
    s.buf.topk_weights[:s.num_tokens].copy_(s.topk_weights)

    y = torch.empty((s.num_tokens, s.hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.mxfp4_mxfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=s.buf,
                                   activation_clamp=s.clamp, fast_math=True)
    torch.cuda.synchronize()
    return y


def _reference_mxfp4(s) -> torch.Tensor:
    """Full MoE reference over ALL experts (the kernel's combine produces the full
    per-rank output, so the reference ignores EP sharding)."""
    x_deq = _fp4_roundtrip_mxfp4(s.x)
    w1_deq = torch.stack([_fp4_roundtrip_mxfp4(s.l1w[e]) for e in range(s.num_experts)])
    w2_deq = torch.stack([_fp4_roundtrip_mxfp4(s.l2w[e]) for e in range(s.num_experts)])

    y = torch.zeros((s.num_tokens, s.hidden), dtype=torch.float, device=s.x.device)
    for t in range(s.num_tokens):
        for k in range(s.topk_idx.shape[1]):
            e = int(s.topk_idx[t, k].item())
            if e < 0:
                continue
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            act = _swiglu(l1, s.inter, float(s.topk_weights[t, k].item()), s.clamp)
            act_deq = _fp4_roundtrip_mxfp4(act.to(torch.bfloat16).unsqueeze(0)).squeeze(0)
            y[t] += act_deq.float() @ w2_deq[e].float().T
    return y.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# NVFP4 strategy
# --------------------------------------------------------------------------- #

def _rt_nvfp4(x: torch.Tensor, gs) -> torch.Tensor:
    p, sb = per_token_cast_to_nvfp4(x, gs, gran_k=NVFP4_GRAN_K)
    return cast_back_from_nvfp4(p, sb, gs, gran_k=NVFP4_GRAN_K).to(x.dtype)


def _cast_l1_w_nvfp4(l1w: torch.Tensor):
    """l1w: [g, inter*2, hidden] (a local expert shard). gate = rows [:inter], up = [inter:]."""
    g, n2, k = l1w.shape
    inter = n2 // 2
    wp = torch.empty((g, n2, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n2, k // NVFP4_GRAN_K), device='cuda', dtype=torch.uint8)
    gate_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    up_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    for e in range(g):
        ggs = nvfp4_global_scale(l1w[e][:inter])
        ugs = nvfp4_global_scale(l1w[e][inter:])
        gate_gs[e], up_gs[e] = ggs, ugs
        wp[e][:inter], wsf[e][:inter] = per_token_cast_to_nvfp4(l1w[e][:inter], ggs, gran_k=NVFP4_GRAN_K)
        wp[e][inter:], wsf[e][inter:] = per_token_cast_to_nvfp4(l1w[e][inter:], ugs, gran_k=NVFP4_GRAN_K)
    return (wp, _mn_major_packed_e4m3_3d(wsf)), gate_gs, up_gs


def _cast_l2_w_nvfp4(l2w: torch.Tensor):
    g, n, k = l2w.shape
    wp = torch.empty((g, n, k // 2), device='cuda', dtype=torch.int8)
    wsf = torch.empty((g, n, k // NVFP4_GRAN_K), device='cuda', dtype=torch.uint8)
    down_gs = torch.empty(g, device='cuda', dtype=torch.float32)
    for e in range(g):
        dgs = nvfp4_global_scale(l2w[e])
        down_gs[e] = dgs
        wp[e], wsf[e] = per_token_cast_to_nvfp4(l2w[e], dgs, gran_k=NVFP4_GRAN_K)
    return (wp, _mn_major_packed_e4m3_3d(wsf)), down_gs


def _estimate_l2act_gs(x, l1w, topk_idx, topk_weights, inter, gs_x, gate_gs, up_gs,
                       clamp, num_experts) -> torch.Tensor:
    # Per-expert L2-input (intermediate) global scale (convention: amax/(6*448)).
    x_deq = _rt_nvfp4(x, gs_x)
    w1_deq = [torch.cat([_rt_nvfp4(l1w[e][:inter], gate_gs[e].item()),
                         _rt_nvfp4(l1w[e][inter:], up_gs[e].item())])
              for e in range(num_experts)]
    amax = torch.full((num_experts,), 1e-6, device=x.device)
    for t in range(x.shape[0]):
        for k in range(topk_idx.shape[1]):
            e = int(topk_idx[t, k].item())
            if e < 0:
                continue
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            act = _swiglu(l1, inter, float(topk_weights[t, k].item()), clamp)
            amax[e] = torch.maximum(amax[e], act.abs().amax())
    return amax / (6.0 * 448.0)


def _run_nvfp4(s) -> torch.Tensor:
    s.gs_x = nvfp4_global_scale(s.x)
    lo, le = s.local_offset, s.local_experts

    # Full per-expert global scales (reference uses ALL experts; kernel uses the local slice).
    # nvfp4_global_scale returns a python float, so build the tensors explicitly.
    s.gate_gs = torch.tensor([nvfp4_global_scale(s.l1w[e][:s.inter]) for e in range(s.num_experts)],
                             dtype=torch.float32, device='cuda')
    s.up_gs = torch.tensor([nvfp4_global_scale(s.l1w[e][s.inter:]) for e in range(s.num_experts)],
                           dtype=torch.float32, device='cuda')
    s.down_gs = torch.tensor([nvfp4_global_scale(s.l2w[e]) for e in range(s.num_experts)],
                             dtype=torch.float32, device='cuda')
    s.l2act_gs = _estimate_l2act_gs(s.x, s.l1w, s.topk_idx, s.topk_weights, s.inter,
                                    s.gs_x, s.gate_gs, s.up_gs, s.clamp, s.num_experts)

    # Local-shard weight cast + per-local-expert alphas (TRT-LLM convention).
    l1_local, _gate_gs_local, _up_gs_local = _cast_l1_w_nvfp4(s.l1w[lo:lo + le])
    l2_local, _down_gs_local = _cast_l2_w_nvfp4(s.l2w[lo:lo + le])
    gate_gs_local = s.gate_gs[lo:lo + le].contiguous()
    up_gs_local = s.up_gs[lo:lo + le].contiguous()
    down_gs_local = s.down_gs[lo:lo + le].contiguous()
    l2act_local = s.l2act_gs[lo:lo + le].contiguous()
    gate_alpha = (s.gs_x * gate_gs_local).contiguous()
    up_alpha = (s.gs_x * up_gs_local).contiguous()
    down_alpha = (l2act_local * down_gs_local).contiguous()
    l2_input_global_scale = (1.0 / l2act_local).contiguous()

    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1_local, l2_local)
    xp, xsf = per_token_cast_to_nvfp4(s.x, s.gs_x, gran_k=NVFP4_GRAN_K)

    s.buf.x[:s.num_tokens].copy_(xp)
    s.buf.x_sf[:s.num_tokens].copy_(xsf.contiguous().view(torch.int32))
    s.buf.topk_idx[:s.num_tokens].copy_(s.topk_idx)
    s.buf.topk_weights[:s.num_tokens].copy_(s.topk_weights)

    y = torch.empty((s.num_tokens, s.hidden), dtype=torch.bfloat16, device='cuda')
    deep_gemm.nvfp4_nvfp4_mega_moe(y=y, l1_weights=tl1, l2_weights=tl2, sym_buffer=s.buf,
                                   gate_alpha=gate_alpha, up_alpha=up_alpha,
                                   l2_input_global_scale=l2_input_global_scale, down_alpha=down_alpha,
                                   activation_clamp=s.clamp, fast_math=True)
    torch.cuda.synchronize()
    return y


def _reference_nvfp4(s) -> torch.Tensor:
    x_deq = _rt_nvfp4(s.x, s.gs_x)
    w1_deq = [torch.cat([_rt_nvfp4(s.l1w[e][:s.inter], s.gate_gs[e].item()),
                         _rt_nvfp4(s.l1w[e][s.inter:], s.up_gs[e].item())])
              for e in range(s.num_experts)]
    w2_deq = [_rt_nvfp4(s.l2w[e], s.down_gs[e].item()) for e in range(s.num_experts)]
    y = torch.zeros((s.num_tokens, s.hidden), dtype=torch.float, device=s.x.device)
    for t in range(s.num_tokens):
        for k in range(s.topk_idx.shape[1]):
            e = int(s.topk_idx[t, k].item())
            if e < 0:
                continue
            l1 = x_deq[t].float() @ w1_deq[e].float().T
            act = _swiglu(l1, s.inter, float(s.topk_weights[t, k].item()), s.clamp).to(torch.bfloat16)
            act_deq = _rt_nvfp4(act.unsqueeze(0), s.l2act_gs[e].item()).squeeze(0)
            y[t] += act_deq.float() @ w2_deq[e].float().T
    return y.to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Per-format strategy table
# --------------------------------------------------------------------------- #

FP4_MOE = {
    'mxfp4': SimpleNamespace(mma_type='mxfp4xmxfp4', run=_run_mxfp4, reference=_reference_mxfp4),
    'nvfp4': SimpleNamespace(mma_type='nvfp4xnvfp4', run=_run_nvfp4, reference=_reference_nvfp4),
}

# Baseline single-shape topology (matches the original pre-consolidation test).
BASELINE_SHAPE = (128, 8, 2, 512, 512, 0.0)

# 1-rank coverage: (num_tokens, num_experts, num_topk, hidden, inter, masked_ratio).
# Covers small / odd-token / single-token / large-asymmetric / masked routing.
SHAPES_1RANK = [
    (32, 8, 2, 512, 512, 0.0),       # small
    (128, 8, 2, 512, 512, 0.0),      # baseline
    (96, 8, 2, 512, 512, 0.0),       # odd token count (partial M-block)
    (1, 8, 2, 512, 512, 0.0),        # single token (smallest block_m path)
    (128, 8, 2, 2048, 2048, 0.0),    # larger hidden/inter
    (512, 8, 2, 2048, 2048, 0.0),    # more tokens
    (1024, 32, 4, 4096, 1536, 0.0),  # large + asymmetric + more experts
    (128, 8, 2, 512, 512, 0.3),      # masked routing
]

# Multi-rank EP coverage: (world, total_experts, num_topk, num_tokens, hidden, inter, masked_ratio).
# Tokens/weights/routing are REPLICATED across ranks (seed=0) so the NVFP4 per-expert
# l2act_gs reference is computable without a cross-rank all-gather; the cross-rank
# dispatch+combine path is still exercised (tokens leave to the expert-owning rank).
SHAPES_MULTIRANK = [
    (2, 8, 2, 128, 512, 512, 0.0),
    (4, 8, 2, 128, 512, 512, 0.0),
    (8, 8, 2, 128, 512, 512, 0.0),
    (2, 32, 4, 512, 2048, 2048, 0.0),
    (4, 32, 4, 512, 2048, 2048, 0.0),
    (8, 32, 4, 1024, 4096, 1536, 0.0),
    (2, 8, 2, 128, 512, 512, 0.3),   # masked EP
]

# Prefill (cycle-chunked) coverage: sweep chunk_ratio across [0.5, 1, 1.5, 2, 5] for each
# shape. When `chunk_ratio * num_max >= num_ranks * num_max` (ratio >= world), the ring is
# >= num_min and the kernel runs a single cycle (num_cycles == 1); when ratio < world, the
# ring is < num_min and the kernel runs num_cycles >= 2 dispatch rounds. We assert both the
# correctness (diff < DIFF_TOL) and the num_cycles behavior (captured from the kernel printf).
PREFILL_RATIOS = [0.5, 1.0, 1.5, 2.0, 5.0, 'auto']
PREFILL_1RANK_SHAPES = [
    (1024, 8, 2, 512, 512, 0.0),
    (1024, 32, 4, 4096, 1536, 0.0),
    (512, 8, 2, 2048, 2048, 0.0),
]
# (world, total_experts, num_topk, num_tokens, hidden, inter, masked_ratio)
PREFILL_MULTIRANK_SHAPES = [
    (2, 8, 2, 1024, 512, 512, 0.0),
    (4, 32, 4, 1024, 4096, 1536, 0.0),
    (8, 32, 4, 1024, 4096, 1536, 0.0),
]


def _build_s(spec, shape, rank, world, group, chunk_ratio=None) -> SimpleNamespace:
    num_tokens, num_experts, num_topk, hidden, inter, masked_ratio = shape
    assert num_experts % world == 0, f'num_experts={num_experts} not divisible by world={world}'
    assert hidden % 128 == 0 and inter % 128 == 0, assert_msg
    local_experts = num_experts // world
    local_offset = rank * local_experts

    # num_max_tokens is aligned up to 384 internally; num_tokens may be any value <= it.
    num_max_tokens = max(num_tokens, 1)
    buf = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk,
                                                 hidden, inter, mma_type=spec.mma_type,
                                                 chunk_ratio=chunk_ratio)

    # Replicated inputs across ranks (same seed) -> every rank computes the same full
    # reference; each rank passes only its local expert shard to the kernel.
    torch.manual_seed(0)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1w = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda') / (hidden ** 0.5)
    l2w = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda') / (inter ** 0.5)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx = topk_idx.masked_fill(rand_mask < masked_ratio, -1)
        topk_weights = topk_weights.masked_fill(topk_idx < 0, 0)

    return SimpleNamespace(
        num_tokens=num_tokens, num_experts=num_experts, num_topk=num_topk,
        hidden=hidden, inter=inter, clamp=CLAMP,
        local_experts=local_experts, local_offset=local_offset,
        x=x, l1w=l1w, l2w=l2w, topk_idx=topk_idx, topk_weights=topk_weights,
        buf=buf,
    )


def _run_one(spec, shape, rank, world, group, chunk_ratio=None):
    """Build the shape, run the kernel + reference, return (y, ref, diff)."""
    s = _build_s(spec, shape, rank, world, group, chunk_ratio=chunk_ratio)
    try:
        y = spec.run(s)
        ref = spec.reference(s)
        return y, ref, calc_diff(y, ref), s
    except Exception:
        s.buf.destroy()
        raise


def _shape_tag(shape, world=None):
    if world is None:
        tok, exp, topk, h, i, m = shape
        return f'1rank tok={tok} E={exp} topk={topk} h={h} inter={i} mask={m}'
    _w, exp, topk, tok, h, i, m = shape
    return f'EP{world} tok={tok} E={exp} topk={topk} h={h} inter={i} mask={m}'


def _check_and_print(label, fmt, diff, y, ref):
    """Print (only on local_rank 0) and assert on ALL ranks. ``dist_print`` barriers
    on every rank, so all ranks must call this — never gate it behind a rank check."""
    ok = diff < DIFF_TOL
    dist_print(f'[{label} {fmt}] diff={diff:.5f} '
               f'(y~{y.float().abs().mean():.3f}, ref~{ref.float().abs().mean():.3f}) '
               f'{"OK" if ok else "FAIL"}', once_in_node=True)
    assert ok, f'{fmt=}, {label}, diff={diff:.5f}'


def _ensure_master(port: int):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = str(port)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #

def test_fp4_mega_moe(fmt: str = 'mxfp4') -> None:
    """Single-shape 1-rank smoke test (baseline topology). Kept for backward compat."""
    assert fmt in FP4_MOE, f'unknown FP4 format {fmt!r}; expected one of {list(FP4_MOE)}'
    _ensure_master(13799)
    rank, world, group = init_dist(0, 1)
    spec = FP4_MOE[fmt]
    try:
        y, ref, diff, s = _run_one(spec, BASELINE_SHAPE, rank, world, group)
        try:
            print(f'[{fmt}] diff = {diff:.5f}  '
                  f'(y~{y.float().abs().mean():.3f}, ref~{ref.float().abs().mean():.3f})')
            assert diff < DIFF_TOL, f'{fmt=}, {diff=}'
            print(f'{fmt.upper()} mega MoE passed.')
        finally:
            s.buf.destroy()
    finally:
        dist.destroy_process_group()


def test_fp4_mega_moe_1rank(fmt: str = 'mxfp4') -> None:
    """Run the full 1-rank shape matrix for one format."""
    assert fmt in FP4_MOE, f'unknown FP4 format {fmt!r}; expected one of {list(FP4_MOE)}'
    _ensure_master(13799)
    rank, world, group = init_dist(0, 1)
    spec = FP4_MOE[fmt]
    print(f'=== 1-rank {fmt.upper()} x {fmt.upper()} mega-MoE ({len(SHAPES_1RANK)} shapes) ===')
    try:
        for shape in SHAPES_1RANK:
            y, ref, diff, s = _run_one(spec, shape, rank, world, group)
            try:
                _check_and_print(_shape_tag(shape), fmt, diff, y, ref)
            finally:
                s.buf.destroy()
        print(f'All 1-rank {fmt.upper()} cases passed.\n')
    finally:
        dist.destroy_process_group()


def _multirank_worker(local_rank, num_local_ranks, cases):
    """Spawned per-rank worker: init dist once, loop (fmt, shape) cases, assert per rank."""
    rank, world, group = init_dist(local_rank, num_local_ranks)
    try:
        for fmt, shape in cases:
            _w, exp, topk, tok, h, i, m = shape
            inner = (tok, exp, topk, h, i, m)  # strip the leading world field
            spec = FP4_MOE[fmt]
            y, ref, diff, s = _run_one(spec, inner, rank, world, group)
            try:
                # All ranks compute the same replicated reference; rank 0 reports.
                _check_and_print(_shape_tag(shape, world), fmt, diff, y, ref)
            finally:
                s.buf.destroy()
        dist_print(f'All EP{world} cases passed.', once_in_node=True)
    finally:
        dist.destroy_process_group()


def test_fp4_mega_moe_multirank(world: int = 2) -> None:
    """Run the multi-rank EP shape matrix for a given world size (both formats)."""
    cases = [(fmt, shape) for shape in SHAPES_MULTIRANK if shape[0] == world for fmt in FP4_MOE]
    if not cases:
        print(f'No multi-rank shapes for world={world}; skipping.')
        return
    _ensure_master(13800 + world)
    print(f'=== multi-rank EP (world={world}, {len(cases)} cases) ===', flush=True)
    torch.multiprocessing.spawn(_multirank_worker, args=(world, cases), nprocs=world)
    print()


def _run_one_captured(spec, shape, rank, world, group, chunk_ratio):
    """Like _run_one but also captures the kernel's num_cycles printf and the ring size."""
    s = _build_s(spec, shape, rank, world, group, chunk_ratio=chunk_ratio)
    ring = s.buf.num_ring_tokens
    try:
        num_cycles, y, _ = capture_num_cycles(lambda: spec.run(s))
        ref = spec.reference(s)
        return y, ref, calc_diff(y, ref), s, num_cycles, ring
    except Exception:
        s.buf.destroy()
        raise


def _check_num_cycles(num_cycles, world, num_max, ring, ratio, label):
    """Assert num_cycles behavior:
    - ring >= num_min (worst-case bound) -> the ring holds the whole pool -> num_cycles == 1
      ("big enough ratio = 1 cycle").
    - ring < num_min AND ratio <= 1 -> chunking definitely happens -> num_cycles >= 2.
    - ring < num_min AND ratio > 1 -> the ring may or may not hold the actual (smaller) pool
      -> num_cycles >= 1 (no over-constraint; correctness is already guarded by diff<DIFF_TOL).
    - 'auto' -> ring = num_max*num_topk (no-wrap minimum) -> num_cycles == 1.
    """
    align_m = deep_gemm._C.get_token_alignment_for_mega_moe()
    num_min = world * align(num_max, align_m)
    assert num_cycles is not None and num_cycles >= 1, f'{label}: num_cycles missing/got {num_cycles}'
    if ratio == 'auto':
        # 'auto' sizes the ring to num_max*num_topk (no-wrap minimum) -> num_cycles == 1.
        assert num_cycles == 1, (
            f'{label}: auto should give no-wrap num_cycles==1 (ring={ring}), got {num_cycles}')
    elif ring >= num_min:
        assert num_cycles == 1, (
            f'{label}: expected num_cycles==1 (ring={ring} >= num_min={num_min}), got {num_cycles}')
    elif ratio <= 1.0:
        assert num_cycles >= 2, (
            f'{label}: expected num_cycles>=2 (ring={ring} < num_min={num_min}, ratio={ratio}), got {num_cycles}')


def test_fp4_mega_moe_prefill_1rank(fmt: str = 'mxfp4') -> None:
    """1-rank prefill: sweep chunk_ratio, assert diff<DIFF_TOL and num_cycles behavior."""
    assert fmt in FP4_MOE, f'unknown FP4 format {fmt!r}; expected one of {list(FP4_MOE)}'
    _ensure_master(13801)
    rank, world, group = init_dist(0, 1)
    spec = FP4_MOE[fmt]
    n_cases = len(PREFILL_1RANK_SHAPES) * len(PREFILL_RATIOS)
    print(f'=== prefill 1-rank {fmt.upper()} ({n_cases} cases: {len(PREFILL_1RANK_SHAPES)} shapes x {PREFILL_RATIOS}) ===')
    try:
        for shape in PREFILL_1RANK_SHAPES:
            num_max = max(shape[0], 1)
            for ratio in PREFILL_RATIOS:
                y, ref, diff, s, num_cycles, ring = _run_one_captured(spec, shape, rank, world, group, ratio)
                try:
                    tag = f'{_shape_tag(shape)} ratio={ratio} ring={ring} nc={num_cycles}'
                    _check_and_print(tag, fmt, diff, y, ref)
                    _check_num_cycles(num_cycles, world, num_max, ring, ratio, tag)
                finally:
                    s.buf.destroy()
        print(f'All prefill 1-rank {fmt.upper()} cases passed.\n')
    finally:
        dist.destroy_process_group()


def _prefill_multirank_worker(local_rank, num_local_ranks, cases):
    """Spawned per-rank worker: sweep (fmt, shape, ratio), assert diff + num_cycles per rank."""
    rank, world, group = init_dist(local_rank, num_local_ranks)
    try:
        for fmt, shape, ratio in cases:
            _w, exp, topk, tok, h, i, m = shape
            inner = (tok, exp, topk, h, i, m)
            num_max = max(tok, 1)
            spec = FP4_MOE[fmt]
            y, ref, diff, s, num_cycles, ring = _run_one_captured(spec, inner, rank, world, group, ratio)
            try:
                tag = f'{_shape_tag(shape, world)} ratio={ratio} ring={ring} nc={num_cycles}'
                _check_and_print(tag, fmt, diff, y, ref)
                _check_num_cycles(num_cycles, world, num_max, ring, ratio, tag)
            finally:
                s.buf.destroy()
        dist_print(f'All prefill EP{world} cases passed.', once_in_node=True)
    finally:
        dist.destroy_process_group()


def test_fp4_mega_moe_prefill_multirank(world: int = 2) -> None:
    """Multi-rank prefill: sweep chunk_ratio across shapes for a given world size."""
    cases = [(fmt, shape, ratio) for shape in PREFILL_MULTIRANK_SHAPES if shape[0] == world
             for ratio in PREFILL_RATIOS for fmt in FP4_MOE]
    if not cases:
        print(f'No prefill multi-rank shapes for world={world}; skipping.')
        return
    _ensure_master(13900 + world)
    n_shapes = sum(1 for s in PREFILL_MULTIRANK_SHAPES if s[0] == world)
    print(f'=== prefill multi-rank EP (world={world}, {len(cases)} cases: {n_shapes} shapes x '
          f'{PREFILL_RATIOS} x {len(FP4_MOE)} fmts) ===', flush=True)
    torch.multiprocessing.spawn(_prefill_multirank_worker, args=(world, cases), nprocs=world)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Packed FP4 mega-MoE correctness (1-rank + multi-rank EP)')
    parser.add_argument('--scope', choices=['all', '1rank', 'multirank', 'baseline', 'prefill'],
                        default='all', help='Test scope (default: all)')
    parser.add_argument('--worlds', type=str, default='2,4,8',
                        help='Comma-separated world sizes for the multi-rank scope')
    args = parser.parse_args()

    if args.scope in ('all', 'baseline'):
        for fmt in FP4_MOE:
            test_fp4_mega_moe(fmt)
    if args.scope in ('all', '1rank'):
        for fmt in FP4_MOE:
            test_fp4_mega_moe_1rank(fmt)
    if args.scope in ('all', 'multirank'):
        for w in [int(x) for x in args.worlds.split(',') if x.strip()]:
            test_fp4_mega_moe_multirank(w)
    if args.scope in ('all', 'prefill'):
        for fmt in FP4_MOE:
            test_fp4_mega_moe_prefill_1rank(fmt)
        for w in [int(x) for x in args.worlds.split(',') if x.strip()]:
            test_fp4_mega_moe_prefill_multirank(w)
