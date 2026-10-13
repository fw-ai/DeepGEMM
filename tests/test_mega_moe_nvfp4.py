# Tests for `deep_gemm.mega_nvfp4` (sgl-project/DeepGEMM's NVFP4 MegaMoE, ported additively).
#
# Ported from sgl-project/DeepGEMM@ec226bcf400c11d62755137b916fae9ee359c684:
#   sgl_deep_gemm/tests/test_mega_moe_pre_dispatch.py  -> `test_pre_dispatch`
#   sgl_deep_gemm/tests/test_mega_moe_nvfp4_alphas.py  -> `test_alphas`
# Those files are licensed under the Apache License 2.0 (sgl-project/DeepGEMM `sgl_deep_gemm/LICENSE`);
# changes: target `deep_gemm.mega_nvfp4` instead of `deep_gemm`, `nvfp4_mega_moe` instead of
# `fp8_fp4_mega_moe(recipe=(1, 1, 16))`, both tests in one file, and a new bytewise check of the NVFP4
# pre-dispatch quantization (`--dtype nvfp4`), which is the format Fireworks feeds the kernel.
#
# Usage (B200/B300):
#   python tests/test_mega_moe_nvfp4.py                     # pre-dispatch (1 GPU) + alphas (2 ranks)
#   python tests/test_mega_moe_nvfp4.py --test alphas --num-processes 8

import argparse
import sys
import torch
import torch.distributed as dist

import deep_gemm.mega_nvfp4 as mega_nvfp4
from deep_gemm.mega_nvfp4 import transform_ue4m3_sf_into_required_layout
from deep_gemm.mega_nvfp4.utils import cast_to_ue4m3
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import _quantize_to_fp4_e2m1
from deep_gemm.utils.dist import dist_print, init_dist

GRAN_K = 16
E4M3_MAX = 448.0
FP4_MAX = 6.0


# ---------------------------------------------------------------------------------------------------------------
# Pre-dispatch: bytewise + correctness probe for `mega_moe_pre_dispatch`.
#
# The fused pre-dispatch kernel produces the exact byte layout the mega-MoE symmetric `x`, `x_sf`,
# `topk_idx`, and `topk_weights` slots expect. This verifies bit-for-bit equivalence against the in-tree
# host helpers (`per_token_cast_to_fp8`, `per_token_cast_to_fp4`) for the FP8 and the packed FP4 dtype
# branches, an NVFP4 host reference, plus the pad-fill correctness contract.
# ---------------------------------------------------------------------------------------------------------------

def _alloc_outputs(padded_max: int, hidden: int, top_k: int, group_size: int, use_fp4_acts: bool):
    num_groups = hidden // group_size
    assert num_groups % 4 == 0
    if use_fp4_acts:
        buf_x = torch.empty((padded_max, hidden // 2), dtype=torch.int8, device='cuda')
    else:
        buf_x = torch.empty((padded_max, hidden), dtype=torch.float8_e4m3fn, device='cuda')
    buf_x_sf = torch.empty((padded_max, num_groups // 4), dtype=torch.int32, device='cuda')
    buf_topk_idx = torch.empty((padded_max, top_k), dtype=torch.int64, device='cuda')
    buf_topk_weights = torch.empty((padded_max, top_k), dtype=torch.float32, device='cuda')
    # Sentinel-fill so any write-correctness bug shows up as a non-zero diff.
    buf_x.fill_(0)
    buf_x_sf.fill_(0)
    buf_topk_idx.fill_(0)
    buf_topk_weights.fill_(0)
    return buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights


def _nvfp4_per_token_reference(x: torch.Tensor):
    """Host model of the NVFP4 per-token recipe in `sm100_mega_moe_pre_dispatch.cuh`:
    pts = max(row_absmax, 1e-4) / (448 * 6); sf = e4m3_rn(group_absmax / (6 * pts)); codes = e2m1(x / (sf * pts))."""
    # NOTES: each FP32 op is evaluated in FP64 and rounded once to FP32, which reproduces IEEE FP32 `*` and `/`
    # exactly (PyTorch itself lowers division by a scalar to a reciprocal multiply, which is not bit-exact)
    def f32(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.float32)

    m, h = x.shape
    v = x.float().view(m, -1, GRAN_K)
    group_max = v.abs().amax(dim=2)
    row_max = group_max.amax(dim=1, keepdim=True).clamp_min(1e-4)
    pts = f32(row_max.double() / (448.0 * 6.0))
    sf = f32(group_max.double() / f32(6.0 * pts.double()).double()).to(torch.float8_e4m3fn)
    sf_dec = sf.float()
    inv = torch.where(sf_dec > 0, f32(1.0 / f32(sf_dec.double() * pts.double()).double()), torch.zeros_like(sf_dec))
    codes = _quantize_to_fp4_e2m1(f32(v.double() * inv.unsqueeze(2).double())).view(m, h // 2, 2)
    packed = (codes[..., 0] & 0x0F) | ((codes[..., 1] & 0x0F) << 4)
    return packed.contiguous(), sf.view(torch.uint8), pts.squeeze(1)


def _run_pre_dispatch(dtype: str, args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    M = args.num_tokens
    P = args.padded_max
    H = args.hidden
    K = args.top_k
    use_fp4_acts = dtype in ('fp4', 'nvfp4')
    G = GRAN_K if dtype == 'nvfp4' else args.group_size
    assert P >= M, 'padded_max must be >= num_tokens'

    # --- Inputs (BF16 acts, int32 topk_idx, float topk_weights) ---
    x = torch.randn((M, H), dtype=torch.bfloat16, device='cuda')
    if dtype == 'fp4' and M > 0 and G >= 32:
        # Force scale=1 in the first group and exercise every positive/negative
        # E2M1 midpoint; random BF16 inputs alone need not hit each tie.
        x[0, :G] = 0
        x[0, :16] = torch.tensor(
            [0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6,
             -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5],
            dtype=x.dtype, device=x.device)
    # Use plausible expert ids in [0, num_experts) and float weights.
    num_experts = args.num_experts
    topk_idx = torch.randint(0, num_experts, (M, K), dtype=torch.int32, device='cuda')
    topk_weights = torch.randn((M, K), dtype=torch.float32, device='cuda')

    buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights = _alloc_outputs(P, H, K, G, use_fp4_acts)
    buf_x_scales = torch.zeros((P, ), dtype=torch.float32, device='cuda') if dtype == 'nvfp4' else None

    # --- Kernel under test ---
    mega_nvfp4.mega_moe_pre_dispatch(
        x, topk_idx, topk_weights,
        buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights,
        num_tokens=M, group_size=G,
        mma_type={'fp8': 'fp8xfp4', 'fp4': 'mxf4xmxf4', 'nvfp4': 'nvfp4xnvfp4'}[dtype],
        buf_x_scales=buf_x_scales,
    )
    torch.cuda.synchronize()

    # --- Reference (host helper) ---
    if dtype == 'fp4':
        ref_x, ref_sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)
    elif dtype == 'fp8':
        ref_x, ref_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=G, use_packed_ue8m0=True)
    else:
        ref_x, ref_sf, ref_pts = _nvfp4_per_token_reference(x)
        assert torch.equal(buf_x_scales[:M], ref_pts), 'NVFP4 per-token outer scale mismatch'

    # --- Bytewise compare on valid-token rows ---
    kernel_bytes = buf_x[:M].view(torch.uint8)
    ref_bytes = ref_x.view(torch.uint8)
    diff_x = (kernel_bytes != ref_bytes)
    if diff_x.any().item():
        i, j = diff_x.nonzero()[0].tolist()
        raise AssertionError(
            f'[{dtype}] buf_x mismatch at row {i}, col {j}: kernel={int(kernel_bytes[i, j])} '
            f'ref={int(ref_bytes[i, j])} (total mismatches={int(diff_x.sum())})')
    if dtype == 'fp4' and M > 0 and G >= 32:
        expected_midpoints = torch.tensor(
            [0x00, 0x22, 0x44, 0x66, 0x07, 0xaa, 0xcc, 0xee],
            dtype=torch.uint8, device=x.device)
        assert torch.equal(kernel_bytes[0, :8], expected_midpoints), 'FP4 ties must round to even'

    # SF byte layout: (M, num_groups/4) int32 → (M, num_groups) bytes.
    kernel_sf_bytes = buf_x_sf[:M].view(torch.uint8)
    ref_sf_bytes = ref_sf.view(torch.uint8)
    diff_sf = (kernel_sf_bytes != ref_sf_bytes)
    if diff_sf.any().item():
        i, j = diff_sf.nonzero()[0].tolist()
        raise AssertionError(
            f'[{dtype}] buf_x_sf mismatch at row {i}, byte {j}: kernel={int(kernel_sf_bytes[i, j])} '
            f'ref={int(ref_sf_bytes[i, j])} (total mismatches={int(diff_sf.sum())})')

    # --- topk pass-through and pad-fill ---
    if not torch.equal(buf_topk_idx[:M], topk_idx.to(torch.int64)):
        raise AssertionError('topk_idx pass-through mismatch on valid rows')
    if not torch.equal(buf_topk_weights[:M], topk_weights):
        raise AssertionError('topk_weights pass-through mismatch on valid rows')
    if P > M:
        if not torch.all(buf_topk_idx[M:] == -1).item():
            raise AssertionError('pad rows of buf_topk_idx must equal -1')
        if not torch.all(buf_topk_weights[M:] == 0.0).item():
            raise AssertionError('pad rows of buf_topk_weights must equal 0.0')

    print(f'  PASS  [{dtype}] M={M} P={P} H={H} K={K} G={G} — bytewise equal vs host reference + pad-fill correct')


def test_pre_dispatch(args: argparse.Namespace) -> None:
    print('Testing pre-dispatch:')
    for dtype in (('fp8', 'fp4', 'nvfp4') if args.dtype == 'all' else (args.dtype, )):
        _run_pre_dispatch(dtype, args)


# ---------------------------------------------------------------------------------------------------------------
# NVFP4 outer-scale (global scale) plumbing for the mega-MoE kernel.
#
# Four tables carry the NVFP4 dequant alphas:
#
#   x_scales      per token, written by `mega_moe_pre_dispatch` (1 / gs_x1)
#   l1_alphas     [num_LOCAL_experts, 2] = 1 / (gs_w1, gs_w3), on the L1 accum
#   l2_alphas     [num_LOCAL_experts]    = 1 / gs_w2,          on the L2 accum
#   expert_scales [num_GLOBAL_experts]   folded into the topk weight, O(1) only
#
# The fc2 alpha is deliberately NOT routed through `expert_scales`: that lands on the L1 output, which is
# then re-quantized to NVFP4 with a bare per-block E4M3 SF, and a realistic `1 / gs_w2` (~1e-4) sinks that SF
# under E4M3's subnormal floor — `y` comes back all zeros (measured). `expert_scales` is exercised
# separately below with an O(1) ratio, which is all it can carry.
#
# Method: quantize the same BF16 weights twice — once with gs = 1 and no alphas, once with per-expert gs
# and the alphas that undo it — and compare `y`.
#   - Power-of-two gs: every scaling is exact in FP32/E4M3, so the two runs must be BITWISE equal.
#   - Realistic gs (= 448*6/amax, not powers of two): the two encodings round differently, so this arm is
#     only a sanity band — measured 0.18 rel-L2 on `y`, which is what two independent NVFP4 encodings cost.
#   - Control: rolling either alpha table by one expert must change `y`, otherwise the test would pass on a
#     kernel that ignores the table.
#
# Run with >= 2 ranks: local- vs global-expert index confusion only shows up on a rank whose `expert_lo`
# is nonzero.
# ---------------------------------------------------------------------------------------------------------------

def cast_to_nvfp4_with_gs(w: torch.Tensor, gs) -> tuple:
    """NVFP4 with an outer scale. Dequant contract: `w ~= fp4 * sf / gs`.

    `gs` is a scalar or a per-row (m,) tensor — the L1 weight carries gs_w1 on its gate rows and gs_w3 on its
    up rows. `gs = 1` reproduces `per_token_cast_to_nvfp4` exactly.
    """
    m, n = w.shape
    v = w.view(m, -1, GRAN_K).float()
    gs = torch.as_tensor(gs, dtype=torch.float, device=w.device).reshape(-1, 1).expand(m, 1)
    sf = cast_to_ue4m3(v.abs().amax(dim=2) * (gs / FP4_MAX))
    scale = torch.where(sf > 0, gs / sf, torch.zeros_like(sf))
    codes = _quantize_to_fp4_e2m1(v * scale.unsqueeze(2)).view(m, n)
    codes2 = codes.view(m, n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed.contiguous(), sf


def cast_grouped(w: torch.Tensor, gs_rows) -> tuple:
    """`w` is (E, N, K) BF16; `gs_rows` is one per-row gs tensor per expert."""
    num_experts, n, k = w.shape
    packed = torch.empty((num_experts, n, k // 2), dtype=torch.int8, device=w.device)
    sf = torch.empty((num_experts, n, k // GRAN_K), dtype=torch.float, device=w.device)
    for e in range(num_experts):
        packed[e], sf[e] = cast_to_nvfp4_with_gs(w[e], gs_rows[e])
    return packed, transform_ue4m3_sf_into_required_layout(sf, n)


def make_gs(amax_per_expert: torch.Tensor, pow2: bool) -> torch.Tensor:
    """Realistic NVFP4 global scale, `448 * 6 / amax`, optionally snapped to a power of two so the two runs
    stay bitwise comparable, and spread downwards per expert so the roll-by-one control is not vacuous."""
    gs = (E4M3_MAX * FP4_MAX) / amax_per_expert.clamp_min(1e-4)
    if not pow2:
        return gs
    spread = torch.pow(2.0, -(torch.arange(gs.numel(), device=gs.device) % 4).float())
    return torch.pow(2.0, torch.floor(torch.log2(gs))) * spread


# noinspection PyUnboundLocalVariable
def _alphas_worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(1000 + rank_idx)

    num_tokens = args.num_tokens_alphas
    hidden, inter = args.hidden_alphas, args.intermediate_hidden
    num_experts, num_topk = args.num_experts_alphas, args.num_topk
    num_local_experts = num_experts // num_ranks
    expert_lo = rank_idx * num_local_experts

    buffer = mega_nvfp4.get_symm_buffer_for_mega_moe(
        group, num_experts, args.num_max_tokens_per_rank, num_topk,
        hidden, inter, mma_type='nvfp4xnvfp4')

    # `x` is damped so the L1 output amax stays under the L2 activation SF's 448*6 ceiling; weights stay
    # unit-variance so their own block SFs stay in E4M3's normal range in BOTH runs.
    x = (torch.randn((num_tokens, hidden), device='cuda') * args.x_scale).bfloat16()
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.int()

    # Weights are local, but a GLOBAL table (`expert_scales`) has to know every rank's values. Draw all
    # experts from a shared seed and keep the local slice, so the global tables agree across ranks.
    torch.manual_seed(7)
    l1_all = torch.randn((num_experts, inter * 2, hidden), dtype=torch.bfloat16, device='cuda')
    l2_all = torch.randn((num_experts, hidden, inter), dtype=torch.bfloat16, device='cuda')
    gate_amax = l1_all[:, :inter].abs().float().amax(dim=(1, 2))
    up_amax = l1_all[:, inter:].abs().float().amax(dim=(1, 2))
    l2_amax = l2_all.abs().float().amax(dim=(1, 2))
    l1_local = l1_all[expert_lo:expert_lo + num_local_experts].contiguous()
    l2_local = l2_all[expert_lo:expert_lo + num_local_experts].contiguous()
    del l1_all, l2_all

    base_w = mega_nvfp4.transform_weights_for_mega_moe(
        cast_grouped(l1_local, [torch.ones(inter * 2, device='cuda')] * num_local_experts),
        cast_grouped(l2_local, [torch.ones(hidden, device='cuda')] * num_local_experts),
        'swiglu', 'nvfp4xnvfp4')

    def run(weights, l1_alphas=None, l2_alphas=None, expert_scales=None, tw=None, l2_act_scales=None):
        mega_nvfp4.mega_moe_pre_dispatch(
            x, topk_idx, topk_weights if tw is None else tw,
            buffer.x, buffer.x_sf, buffer.topk_idx, buffer.topk_weights,
            num_tokens=num_tokens, group_size=GRAN_K, mma_type='nvfp4xnvfp4',
            buf_x_scales=buffer.x_scales, expert_scales=expert_scales)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        mega_nvfp4.nvfp4_mega_moe(
            y=y, l1_weights=weights[0], l2_weights=weights[1], sym_buffer=buffer,
            activation='swiglu', fast_math=bool(args.fast_math),
            use_x_scales=True, l1_alphas=l1_alphas, l2_alphas=l2_alphas,
            l2_act_scales=l2_act_scales)
        dist.barrier()
        torch.cuda.synchronize()
        return y

    y_base = run(base_w)
    dist_print(f'Config: {num_ranks} ranks x {num_local_experts} local experts, '
               f'{num_tokens} tokens, hidden={hidden}, inter={inter}, '
               f'|y_base| mean {y_base.float().abs().mean().item():.4g}', once_in_node=True)

    for pow2 in (True, False):
        gs_w1, gs_w3, gs_w2 = (make_gs(a, pow2) for a in (gate_amax, up_amax, l2_amax))
        scaled = mega_nvfp4.transform_weights_for_mega_moe(
            cast_grouped(l1_local, [torch.cat([gs_w1[expert_lo + e].expand(inter),
                                               gs_w3[expert_lo + e].expand(inter)])
                                    for e in range(num_local_experts)]),
            cast_grouped(l2_local, [gs_w2[expert_lo + e].expand(hidden)
                                    for e in range(num_local_experts)]),
            'swiglu', 'nvfp4xnvfp4')

        # Both LOCAL index space; `l1_alphas` columns are (gate, up).
        local = slice(expert_lo, expert_lo + num_local_experts)
        l1_alphas = torch.stack([1.0 / gs_w1[local], 1.0 / gs_w3[local]], dim=1).contiguous()
        l2_alphas = (1.0 / gs_w2[local]).contiguous()

        y_scaled = run(scaled, l1_alphas, l2_alphas)
        tag = 'pow2' if pow2 else 'realistic'
        if pow2:
            assert torch.equal(y_base, y_scaled), \
                f'[{tag}] power-of-two global scales must cancel bitwise, max |delta| = ' \
                f'{(y_base.float() - y_scaled.float()).abs().max().item():.3e}'
            dist_print(f' > [{tag}] gs_w2 = {gs_w2.tolist()} — bitwise equal', once_in_node=True)
        else:
            rel = ((y_scaled.float() - y_base.float()).norm() / y_base.float().norm()).item()
            assert rel < args.rel_tol, f'[{tag}] rel-L2 {rel:.4f} >= {args.rel_tol}'
            dist_print(f' > [{tag}] rel-L2 = {rel:.5f} (< {args.rel_tol})', once_in_node=True)

        for name, wrong in (('l1_alphas', (l1_alphas.roll(1, dims=0).contiguous(), l2_alphas)),
                            ('l2_alphas', (l1_alphas, l2_alphas.roll(1).contiguous()))):
            assert not torch.equal(y_scaled, run(scaled, *wrong)), \
                f'[{tag}] rolling `{name}` changed nothing — the kernel ignores it'
        dist_print(f' > [{tag}] roll-by-one control: both tables have teeth', once_in_node=True)

    # `l2_act_scales` (LOCAL index): pow2 fc2 input gs is cancelled by `1/gs` in `l2_alphas`. Damp x so the
    # scaled block SFs stay under E4M3's 448 satfinite ceiling. Exact pow2 invariance still breaks for blocks
    # whose SF lands in E4M3's SUBNORMAL range; bound by ~1 BF16 ulp of the tensor scale, not bitwise.
    x = (x.float() * 0.25).bfloat16()
    y_damped = run(base_w)
    gs_in = torch.pow(2.0, (torch.arange(num_local_experts, device='cuda', dtype=torch.float) % 3))
    y_gs = run(base_w, l2_alphas=(1.0 / gs_in).contiguous(), l2_act_scales=gs_in.contiguous())
    d = (y_gs.float() - y_damped.float()).abs()
    tol = torch.finfo(torch.bfloat16).eps * y_damped.float().abs().max()
    rel = (d.norm() / y_damped.float().norm()).item()
    assert d.max() <= tol and rel < 1e-3, \
        f'l2_act_scales cancellation off: max |d| {d.max().item():.3e} (tol {tol.item():.3e}), rel-L2 {rel:.3e}'
    assert not torch.equal(y_damped, run(base_w, l2_act_scales=gs_in.contiguous())), \
        'l2_act_scales alone changed nothing -- the kernel ignores it'
    dist_print(f' > l2_act_scales: pow2 gs cancels to 1 tensor-scale ulp '
               f'(max |d| {d.max().item():.2e}, rel {rel:.1e}), teeth ok', once_in_node=True)

    # `expert_scales`, GLOBAL index space: folding an O(1) per-expert ratio in the kernel must equal folding
    # it into `topk_weights` on the host. Bitwise — both are the same FP32 multiply.
    ratios = (1.0 + 0.25 * torch.arange(num_experts, device='cuda', dtype=torch.float) / num_experts).contiguous()
    host_tw = topk_weights * torch.where(topk_idx >= 0, ratios[topk_idx.long().clamp_min(0)], 1.0)
    assert torch.equal(run(base_w, expert_scales=ratios), run(base_w, tw=host_tw)), \
        '`expert_scales` does not match a host-side fold — check the global expert index'
    dist_print(' > expert_scales: matches a host-side topk_weights fold', once_in_node=True)

    dist_print('OK', once_in_node=True)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


def test_alphas(args: argparse.Namespace) -> None:
    print(f'Testing NVFP4 alphas ({args.num_processes} ranks):')
    assert args.num_experts_alphas % args.num_processes == 0
    torch.multiprocessing.spawn(_alphas_worker, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=('all', 'pre_dispatch', 'alphas'), default='all')
    # Pre-dispatch
    parser.add_argument('--num-tokens', type=int, default=512)
    parser.add_argument('--padded-max', type=int, default=576)  # > num_tokens to exercise pad
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--group-size', type=int, default=32)
    parser.add_argument('--num-experts', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dtype', choices=['fp8', 'fp4', 'nvfp4', 'all'], default='all')
    # Alphas
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=1024)
    parser.add_argument('--num-tokens-alphas', type=int, default=1024)
    parser.add_argument('--hidden-alphas', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts-alphas', type=int, default=8)
    parser.add_argument('--num-topk', type=int, default=2)
    parser.add_argument('--x-scale', type=float, default=0.1)
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--rel-tol', type=float, default=0.30)
    args = parser.parse_args()

    if args.test in ('all', 'pre_dispatch'):
        test_pre_dispatch(args)
    if args.test in ('all', 'alphas'):
        test_alphas(args)
    print('OK')
    sys.exit(0)
