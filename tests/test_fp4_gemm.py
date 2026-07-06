import random
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.layout import (
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_tma_aligned_size,
)
from deep_gemm.utils.math import (
    align,
    per_token_cast_to_fp4,
    cast_back_from_fp4,
    per_token_cast_to_nvfp4,
    cast_back_from_nvfp4,
    nvfp4_global_scale,
)

# Packed FP4 GEMM is a de-risk kernel hardcoded to a 2-CTA (cluster_n = 2) config,
# so it requires N divisible by 256, M divisible by 128, and K divisible by 128.
GEMM_SHAPES = (
    (256, 256, 256),
    (256, 256, 512),
    (512, 256, 1024),
    (128, 512, 256),
    (1024, 768, 512),
)

DIFF_TOL = 0.05

MXFP4_GRAN_K = 32
NVFP4_GRAN_K = 16


def _mn_major_packed_e4m3(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[mn, k_sf] uint8 (E4M3 bytes) -> MN-major TMA-aligned int32 (4 SF bytes/int32),
    matching ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` but for raw E4M3 bytes."""
    mn, k = sf_bytes.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((aligned_mn, aligned_k), dtype=torch.uint8, device=sf_bytes.device)
    padded[:mn, :k] = sf_bytes
    padded = padded.reshape(-1).view(torch.int32).view(aligned_mn, aligned_k // 4)
    out = torch.empty_strided((aligned_mn, aligned_k // 4), (1, aligned_mn),
                              dtype=torch.int32, device=sf_bytes.device)
    return out.copy_(padded)[:mn]


def _prepare_mxfp4(x: torch.Tensor):
    # Packed E2M1 (int8, K/2) + float UE8M0 SF (gran-K 32); no global scale.
    packed, sf_f = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=MXFP4_GRAN_K)
    deq = cast_back_from_fp4(packed, sf_f, gran_k=MXFP4_GRAN_K)
    sf_packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_f)
    return packed, sf_packed, deq, None


def _prepare_nvfp4(x: torch.Tensor):
    # Packed E2M1 (int8, K/2) + E4M3 SF bytes (gran-K 16) + per-tensor global scale.
    gs = nvfp4_global_scale(x)
    packed, sf_bytes = per_token_cast_to_nvfp4(x, gs, gran_k=NVFP4_GRAN_K)
    deq = cast_back_from_nvfp4(packed, sf_bytes, gs, gran_k=NVFP4_GRAN_K)
    sf_packed = _mn_major_packed_e4m3(sf_bytes)
    return packed, sf_packed, deq, gs


def _run_mxfp4_gemm(a, b, d, gs_a, gs_b) -> None:
    deep_gemm.mxfp4_gemm_nt(a, b, d)


def _run_nvfp4_gemm(a, b, d, gs_a, gs_b) -> None:
    deep_gemm.nvfp4_gemm_nt(a, b, d, a_global_scale=gs_a, b_global_scale=gs_b)


# Per-format strategy: (prepare, kernel runner). ``prepare`` returns
# (packed, sf_packed, deq, global_scale); ``run`` takes the two (packed, sf) pairs,
# the output, and the two global scales (None for MXFP4).
FP4_GEMM = {
    'mxfp4': (_prepare_mxfp4, _run_mxfp4_gemm),
    'nvfp4': (_prepare_nvfp4, _run_nvfp4_gemm),
}


def test_fp4_gemm(fmt: str = 'mxfp4') -> None:
    assert fmt in FP4_GEMM, f'unknown FP4 format {fmt!r}; expected one of {list(FP4_GEMM)}'
    prepare, run_gemm = FP4_GEMM[fmt]
    print(f'Testing packed {fmt.upper()} x {fmt.upper()} GEMM:')
    for m, n, k in GEMM_SHAPES:
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

        a_packed, sfa, a_deq, gs_a = prepare(a)
        b_packed, sfb, b_deq, gs_b = prepare(b)

        ref_d = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)

        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        run_gemm((a_packed, sfa), (b_packed, sfb), d, gs_a, gs_b)

        diff = calc_diff(d, ref_d)
        status = 'OK' if diff < DIFF_TOL else 'FAIL'
        print(f' > [{fmt}] m={m:5}, n={n:5}, k={k:5}: diff={diff:.5f}  [{status}]')
        assert diff < DIFF_TOL, f'{fmt=}, {m=}, {n=}, {k=}, diff={diff:.5f}'
    print(f'All {fmt.upper()} GEMM cases passed.\n')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    for fmt in FP4_GEMM:
        test_fp4_gemm(fmt)
