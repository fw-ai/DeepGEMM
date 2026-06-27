import random
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import (
    align, per_token_cast_to_nvfp4, cast_back_from_nvfp4, nvfp4_global_scale,
)
from deep_gemm.utils.layout import get_tma_aligned_size

GRAN_K = 16


def _mn_major_packed_e4m3(sf_bytes: torch.Tensor) -> torch.Tensor:
    """[mn, k_sf] uint8 (E4M3 bytes) -> MN-major TMA-aligned int32 (4 SF bytes/int32),
    matching `get_mn_major_tma_aligned_packed_ue8m0_tensor` but for raw E4M3 bytes."""
    mn, k = sf_bytes.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((aligned_mn, aligned_k), dtype=torch.uint8, device=sf_bytes.device)
    padded[:mn, :k] = sf_bytes
    padded = padded.reshape(-1).view(torch.int32).view(aligned_mn, aligned_k // 4)
    out = torch.empty_strided((aligned_mn, aligned_k // 4), (1, aligned_mn),
                              dtype=torch.int32, device=sf_bytes.device)
    return out.copy_(padded)[:mn]


def _prepare(x: torch.Tensor, gs: float):
    packed, sf_bytes = per_token_cast_to_nvfp4(x, gs, gran_k=GRAN_K)
    deq = cast_back_from_nvfp4(packed, sf_bytes, gs, gran_k=GRAN_K)
    sf_packed = _mn_major_packed_e4m3(sf_bytes)
    return packed, sf_packed, deq


def test_nvfp4_gemm() -> None:
    print('Testing packed NVFP4 x NVFP4 GEMM:')
    for m, n, k in ((256, 256, 256), (256, 256, 512), (512, 256, 1024), (128, 512, 256), (1024, 768, 512)):
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        gs_a = nvfp4_global_scale(a)
        gs_b = nvfp4_global_scale(b)

        a_packed, sfa, a_deq = _prepare(a, gs_a)
        b_packed, sfb, b_deq = _prepare(b, gs_b)

        ref_d = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)

        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.nvfp4_gemm_nt((a_packed, sfa), (b_packed, sfb), d,
                                a_global_scale=gs_a, b_global_scale=gs_b)

        diff = calc_diff(d, ref_d)
        status = 'OK' if diff < 0.05 else 'FAIL'
        print(f' > m={m:5}, n={n:5}, k={k:5}: diff={diff:.5f}  [{status}]')
        assert diff < 0.05, f'{m=}, {n=}, {k=}, diff={diff:.5f}'
    print('All NVFP4 GEMM cases passed.\n')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    test_nvfp4_gemm()
