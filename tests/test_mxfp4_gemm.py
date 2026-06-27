import random
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor
from deep_gemm.utils.math import per_token_cast_to_fp4, cast_back_from_fp4


def _prepare(x: torch.Tensor, gran_k: int = 32):
    # Packed E2M1 (int8, K/2) + float UE8M0 SF (gran-K 32)
    packed, sf_f = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=gran_k)
    # Dequantized reference using the *same* packed values + SF
    deq = cast_back_from_fp4(packed, sf_f, gran_k=gran_k)
    # SF in the MN-major TMA-aligned packed UE8M0 layout the kernel expects
    sf_packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_f)
    return packed, sf_packed, deq


def test_mxfp4_gemm() -> None:
    print('Testing packed MXFP4 x MXFP4 GEMM:')
    # NOTES: this de-risk kernel is hardcoded to a 2-CTA (cluster_n = 2) config, so it
    #        requires N divisible by 256, M divisible by 128, and K divisible by 128.
    for m, n, k in ((256, 256, 256), (256, 256, 512), (512, 256, 1024), (128, 512, 256), (1024, 768, 512)):
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

        a_packed, sfa, a_deq = _prepare(a)
        b_packed, sfb, b_deq = _prepare(b)

        ref_d = (a_deq.float() @ b_deq.float().T).to(torch.bfloat16)

        d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.mxfp4_gemm_nt((a_packed, sfa), (b_packed, sfb), d)

        diff = calc_diff(d, ref_d)
        status = 'OK' if diff < 0.05 else 'FAIL'
        print(f' > m={m:5}, n={n:5}, k={k:5}: diff={diff:.5f}  [{status}]')
        assert diff < 0.05, f'{m=}, {n=}, {k=}, diff={diff:.5f}'
    print('All MXFP4 GEMM cases passed.\n')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    test_mxfp4_gemm()
