"""MegaMoE wgrad must follow the upstream K-grouped 3D output ABI."""

import pytest
import torch

import deep_gemm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Blackwell GPU")
@pytest.mark.parametrize("pool_block_m", [16, 32, 64, 96, 128, 192, 240])
@pytest.mark.parametrize("output_n", [128, 256])
def test_wgrad_group_dimension(pool_block_m, output_n):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100/SM103")
    torch.manual_seed(123)
    # Include an empty expert and unequal group extents to catch a descriptor
    # that silently flattens (or aliases) the output group coordinate.
    sizes = [pool_block_m, 0, 2 * pool_block_m, 3 * pool_block_m]
    counts = torch.tensor(sizes, dtype=torch.int32, device="cuda")
    a = torch.randn(sum(sizes), 512, dtype=torch.bfloat16, device="cuda") * 0.1
    b = torch.randn(sum(sizes), output_n, dtype=torch.bfloat16, device="cuda") * 0.1
    output = torch.full((4, 512, output_n), float("nan"), dtype=torch.bfloat16, device="cuda")
    deep_gemm.bf16_mega_moe_backward_w13(output, a, b, counts, pool_block_m)
    torch.cuda.synchronize()
    offset = 0
    for expert, rows in enumerate(sizes):
        reference = a[offset:offset + rows].float().T @ b[offset:offset + rows].float()
        assert torch.isfinite(output[expert]).all()
        torch.testing.assert_close(output[expert].float(), reference, rtol=0.004, atol=0.002)
        offset += rows
    assert torch.count_nonzero(output[1]) == 0
