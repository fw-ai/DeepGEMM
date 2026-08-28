"""Static contracts for the GPU-verified K3 three-range high-water."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
PARENT = (
    ROOT / "deep_gemm/include/deep_gemm/impls/" "sm100_fp8_fp4_mega_moe_backward.cuh"
)
RANGES = ROOT / "deep_gemm/include/deep_gemm/impls/k3_multirange_backward.hpp"


def test_three_range_exact_ring_stays_compile_time_disabled() -> None:
    """Three-range training must retain the verified ready-BF16 suffix."""
    jit = JIT.read_text()
    assert jit.count("const bool enable_k3_mxfp8_exact_epilogue_ring = false;") == 2
    assert (
        "args.backward_ranges.num_ranges >= 2u &&\n" "            args.num_topk == 16u"
    ) in jit
    assert "DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS 32" in jit
    assert "DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING" in jit


def test_terminal_grad_x_reduction_uses_verified_two_warps() -> None:
    """Keep the 64K reduction cohort from contending with dW13 UMMA/TMA."""
    parent = PARENT.read_text()
    begin = parent.index("constexpr uint32_t kThreeRangeReduceFirstWarp")
    end = parent.index("const auto* const three_range_backward_ranges", begin)
    reduction = parent[begin:end]
    assert "kThreeRangeReduceFirstWarp = 8u" in reduction
    assert "kThreeRangeReduceWarps = 2u" in reduction
    assert "kThreeRangeReduceWarps = 4u" not in reduction
    assert "concurrent dW13 UMMA/TMA stream" in reduction


def test_highwater_keeps_the_28_map_descriptor_abi() -> None:
    """Reject the unverified K128 ring maps that expanded the host ABI."""
    ranges = RANGES.read_text()
    assert "kK3MxFp8WgradNumProducerTensorMaps = 12u" in ranges
    assert "kK3MxFp8DW13RingK128ValueAPrimaryMap" not in ranges
    assert "kK3MxFp8DW13RingK128ValueBResidualMap" not in ranges
