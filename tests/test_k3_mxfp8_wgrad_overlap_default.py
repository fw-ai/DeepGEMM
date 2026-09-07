"""Source contract for K3's explicitly selected MXFP8 wgrad overlap."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JIT_HEADER = ROOT / (
    "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
)
DEVICE_HEADER = ROOT / (
    "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)


def test_mxfp8_wgrad_overlap_is_off_unless_the_host_selects_it() -> None:
    """Do not compile the fixed-SM148 path for an ineligible launch."""
    device_source = DEVICE_HEADER.read_text()
    assert (
        "#ifndef DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP\n"
        "#define DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP 0\n"
        "#endif"
    ) in device_source

    jit_source = JIT_HEADER.read_text()
    assert (
        '"#define DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP 1\\n"'
        in jit_source
    )


def test_fixed_geometry_helper_is_compile_time_guarded() -> None:
    """Ineligible launch geometries must not instantiate the SM148 helper."""
    device_source = DEVICE_HEADER.read_text()
    call = "detail::k3_mxfp8_stream_dw2_operands_during_w13<"
    call_offset = device_source.index(call)
    guard_offset = device_source.rfind(
        "if constexpr (kK3MxFp8WgradOverlap)", 0, call_offset
    )
    assert guard_offset >= 0
    assert call_offset - guard_offset < 256
