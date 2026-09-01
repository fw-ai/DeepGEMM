"""Static contracts for allocation-free K3 source-side router adjoints."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RANGES = ROOT / "deep_gemm/include/deep_gemm/impls/k3_multirange_backward.hpp"
BACKWARD = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)
WRAPPER = ROOT / "deep_gemm/mega/backward.py"
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"


def test_per_token_ownership_requires_no_spill() -> None:
    """Consume every route slot before raw dY and exact X reclaim two planes."""
    source = RANGES.read_text()
    begin = source.index("struct K3SourceRouteRetentionContract")
    end = source.index("#undef DG_K3_MULTIRANGE_HOST_DEVICE", begin)
    contract = source[begin:end]
    assert "kGradYPlane = 0u" in contract
    assert "kExactSourceXPlane = 1u" in contract
    assert "kNumSpillPlanes = 0u" in contract
    assert "return false" in contract
    assert "can_reclaim_token" in contract
    assert "completed_route_slots == num_topk" in contract


def test_source_rows_use_rank_uniform_physical_token_capacity() -> None:
    """Keep every EP rank's token/top-k source plane address identical."""
    source = RANGES.read_text()
    begin = source.index("struct K3SourceRouteRetentionContract")
    end = source.index("#undef DG_K3_MULTIRANGE_HOST_DEVICE", begin)
    contract = source[begin:end]
    assert "physical_token_idx" in contract
    assert "physical_token_capacity" in contract
    assert (
        "static_cast<uint64_t>(topk_slot) *\n"
        "            physical_token_capacity + physical_token_idx"
    ) in contract
    assert "cudaMalloc" not in contract


def test_pipelined_dispatch_does_not_recompute_source_retained_route_dot() -> None:
    """The source-token reducer must be dRoute's only producer in this mode."""
    source = BACKWARD.read_text()
    begin = source.index("const auto run_pipelined_grad_y_dispatch")
    end = source.index("if constexpr (kExactBF16PipelinedGradYDispatch)", begin)
    route_prefix = source[begin:end]
    assert route_prefix.count("!kK3SourceRouteRetention") == 2
    assert "saved_down_row" in route_prefix


def test_source_retention_is_explicit_and_supports_one_range() -> None:
    """Production's full-capacity one-range launch uses the same ownership."""
    wrapper = WRAPPER.read_text()
    jit = JIT.read_text()
    cuda = BACKWARD.read_text()

    assert "mxfp8_three_term_wgrad and down_unweighted_output is None" in wrapper
    assert "requested_source_route_retention and len(backward_ranges)" not in wrapper
    assert "if not source_route_retention:" in wrapper
    assert "bool source_route_retention = false" in jit
    assert "args.source_route_retention" in jit
    begin = cuda.index("constexpr bool kK3SourceRouteRetention")
    end = cuda.index("constexpr bool kK3MxFp8WgradOverlap", begin)
    specialization = cuda[begin:end]
    assert "kMultiRangeBackward" not in specialization
    assert "kCompileW13Dgrad && kInlineWgrad" in specialization
