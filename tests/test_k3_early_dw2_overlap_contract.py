"""Static contracts for Kimi-K3's branch-major early-dW2 overlap."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT / "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"


def _between(text: str, begin: str, end: str) -> str:
    """Return a fail-closed source slice delimited by exact markers."""
    start = text.index(begin)
    return text[start : text.index(end, start)]


def test_two_range_branch_major_path_enters_dw2_after_local_w13() -> None:
    """Keep dW2 inside the parent kernel and ahead of the terminal suffix."""
    parent = PARENT.read_text()
    predicate = parent.index("kK3BranchMajorBF16EarlyDW2Overlap")
    w13_retired = parent.index("trace_end(21)", predicate)
    early_dw2 = parent.index(
        "run_branch_major_early_dw2.template operator()<true>()", w13_retired
    )
    terminal_suffix = parent.index("trace_begin(20)", early_dw2)

    assert w13_retired < early_dw2 < terminal_suffix
    assert "kMultiRangeBackward" in parent[predicate:w13_retired]
    assert "DynamicTwoSegmentDW2Provider" in parent[predicate:early_dw2]


def test_early_dw2_retains_resources_for_terminal_dw13() -> None:
    """Prevent dW2 from releasing shared/TMEM resources before dW13 reuses them."""
    parent = PARENT.read_text()
    begin = parent.index("const auto run_branch_major_early_dw2")
    end = parent.index("#endif", begin)
    overlap = parent[begin:end]

    assert "Sm100Bf16GemmBatchResourceHooks<true, false>" in overlap
    assert "kReleaseBatchResources" in overlap
    assert "DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_EARLY_DW2_OVERLAP" in parent
    assert "kTerminalDW13ExtraCombineThreads" in parent
    assert "? 0u" in parent
    assert "!kK3BranchMajorBF16EarlyDW2Overlap" in parent[end:]


def test_jit_selects_early_overlap_only_for_rank_uniform_two_range_buckets() -> None:
    """Prevent three-range or cache-grown arenas from selecting early dW2."""
    jit = JIT.read_text()
    selection = _between(
        jit,
        "const int64_t first_backward_range_capacity =",
        "const bool enable_k3_mxfp8_three_term_wgrad =",
    )
    assert "backward_range_sizes[1]" in selection
    assert "num_backward_ranges == 2" in selection
    assert "first_backward_range_capacity > 0" in selection
    assert "first_backward_range_capacity <= 65536" in selection
    assert "num_pool_rows" not in selection
    assert "num_acts_rows" not in selection

    generation = _between(
        jit,
        "const std::string ready_wgrad_defines =",
        "const std::string two_segment_bf16_progressive_define =",
    )
    assert "#define DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS 8" in generation
    assert "#define DG_EXPERIMENTAL_K3_READY_WGRAD 1" in generation
    assert "DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_EARLY_DW2_OVERLAP 1" in generation
    assert "_d4{}_e{}_mxfp8wgrad{}" in jit


def test_early_dw2_uses_the_published_union_and_skips_only_terminal_dw2() -> None:
    """Bind the early consumer to the existing scheduler and dW13 handoff."""
    parent = PARENT.read_text()
    overlap = _between(
        parent,
        "const auto run_branch_major_early_dw2",
        "// Co-schedule a bounded prefix",
    )
    for required in (
        "DG_DEVICE_ASSERT(backward_ranges.num_ranges == 2u)",
        "weight_tile_states + kReadyTerminalUnionStateWord",
        "union_state + kReadyDW2CursorWord",
        "union_state[kReadyDW2TasksWord]",
        "kReadyDW2ClusterSlotWord",
        "ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider",
        "Sm100Bf16GemmBatchResourceHooks<true, false>",
    ):
        assert required in overlap

    terminal = _between(
        parent,
        "if constexpr (kK3BranchMajorBF16DynamicTail) {",
        "} else {\n                using DynamicDW2Provider",
    )
    assert (
        "!kEarlyW2Wgrad &&\n"
        "                        !kK3BranchMajorBF16EarlyDW2Overlap"
    ) in terminal
    assert "DynamicTwoSegmentDW13Provider" in terminal
    assert "DynamicDW13ReleaseResources" in terminal
