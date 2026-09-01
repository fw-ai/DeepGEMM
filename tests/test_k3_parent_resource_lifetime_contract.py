from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)
BF16 = ROOT / "deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh"
MXFP8 = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/"
    "sm100_mxfp8_three_term_grouped_wgrad.cuh"
)


def _between(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    return text[start : text.index(end, start)]


def test_w13_reinitializes_only_invalid_parent_barriers() -> None:
    """Every selected W13 init must consume an invalid mbarrier object."""
    parent = PARENT.read_text()
    reset = _between(
        parent,
        "trace_begin(15);\n            comm::cluster_sync_with_relaxed_arrive();",
        "if constexpr (kEarlyW2Wgrad) {",
    )
    assert (
        "!kEarlyW2Wgrad &&\n"
        "                    !kK3MxFp8ExactEpilogueRing"
    ) in reset
    assert (
        "kResidualMXFP8Dgrad &&\n"
        "                    !kEarlyW2Wgrad"
    ) not in reset

    invalidation = reset.index("for (uint32_t i = 0; i < kNumStages; ++i)")
    pipeline_init = reset.index("full_barriers[i]->init(")
    epilogue_invalidation = reset.index(
        "i < kNumEpilogueStages; ++i)", invalidation
    )
    epilogue_init = reset.index("tmem_full_barriers[i]->init(1)")
    assert invalidation < pipeline_init
    assert epilogue_invalidation < epilogue_init

    # Residual-only objects retain their narrower compile-time ownership gate;
    # the parent pipeline and epilogue barriers do not depend on that mode.
    residual = _between(
        reset,
        "if constexpr (kResidualMXFP8Dgrad) {",
        "#pragma unroll\n                    for (uint32_t i = 0;\n"
        "                         i < kNumEpilogueStages; ++i)",
    )
    for name in (
        "weight_load_barrier",
        "primary_mma_barrier",
        "residual_mma_barrier",
    ):
        assert name in residual


def test_exact_ring_dispatch_is_invalidated_exactly_once() -> None:
    """The three-range ring must not execute mbarrier.inval twice."""
    parent = PARENT.read_text()
    early = _between(
        parent,
        "if constexpr (kK3MxFp8ExactEpilogueRing) {\n"
        "            // Both cohorts enter one readiness-driven grouped stream.",
        "const uint32_t dw13_epoch = launch_epoch ^ 0x80000000u;",
    )
    assert "dispatch_barriers[i]" in early

    terminal = _between(
        parent,
        "// Initial-consumer clusters have no W13 tiles and arrive here",
        "static_assert(\n                !kUseReducedW2ProducerSet",
    )
    guarded_dispatch = _between(
        terminal,
        "if constexpr (!kK3MxFp8ExactEpilogueRing) {",
        "constexpr uint32_t kNumLiveDequantBarriers =",
    )
    assert guarded_dispatch.count("dispatch_barriers[i]") == 1
    assert "already-invalid object is UB" in guarded_dispatch
    assert terminal.count("dispatch_barriers[i]") == 1

    # The only intervening dispatch reinitialization belongs to the mutually
    # exclusive early-dW2 schedule (which requires residual-weight building).
    early_w2 = _between(
        parent,
        "if constexpr (kEarlyW2Wgrad) {\n"
        "                if constexpr (!kBF16Mode)",
        "if constexpr (kSplitResidualWeightCache) {",
    )
    assert early_w2.count("dispatch_barriers[i]->init") == 1


@dataclass
class _SelectedLifetime:
    valid: set[str] = field(
        default_factory=lambda: {
            "parent_pipeline",
            "parent_epilogue",
            "parent_dispatch",
        }
    )
    tmem_allocated: bool = True
    events: list[str] = field(default_factory=list)

    def invalidate(self, resource: str, site: str) -> None:
        assert resource in self.valid, f"double invalidation: {resource} at {site}"
        self.valid.remove(resource)
        self.events.append(f"inval:{resource}:{site}")

    def initialize(self, resource: str, site: str) -> None:
        assert resource not in self.valid, f"init-on-valid: {resource} at {site}"
        self.valid.add(resource)
        self.events.append(f"init:{resource}:{site}")

    def free_tmem(self, site: str) -> None:
        assert self.tmem_allocated, f"double TMEM free at {site}"
        self.tmem_allocated = False
        self.events.append(f"free:tmem:{site}")

    def run_default_bf16_body(self, name: str) -> None:
        assert not self.tmem_allocated, f"TMEM still owned before {name}"
        self.tmem_allocated = True
        self.events.append(f"alloc:tmem:{name}")
        self.tmem_allocated = False
        self.events.append(f"free:tmem:{name}")


def _trace_selected_range_count(num_ranges: int) -> _SelectedLifetime:
    assert num_ranges in (1, 2, 3)
    exact_ring = num_ranges == 3
    lifetime = _SelectedLifetime()
    lifetime.valid.add("parent_dequant_w2")
    if not exact_ring:
        lifetime.valid.add("parent_dequant_w13")

    if exact_ring:
        for resource in (
            "parent_pipeline",
            "parent_epilogue",
            "parent_dispatch",
        ):
            lifetime.invalidate(resource, "early_exact_ring")

    # kEarlyW2Wgrad is false in all selected no-residual K3 training modes.
    if not exact_ring:
        lifetime.invalidate("parent_pipeline", "w2_to_w13")
        lifetime.invalidate("parent_epilogue", "w2_to_w13")
    lifetime.initialize("parent_pipeline", "w13")
    lifetime.initialize("parent_epilogue", "w13")

    lifetime.invalidate("parent_pipeline", "terminal")
    lifetime.invalidate("parent_epilogue", "terminal")
    if not exact_ring:
        lifetime.invalidate("parent_dispatch", "terminal")
    lifetime.invalidate("parent_dequant_w2", "terminal")
    if not exact_ring:
        lifetime.invalidate("parent_dequant_w13", "terminal")
    lifetime.free_tmem("parent_terminal")

    if exact_ring:
        # Three physical ranges share one retained BF16 lifetime; the final
        # range releases it. The early exact dW13 body retained parent TMEM.
        lifetime.run_default_bf16_body("three_range_dw2")
    else:
        # Terminal branch-major dW2 and dW13 own independent lifetimes with a
        # caller cluster join between their free and subsequent allocation.
        lifetime.run_default_bf16_body("dw2")
        lifetime.run_default_bf16_body("dw13")

    assert not lifetime.valid
    assert not lifetime.tmem_allocated
    return lifetime


def test_selected_one_two_three_range_resource_event_traces() -> None:
    traces = {
        ranges: _trace_selected_range_count(ranges).events
        for ranges in (1, 2, 3)
    }
    assert traces[1] == traces[2]
    assert "inval:parent_pipeline:w2_to_w13" in traces[1]
    assert "inval:parent_dispatch:terminal" in traces[2]
    assert "inval:parent_dispatch:early_exact_ring" in traces[3]
    assert "inval:parent_dispatch:terminal" not in traces[3]
    for trace in traces.values():
        assert trace.count("inval:parent_dequant_w2:terminal") == 1
    assert traces[1].count("inval:parent_dequant_w13:terminal") == 1
    assert traces[2].count("inval:parent_dequant_w13:terminal") == 1
    assert "inval:parent_dequant_w13:terminal" not in traces[3]
    assert traces[3].count("free:tmem:parent_terminal") == 1


def test_tmem_handoffs_keep_cluster_ordering_and_single_ownership() -> None:
    parent = PARENT.read_text()
    bf16 = BF16.read_text()
    mxfp8 = MXFP8.read_text()

    terminal = _between(
        parent,
        "if constexpr (\n"
        "            !kK3MxFp8WgradOverlap &&\n"
        "            !kK3BranchMajorBF16EarlyDW2Overlap) {\n"
        "            comm::cluster_sync_with_relaxed_arrive();",
        "if constexpr (\n"
        "            kK3BranchMajorBF16WgradTail &&",
    )
    assert terminal.index("comm::cluster_sync_with_relaxed_arrive()") < (
        terminal.index("Allocator().free(0, kNumTmemCols)")
    )

    assert (
        "using Sm100Bf16GemmDefaultBatchResourceHooks =\n"
        "    Sm100Bf16GemmBatchResourceHooks<true, true>;"
    ) in bf16
    assert bf16.index("comm::cluster_sync_with_relaxed_arrive()") < (
        bf16.index("Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem)")
    )

    for required in (
        "Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks",
        "!TerminalResources::kAllocateTmem",
        "!TerminalResources::kFreeTmem",
    ):
        assert required in parent
    assert (
        "Sm100K3MxFp8WgradBatchResourceHooks<\n"
        "        true, true, true, false, false>"
    ) in mxfp8


def test_terminal_retires_dedicated_dequant_barriers_before_bf16_alias() -> None:
    """The one/two-range handoff must retire both initial materializers."""
    parent = PARENT.read_text()
    terminal = _between(
        parent,
        "trace_begin(20);\n"
        "        if constexpr (\n"
        "            kInlineWgrad && !kK3MxFp8WgradOverlap &&\n"
        "            !kK3BranchMajorBF16EarlyDW2Overlap) {",
        "if constexpr (\n"
        "            !kK3MxFp8WgradOverlap &&",
    )
    guard = _between(
        terminal,
        "if constexpr (\n                !kBF16Mode && !kInlineWeightDequant",
        "          }\n        }",
    )
    for required in (
        "!kResidualMXFP8Dgrad",
        "kOverlapInitialBF16WeightDequant ||",
        "kPhaseOrderedWeightDequant",
        "dequant_barriers + i",
    ):
        assert required in guard

    dispatch_invalidation = terminal.index("dispatch_barriers[i]")
    dequant_invalidation = terminal.index("dequant_barriers + i")
    parent_free = parent.index(
        "Allocator().free(0, kNumTmemCols)",
        parent.index(terminal),
    )
    assert dispatch_invalidation < dequant_invalidation
    assert parent.index("dequant_barriers + i", parent.index(terminal)) < parent_free


def test_exact_ring_retires_only_initialized_dequant_barriers() -> None:
    """Inline W13 dequant leaves only the W2 materializer mbarrier live."""
    parent = PARENT.read_text()
    terminal = _between(
        parent,
        "// Initial-consumer clusters have no W13 tiles and arrive here",
        "static_assert(\n                !kUseReducedW2ProducerSet",
    )
    dequant = _between(
        terminal,
        "constexpr uint32_t kNumLiveDequantBarriers =",
        "            }\n            comm::cluster_sync_with_relaxed_arrive();",
    )
    assert "kInlineW13WeightDequant ? 1u : 2u" in dequant
    assert "i < kNumLiveDequantBarriers" in dequant
    assert "i < 2u" not in dequant

    initial = _between(
        parent,
        "if constexpr (kOverlapInitialBF16WeightDequant) {",
        "// Every role walks this deterministic schedule independently.",
    )
    assert "dequant_noninline_weights_once(false, false, true)" in initial
    assert "if constexpr (!kInlineW13WeightDequant)" in initial
    assert "dequant_noninline_weights_once(true, false, true)" in initial
