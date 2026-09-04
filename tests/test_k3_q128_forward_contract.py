"""Static contracts for Kimi-K3's Q128 MegaMoE forward specialization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEURISTIC = ROOT / "csrc/jit_kernels/heuristics/mega_moe.hpp"
KERNEL = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"
)
BACKWARD = (
    ROOT
    / "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)


def _between(text: str, begin: str, end: str) -> str:
    """Return a fail-closed source slice delimited by exact markers."""
    start = text.index(begin)
    return text[start : text.index(end, start)]


def test_store64_three_warpgroups_are_limited_to_k3_ep8_q128() -> None:
    """Prevent the performance tile from changing Q32 or unrelated models."""
    source = HEURISTIC.read_text()
    selection = _between(
        source,
        "if (mma_kind == MmaKind::MXFP8FP4 &&",
        "const int block_n = 128;",
    )

    for required in (
        "l2_activation_group_size == 128",
        "num_ranks == 8",
        "num_experts == 896",
        "num_topk == 16",
        "hidden == 3584",
        "intermediate_hidden == 3072",
        "block_m == 192",
        "store_block_m = 64",
        "num_epilogue_threads = 384",
    ):
        assert required in selection


def test_three_warpgroups_use_deadlock_safe_register_budgets() -> None:
    """Require each non-epilogue role to release registers before allocation."""
    source = KERNEL.read_text()
    registers = _between(
        source,
        "constexpr bool kUseMoreEpilogueRegisters",
        "// Grid sync index assignments",
    )

    for required in (
        "constexpr bool kUseK3Q128ThreeEpilogueWarpgroups",
        "kL2ActivationGroup128",
        "RouteWeightMode::PostDown",
        "kHidden == 3584",
        "kIntermediateHidden == 3072",
        "kNumExperts == 896",
        "kNumTopk == 16",
        "kNumRanks == 8",
        "kNumEpilogueThreads == 384",
        "kUseK3Q128ThreeEpilogueWarpgroups ? 88",
        "kUseK3Q128ThreeEpilogueWarpgroups ? 80",
        "kUseK3Q128ThreeEpilogueWarpgroups ? 104",
    ):
        assert required in registers
    assert "<= 64512" in registers


def test_q128_epilogue_keeps_cluster_reduction_and_bf16_boundary() -> None:
    """Keep Q128 scaling and training's post-SiTU BF16 rounding point."""
    source = KERNEL.read_text()
    q128 = _between(
        source,
        "if constexpr (kL2ActivationGroup128) {",
        "// Cast to FP8 E4M3 and store into shared memory",
    )
    boundary = _between(
        source,
        "if constexpr (\n"
        "                                kActivationType == ActivationType::SiTU &&\n"
        "                                kRouteWeightMode ==",
        "// Amax reduction (thread-level)",
    )

    assert "store_cluster_float2_async" in q128
    assert "arrive_expect_tx_cluster" in q128
    assert "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes" in source
    assert "mbarrier.arrive.expect_tx.release.cluster.shared::cluster" in source
    assert "l1_scale_barriers" in q128
    assert "kQ128PeerAmaxBytes" in q128
    assert "ptx::sync_aligned" in q128
    assert ".wait(l1_scale_phase)" in q128
    assert "l1_scale_phase ^= 1u" in q128
    assert "__float22bfloat162_rn" in boundary
    assert "RouteWeightMode::PostDown" in boundary


def test_post_down_combine_hoists_route_weight_load() -> None:
    """Keep one route-weight load per slot instead of one load per chunk."""
    source = KERNEL.read_text()
    combine = _between(
        source,
        "const int stored_topk_slot_idx",
        "// Cast",
    )

    assert "float stored_route_weight = 1.0f;" in combine
    assert "input_topk_weights_buffer.get_base_ptr<float>()" in combine
    assert "__shfl_sync" in combine
    assert combine.count("input_topk_weights_buffer.get_base_ptr<float>()") == 1


def test_situ_fast_intrinsics_reuse_training_identity() -> None:
    """Bind optimized arithmetic to the exact EP8 K3 Q128 training contract."""
    source = KERNEL.read_text()
    situ = _between(
        source,
        "auto gate_numerator = gate;",
        "if constexpr (\n                                kRouteWeightMode ==",
    )

    for required in (
        "constexpr bool kUseK3Q128FastSiTU",
        "kL2ActivationGroup128",
        "RouteWeightMode::PostDown",
        "kHidden == 3584",
        "kIntermediateHidden == 3072",
        "kNumExperts == 896",
        "kNumTopk == 16",
        "kNumRanks == 8",
        "kSituBeta == 4.0f",
        "kSituLinearBeta == 25.0f",
    ):
        assert required in situ
    assert "kFastMath || kUseK3Q128FastSiTU" in situ
    assert "!kFastMath && kUseK3Q128FastSiTU" in situ
    assert "__tanhf(gate.x / kSituBeta)" in situ
    assert situ.count("tanhf(gate.x / kSituBeta)") == 2
    assert "__tanhf(up.x / kSituLinearBeta)" in situ
    assert situ.count("tanhf(up.x / kSituLinearBeta)") == 2
    assert "const auto gate_half_denom" in situ
    assert "math::fast_rcp" in situ
    assert "const auto tanh_gate_half" in situ
    assert "situ_sigmoid" in situ


def test_prepared_backward_keeps_canonical_gate_up_layout() -> None:
    """Lock the verified canonical Forward-save ABI consumed by Backward."""
    source = BACKWARD.read_text()
    alias = _between(
        source,
        "// Legacy FP8/FP4 replay materializes MegaMoE's",
        "uint32_t dispatch_pull_mbarrier_phase",
    )
    prefetch = _between(
        source,
        "const auto prefetch_gate_up_segment",
        "prefetch_gate_up_segment(0u);",
    )
    derivative = _between(
        source,
        "const uint32_t gate_col =\n"
        "                                (kBF16Mode || kGateUpPrepared)",
        "const float gate_unclamped",
    )

    assert "canonical [all gate | all up]" in alias
    assert "inplace_gate_up_grad && !kBF16Mode && !kGateUpPrepared" in alias
    assert "gate_up_output + row_base +\n                                        hidden_col" in prefetch
    assert "kIntermediateHidden + hidden_col" in prefetch
    assert derivative.count("kBF16Mode || kGateUpPrepared") == 2
    assert ": chunk * 16 + in_chunk" in derivative
    assert ": gate_col + 8" in derivative
