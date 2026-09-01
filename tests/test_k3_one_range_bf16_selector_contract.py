"""Source contracts for the K3 EP=8 one-range BF16 backward selector."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JIT_HEADER = ROOT / (
    "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
)
DEVICE_HEADER = ROOT / (
    "deep_gemm/include/deep_gemm/impls/"
    "sm100_fp8_fp4_mega_moe_backward.cuh"
)
SCHEDULER_HEADER = ROOT / "deep_gemm/include/deep_gemm/scheduler/gemm.cuh"


def _region(source: str, start: str, end: str) -> str:
    """Return one fail-closed source interval bounded by unique markers."""
    assert source.count(start) == 1, start
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def test_one_range_host_selector_is_narrow_and_excludes_mxfp8_wgrad() -> None:
    """Select exact BF16 dynamic wgrad only for the eligible K3 EP=8 call."""
    source = JIT_HEADER.read_text()
    selector = _region(
        source,
        "const bool enable_k3_branch_major_bf16_wgrad_tail =",
        "const bool enable_k3_mxfp8_dw13_hybrid =",
    )
    for contract in (
        "mxfp8_three_term_wgrad &&",
        "k3_mxfp8_three_term_wgrad_eligible &&",
        "num_ranks == 8 &&",
        "(!multi_range_backward || num_backward_ranges == 2)",
        "!accumulate_wgrad &&",
        "!inline_weight_dequant && !phase_ordered_weight_dequant &&",
        "!residual_mxfp8_dgrad",
        "!enable_k3_branch_major_bf16_wgrad_tail",
    ):
        assert contract in selector
    assert (
        "!enable_k3_branch_major_bf16_wgrad_tail ||\n"
        "        !enable_k3_mxfp8_three_term_wgrad"
    ) in selector

    generated_defines = _region(
        source,
        "const std::string branch_major_bf16_wgrad_tail_define =",
        "const bool enable_k3_mxfp8_exact_epilogue_ring =",
    )
    assert "DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL 1" in (
        generated_defines
    )
    assert "DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL 1" in (
        generated_defines
    )
    assert "getenv" not in source


def test_one_range_device_suffix_keeps_comm_inside_bf16_umma_tma_body() -> None:
    """Pin terminal queues, resource reuse, and in-kernel publication roles."""
    source = DEVICE_HEADER.read_text()
    suffix = _region(
        source,
        "if constexpr (kK3BranchMajorBF16DynamicTail) {",
        "} else\n#endif",
    )
    for contract in (
        "constexpr uint32_t kDynamicBatchTasks = 4u;",
        "Sm100Bf16GemmBatchResourceHooks<true, false>",
        "Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<>",
        "ExternalKGroupedTerminalDynamicRangeProvider<",
        "kPairedDW2TasksPerExpert",
        "kPairedDW13TasksPerExpert",
        "DynamicDW2RetainedResources,\n                        kPublishRemoteGradients",
        "DynamicDW13ReleaseResources,\n                    kDirectRemoteGradX",
        "kDirectRemoteGradX ? 64u : 0u",
        "weight_tile_states + kReadyDW2CursorWord",
        "weight_tile_states + kReadyDW13CursorWord",
    ):
        assert contract in suffix
    assert "cudaMalloc" not in suffix
    assert "nccl" not in suffix.lower()


def test_terminal_provider_uses_fixed_atomic_claim_and_role_mailbox() -> None:
    """Keep variable-expert work balancing allocation-free and cluster local."""
    source = SCHEDULER_HEADER.read_text()
    provider = _region(
        source,
        "struct ExternalKGroupedTerminalDynamicRangeProvider {",
        "struct ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider {",
    )
    for contract in (
        "kCompleteAcquireMask",
        "while (ptx::ld_acq(cluster_mailbox)",
        "first = atomicAdd(task_cursor, kBatchTasks);",
        "st.release.gpu.global.u32",
        "ptx::red_or_rel_gpu(",
        "Decoder::decode_range_task(",
    ):
        assert contract in provider
    assert "new " not in provider
    assert "malloc" not in provider.lower()
