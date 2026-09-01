"""Static contracts for K3's phase-tagged three-range BF16 wgrad body."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JIT = ROOT / "csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe_backward.hpp"
CPP_API = ROOT / "csrc/apis/mega_backward.hpp"
PYTHON_API = ROOT / "deep_gemm/mega/backward.py"
PARENT = (
    ROOT / "deep_gemm/include/deep_gemm/impls/" "sm100_fp8_fp4_mega_moe_backward.cuh"
)
SCHEDULER = ROOT / "deep_gemm/include/deep_gemm/scheduler/gemm.cuh"


def test_host_gate_is_exact_three_range_ready_bf16() -> None:
    """Emit the experiment only for the exact three-range K3 specialization."""
    jit = JIT.read_text()
    begin = jit.index("const std::string three_segment_bf16_progressive_define")
    end = jit.index("const bool enable_k3_mxfp8_dw13_hybrid", begin)
    gate = jit[begin:end]
    assert "args.three_segment_bf16_progressive_wgrad" in gate
    assert "args.mxfp8_three_term_wgrad" not in gate
    assert "enable_k3_ready_wgrad" in gate
    assert "!enable_k3_mxfp8_two_range_exact" in gate
    assert "args.backward_ranges.num_ranges == kK3MaxBackwardRanges" in gate
    assert "!args.accumulate_wgrad" in gate
    assert "DG_EXPERIMENTAL_K3_THREE_SEGMENT_BF16_PROGRESSIVE_WGRAD" in gate


def test_progressive_bf16_and_mxfp8_hybrid_selectors_are_disjoint() -> None:
    """Keep the BF16 schedule independent from MXFP8 storage lifetimes."""
    jit = JIT.read_text()
    progressive_begin = jit.index(
        "const std::string three_segment_bf16_progressive_define"
    )
    hybrid_begin = jit.index(
        "const bool enable_k3_mxfp8_dw13_hybrid", progressive_begin
    )
    progressive = jit[progressive_begin:hybrid_begin]
    hybrid_end = jit.index(
        "const bool enable_k3_mxfp8_exact_epilogue_ring", hybrid_begin
    )
    hybrid = jit[hybrid_begin:hybrid_end]

    assert "args.three_segment_bf16_progressive_wgrad" in progressive
    assert "args.mxfp8_three_term_wgrad" not in progressive
    assert "args.mxfp8_three_term_wgrad" in hybrid
    assert "args.three_segment_bf16_progressive_wgrad" not in hybrid


def test_progressive_selector_is_per_call_across_the_public_api() -> None:
    """Expose one fail-closed argument without environment/global state."""
    python_api = PYTHON_API.read_text()
    entry_begin = python_api.index(
        "def fp8_fp4_mega_moe_backward_dgrad_swiglu("
    )
    entry_end = python_api.index("def bf16_mega_moe_backward_w2(", entry_begin)
    entry = python_api[entry_begin:entry_end]
    assert "three_segment_bf16_progressive_wgrad: bool = False" in entry
    assert (
        "if mxfp8_three_term_wgrad and "
        "three_segment_bf16_progressive_wgrad" in entry
    )
    assert "len(backward_range_sizes) != 15" in entry
    assert "os.environ" not in entry

    cpp_api = CPP_API.read_text()
    assert (
        "const bool& three_segment_bf16_progressive_wgrad = false"
        in cpp_api
    )
    assert (
        'py::arg("three_segment_bf16_progressive_wgrad") = false'
        in cpp_api
    )

    jit = JIT.read_text()
    assert "const bool& three_segment_bf16_progressive_wgrad = false" in jit
    assert (
        "!(mxfp8_three_term_wgrad &&\n"
        "          three_segment_bf16_progressive_wgrad)" in jit
    )
    assert "_p3{}" in jit


def test_progressive_jit_specialization_elides_dead_short_range_suffixes() -> None:
    """Do not retain selector-off one/two-range call graphs in this binary."""
    parent = PARENT.read_text()

    assert (
        "backward_ranges.num_ranges == kK3MaxBackwardRanges" in parent
    )
    assert parent.count(
        "kK3ThreeSegmentBF16ProgressiveWgrad ||\n"
        "                    backward_ranges.num_ranges > 1u"
    ) >= 1
    assert parent.count(
        "kK3ThreeSegmentBF16ProgressiveWgrad ||\n"
        "                    backward_ranges.num_ranges > 2u"
    ) >= 1
    assert (
        "kK3ThreeSegmentBF16ProgressiveWgrad ||\n"
        "                backward_ranges.num_ranges == 3u" in parent
    )
    assert "constexpr uint32_t kProgressiveUnionRanges" in parent
    assert "static_cast<uint64_t>(INT32_MAX) / BLOCK_M" in parent
    assert "DG_DEVICE_ASSERT(expected < kDW13ReadyBit)" not in parent


def test_scheduler_prioritizes_ready_dw13_without_new_storage() -> None:
    """Reuse one tagged retirement/cursor word and the dW2 cluster mailbox."""
    scheduler = SCHEDULER.read_text()
    provider_begin = scheduler.index(
        "struct ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider"
    )
    provider_end = scheduler.index(
        "/** Exact Kimi K3 three-segment BF16", provider_begin
    )
    provider = scheduler[provider_begin:provider_end]
    assert "kTaskBF16PhaseTagged" in provider
    assert "expert_retired_counts" in provider
    assert "kDW13ReadyBit" in provider
    assert "kDW13TaskMask" in provider
    assert "observed & kDW13ReadyBit" in provider
    assert "observed & kDW13TaskMask" in provider
    assert "kExpertRetirementTargetOffset" not in provider
    assert "expert_retirement_targets" not in provider
    assert "readiness_target" not in provider
    assert "auto* const dw13_cursors = expert_retired_counts;" in provider
    assert "const_cast" not in provider
    assert provider.index("Prefer any expert") < provider.index(
        "atomicAdd(task_cursor, kBatchTasks)"
    )
    assert "constexpr uint32_t kReadyProbeExperts = 2u" in provider
    assert "kReadyProbeDW2ExhaustedBit" in provider
    assert "kReadyProbeIndexMask" in provider
    assert "cluster_idx * kReadyProbeExperts" in provider
    assert "scan_start +" in provider
    assert "active_count <= kReadyProbeExperts" in provider
    assert "const bool complete_dw13_scan" in provider
    assert "scan_count = complete_dw13_scan" in provider
    assert "bool all_dw13_claimed = complete_dw13_scan" in provider
    assert "ptx::ld_acq(task_cursor)" not in provider
    assert "ready_probe_start |=\n                                kReadyProbeDW2ExhaustedBit" in provider
    assert "fence.proxy.async.global" in provider
    assert "cudaMalloc" not in provider


def test_parent_uses_one_unpaired_phase_tagged_umma_tma_lifetime() -> None:
    """Execute complete dW2/dW13 work in the live unified suffix."""
    parent = PARENT.read_text()
    begin = parent.index("using ThreeSegmentUnifiedBF16WgradProvider")
    end = parent.index("trace_end(17);", begin)
    unified = parent[begin:end]

    assert "ExternalKGroupedK3ThreeSegmentBF16WgradDynamicRangeProvider" in unified
    assert "kReadyDW2TasksPerExpert" in unified
    assert "kReadyRangeStateStride>;" in unified
    assert "kReadyDW13ClusterSlotWord - kReadyW13RetiredWord" not in unified
    assert "!ThreeSegmentUnifiedBF16WgradProvider::" in unified
    assert "kTaskPairedN" in unified
    assert "InitializeReleaseResources" in unified
    assert "true, 64u, true, true, false" in unified
    assert "&tensor_map_w13_wgrad_a" in unified
    assert "&tensor_map_w13_wgrad_b" in unified
    assert "&tensor_map_w13_wgrad_d" in unified
    assert "&backward_ranges" in unified
    assert "InitializeRetainResources" not in unified
    assert "ThreeSegmentReadyDW13Provider" not in unified




def test_true_varlen_reducer_acquires_each_physical_token() -> None:
    """Keep the optional streaming reducer ordered before peer-plane loads."""
    reducer_source = (
        ROOT
        / "deep_gemm/include/deep_gemm/impls/"
        "sm100_mxfp8_three_term_grouped_wgrad.cuh"
    ).read_text()
    begin = reducer_source.index("k3_mxfp8_wgrad_fixed_topk_combine(")
    end = reducer_source.index("/** Prefix words retained", begin)
    reducer = reducer_source[begin:end]
    assert "k3_multirange_physical_token_index" in reducer
    assert "ready_counts + physical_token_idx" in reducer
    assert "ptx::ld_acq_sys" in reducer
    assert reducer.index("ptx::ld_acq_sys") < reducer.index("combine_buffer)[")


def test_three_range_combine_runs_inside_the_live_unified_body() -> None:
    """Keep barrier-ordered peer communication under unified UMMA/TMA work."""
    parent = PARENT.read_text()
    gate_begin = parent.index("constexpr bool kStreamingDirectGradXCombine")
    gate_end = parent.index("auto* const direct_grad_x_pool_completions", gate_begin)
    gate = parent[gate_begin:gate_end]
    assert "!kMultiRangeBackward" in gate
    assert "kK3ThreeSegmentBF16ProgressiveWgrad" not in gate

    unified_begin = parent.index("using ThreeSegmentUnifiedBF16WgradProvider")
    unified_end = parent.index("trace_end(17);", unified_begin)
    unified = parent[unified_begin:unified_end]
    assert "!kStreamingDirectGradXCombine" in unified
    assert "true, 64u, true, true, false" in unified
    assert "&backward_ranges" in unified
    assert "nvlink_barrier" not in unified
    assert "full_grid_phase_barrier" not in unified




def test_three_range_retirement_release_acquire_proxy_chain() -> None:
    """Order every final weight read before aliased dW13 TMA stores."""
    parent = PARENT.read_text()
    retirement_begin = parent.index(
        "// The full transaction-barrier\n"
        "                                            // wait above retires both CTAs'"
    )
    retirement_end = parent.index(
        "} else {",
        retirement_begin,
    )
    retirement = parent[retirement_begin:retirement_end]
    release = retirement.index("ptx::atomic_add_rel(")
    assert retirement.index("full transaction-barrier") < release
    assert "constexpr uint32_t kReadyBit =" in retirement
    assert "1u << 31u;" in retirement
    assert "previous < kReadyBit" in retirement
    assert "range_state" not in retirement

    scheduler = SCHEDULER.read_text()
    provider_begin = scheduler.index(
        "struct ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider"
    )
    provider_end = scheduler.index(
        "/** Exact Kimi K3 three-segment BF16", provider_begin
    )
    provider = scheduler[provider_begin:provider_end]
    acquire = provider.index(
        "uint32_t observed = ptx::ld_acq(\n"
        "                            expert_retired_counts + expert);"
    )
    claim = provider.index("const uint32_t previous = atomicCAS(", acquire)
    proxy_fence = provider.index("fence.proxy.async.global", claim)
    mailbox_publish = provider.index("cluster_mailbox[1] = first;", proxy_fence)
    assert acquire < claim < proxy_fence < mailbox_publish


def test_prebiased_word_reaches_ready_once_then_becomes_bounded_cursor() -> None:
    """Model the exact packed retirement/ready/cursor arithmetic."""
    ready_bit = 1 << 31
    task_mask = ready_bit - 1
    tasks_per_expert = 336
    batch_tasks = 12

    for expected_retirements in (1, 2, 14, 45, 511, 4096):
        word = ready_bit - expected_retirements
        for retirement in range(expected_retirements):
            assert word & ready_bit == 0
            previous = word
            word += 1
            assert previous < ready_bit
            if retirement + 1 < expected_retirements:
                assert word & ready_bit == 0
        assert word == ready_bit
        assert word & task_mask == 0

        for first in range(0, tasks_per_expert, batch_tasks):
            observed = word
            assert observed & ready_bit
            assert observed & task_mask == first
            count = min(batch_tasks, tasks_per_expert - first)
            word = observed + count
        assert word & ready_bit
        assert word & task_mask == tasks_per_expert


def test_reference_scheduler_claims_every_phase_task_exactly_once() -> None:
    """Model ready-first aliasing, expert-bounded batches, waits, and termination."""
    tasks_per_dw2_expert = 168
    tasks_per_dw13_expert = 336
    batch_tasks = 12
    readiness_targets = [21, 9, 33, 15, 27, 39, 45]
    active_experts = len(readiness_targets)
    dw2_limit = active_experts * tasks_per_dw2_expert
    dw2_cursor = 0
    # Expert six is outside the first two-expert probe. It becomes visible as
    # the rotating start advances; the other experts become ready only after
    # dW2 drains and the provider switches to complete termination scans.
    ready_bit = 1 << 31
    task_mask = ready_bit - 1
    retirement_or_cursor = [
        ready_bit - target for target in readiness_targets
    ]
    retirement_or_cursor[6] += readiness_targets[6]
    delayed_readiness = list(range(6))
    claims: list[tuple[str, int, int, int]] = []
    sequence = 0

    while True:
        scan_start = (sequence * 2) % active_experts
        complete_dw13_scan = dw2_cursor >= dw2_limit
        scan_count = (
            active_experts
            if complete_dw13_scan
            else min(active_experts, 2)
        )
        all_dw13_claimed = complete_dw13_scan
        claim: tuple[str, int, int, int] | None = None
        for scan in range(scan_count):
            expert = (scan_start + scan) % active_experts
            observed = retirement_or_cursor[expert]
            if observed & ready_bit == 0:
                all_dw13_claimed = False
                continue
            local_first = observed & task_mask
            if local_first < tasks_per_dw13_expert:
                all_dw13_claimed = False
            if local_first < tasks_per_dw13_expert:
                count = min(
                    batch_tasks,
                    tasks_per_dw13_expert - local_first,
                )
                claim = ("dw13", expert, local_first, count)
                retirement_or_cursor[expert] += count
                break

        if claim is None and dw2_cursor < dw2_limit:
            expert = dw2_cursor // tasks_per_dw2_expert
            expert_end = (expert + 1) * tasks_per_dw2_expert
            count = min(batch_tasks, dw2_limit - dw2_cursor, expert_end - dw2_cursor)
            claim = ("dw2", expert, dw2_cursor, count)
            dw2_cursor += count

        if claim is not None:
            claims.append(claim)
            sequence += 1
            continue
        if dw2_cursor == dw2_limit and all_dw13_claimed:
            break

        # This is the publisher's nanosleep path: no terminal mailbox is
        # published while W13 readers can still make progress.
        expert = delayed_readiness.pop(0)
        assert retirement_or_cursor[expert] == (
            ready_bit - readiness_targets[expert]
        )
        retirement_or_cursor[expert] += readiness_targets[expert]
        assert retirement_or_cursor[expert] == ready_bit

    assert claims[0][0] == "dw2"
    assert any(phase == "dw13" and expert == 6 for phase, expert, _, _ in claims)
    dw2_tasks = {
        task
        for phase, _expert, first, count in claims
        if phase == "dw2"
        for task in range(first, first + count)
    }
    assert dw2_tasks == set(range(dw2_limit))
    for expert in range(active_experts):
        dw13_tasks = {
            task
            for phase, claim_expert, first, count in claims
            if phase == "dw13" and claim_expert == expert
            for task in range(first, first + count)
        }
        assert dw13_tasks == set(range(tasks_per_dw13_expert))
    assert all(count <= batch_tasks for _phase, _expert, _first, count in claims)
    assert all(
        first // tasks_per_dw2_expert
        == (first + count - 1) // tasks_per_dw2_expert
        for phase, _expert, first, count in claims
        if phase == "dw2"
    )


def test_cluster_strided_ready_probes_cover_all_k3_experts_immediately() -> None:
    """Cover all experts with half the prior per-generation acquire probes."""
    active_experts = 112
    clusters = 148 // 2
    probe_experts = 2

    covered = {
        (cluster * probe_experts + offset) % active_experts
        for cluster in range(clusters)
        for offset in range(probe_experts)
    }
    probe_hits = {
        expert: sum(
            (cluster * probe_experts + offset) % active_experts == expert
            for cluster in range(clusters)
            for offset in range(probe_experts)
        )
        for expert in range(active_experts)
    }
    previous_probe_loads = {
        (cluster * 4 + offset) % active_experts
        for cluster in range(clusters)
        for offset in range(4)
    }

    assert covered == set(range(active_experts))
    assert min(probe_hits.values()) == 1
    assert max(probe_hits.values()) == 2
    assert clusters * probe_experts == 148
    assert len(previous_probe_loads) == active_experts
    assert clusters * 4 == 296

    for sparse_active_experts in range(1, active_experts + 1):
        sparse_covered = {
            (cluster * probe_experts + offset) % sparse_active_experts
            for cluster in range(clusters)
            for offset in range(min(sparse_active_experts, probe_experts))
        }
        assert sparse_covered == set(range(sparse_active_experts))
