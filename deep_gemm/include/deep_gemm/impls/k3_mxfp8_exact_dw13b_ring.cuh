#pragma once

#include <deep_gemm/impls/k3_mxfp8_exact_epilogue_pipeline.cuh>

namespace deep_gemm {

/** Shared-memory staging for one canonical-pool dW13 A/B group-32 load.
 *
 * One elected loader issues a descriptor-matched 32x128 BF16 TMA from either
 * `grad_gate_up_output` or `x_pool_output`.  The resulting tile is exactly the
 * input expected by the established group-32 exact-two-term quantizer.  Each
 * feature thread explicitly zeros rows outside the source block's true routed
 * count before quantization; correctness therefore does not depend on either
 * canonical pool having initialized its padding.  This object is CTA-local
 * scratch and never changes the kernel's global allocation or public ABI.
 */
struct alignas(128) K3MxFp8DW13BGatherStage {
    cutlass::bfloat16_t source[
        kK3MxFp8EpilogueRowsPerGroup *
        kK3MxFp8EpilogueFeaturePanel];
    alignas(8) cutlass::arch::ClusterTransactionBarrier load_barrier;
    // One engine leader publishes the next atomic-cursor work record here;
    // the engine's named barrier is its release/acquire edge.
    volatile uint32_t work[8];
};

static_assert(
    offsetof(K3MxFp8DW13BGatherStage, load_barrier) == 8192u);
static_assert(offsetof(K3MxFp8DW13BGatherStage, work) == 8200u);
static_assert(sizeof(K3MxFp8DW13BGatherStage) == 8320u);

/** Dense block-major panel order shared by the decoupled A/B publisher.
 *
 * One cursor task owns all six group-32 tickets in a feature-128 panel. A
 * four-task claim therefore owns 24 tickets, but cannot cross A/B or physical
 * block boundaries because 48 A panels and 28 B panels are both divisible by
 * four. Engines may complete already-claimed panels out of order; wrapped
 * remain safe because slot opening waits for the preceding generation's
 * terminal retirement.
 */
struct K3MxFp8DW13ABBlockMajorSchedule {
    static constexpr uint32_t kAPanelsPerBlock =
        K3MxFp8ExactDW13AEpilogueRingLayout::kFeaturePanels;
    static constexpr uint32_t kBPanelsPerBlock =
        K3MxFp8ExactDW13BEpilogueRingLayout::kFeaturePanels;
    static constexpr uint32_t kGroupsPerPanel =
        K3MxFp8ExactDW13AEpilogueRingLayout::kGroupsPerBlock;
    static constexpr uint32_t kPanelBundlesPerBlock =
        kAPanelsPerBlock + kBPanelsPerBlock;
    static constexpr uint32_t kPanelBundlesPerClaim = 4u;
    static_assert(
        kAPanelsPerBlock % kPanelBundlesPerClaim == 0u &&
        kBPanelsPerBlock % kPanelBundlesPerClaim == 0u);

    static constexpr uint32_t bundle_index(
            uint32_t production_ordinal, bool is_b,
            uint32_t feature_panel) {
        const uint32_t local = is_b
            ? kAPanelsPerBlock + feature_panel
            : feature_panel;
        return production_ordinal * kPanelBundlesPerBlock + local;
    }

    static constexpr uint32_t bundle_production_ordinal(uint32_t bundle) {
        return bundle / kPanelBundlesPerBlock;
    }

    static constexpr uint32_t local_bundle(uint32_t bundle) {
        return bundle % kPanelBundlesPerBlock;
    }

    static constexpr bool bundle_is_b(uint32_t bundle) {
        return local_bundle(bundle) >= kAPanelsPerBlock;
    }

    static constexpr uint32_t operand_local_bundle(uint32_t bundle) {
        const uint32_t local = local_bundle(bundle);
        return bundle_is_b(bundle) ? local - kAPanelsPerBlock : local;
    }

    static constexpr uint32_t bundle_feature_panel(uint32_t bundle) {
        return operand_local_bundle(bundle);
    }

    static constexpr uint32_t num_bundles(uint32_t compact_blocks) {
        return compact_blocks * kPanelBundlesPerBlock;
    }
};

static_assert(K3MxFp8DW13ABBlockMajorSchedule::kAPanelsPerBlock == 48u);
static_assert(K3MxFp8DW13ABBlockMajorSchedule::kBPanelsPerBlock == 28u);
static_assert(K3MxFp8DW13ABBlockMajorSchedule::kGroupsPerPanel == 6u);
static_assert(
    K3MxFp8DW13ABBlockMajorSchedule::kPanelBundlesPerBlock == 76u);
static_assert(
    K3MxFp8DW13ABBlockMajorSchedule::bundle_production_ordinal(
        K3MxFp8DW13ABBlockMajorSchedule::bundle_index(
            7u, true, 27u)) == 7u);
static_assert(K3MxFp8DW13ABBlockMajorSchedule::bundle_is_b(
    K3MxFp8DW13ABBlockMajorSchedule::bundle_index(
        7u, true, 27u)));
static_assert(
    K3MxFp8DW13ABBlockMajorSchedule::bundle_feature_panel(
        K3MxFp8DW13ABBlockMajorSchedule::bundle_index(
            7u, true, 27u)) == 27u);

/** Packed canonical-pool readiness published once per physical block.
 *
 * SiTU contributes one low-byte arrival for each of its 24 intermediate
 * feature tiles after the complete gate/up block is visible.  Exact-X
 * dispatch contributes one next-byte arrival for each of the 192 physical
 * rows after `x_pool_output` is visible.  The parent clears the complete word
 * before either producer starts, so the upper 16 bits remain zero and no
 * process-global epoch is required.
 */
constexpr uint32_t kK3MxFp8DW13GUBlockReadyTarget = 24u;
constexpr uint32_t kK3MxFp8DW13XBlockReadyTarget = 192u;
constexpr uint32_t kK3MxFp8DW13ABBlockReadyWord =
    kK3MxFp8DW13GUBlockReadyTarget |
    (kK3MxFp8DW13XBlockReadyTarget << 8u);

CUTLASS_HOST_DEVICE constexpr bool
k3_mxfp8_dw13_ab_block_ready(uint32_t word) {
    return word == kK3MxFp8DW13ABBlockReadyWord;
}

static_assert(kK3MxFp8DW13ABBlockReadyWord == 0x0000c018u);

/** True rows in one group-32 tile of a possibly partial 192-row block. */
CUTLASS_HOST_DEVICE constexpr uint32_t
k3_mxfp8_dw13_ab_group_valid_rows(
        uint32_t block_valid_rows, uint32_t group_in_block) {
    const uint32_t row_begin =
        group_in_block * kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t remaining = row_begin < block_valid_rows
        ? block_valid_rows - row_begin : 0u;
    return remaining < kK3MxFp8EpilogueRowsPerGroup
        ? remaining : kK3MxFp8EpilogueRowsPerGroup;
}

static_assert(k3_mxfp8_dw13_ab_group_valid_rows(17u, 0u) == 17u);
static_assert(k3_mxfp8_dw13_ab_group_valid_rows(17u, 1u) == 0u);
static_assert(k3_mxfp8_dw13_ab_group_valid_rows(192u, 5u) == 32u);

/** Physical source coordinates for one compact expert-major block. */
struct K3MxFp8DW13ABSourceBlock {
    uint32_t expert;
    uint32_t range_index;
    uint32_t compact_block_ordinal;
    uint32_t expert_local_block;
    uint32_t physical_pool_block;
    uint32_t valid_rows;
};

/** Resolve one compact expert-major ordinal into its varlen physical block.
 *
 * Compact K concatenates each expert's ranges in reverse range order, while
 * the source pools remain range-major.  Prefixes are all 192-row aligned, so
 * a compact block belongs to exactly one `(expert, range, physical block)`.
 * The returned `valid_rows` is the true routed count in the final block and
 * is never inferred from another rank's nominal sequence length.
 */
template <uint32_t kNumExperts, uint32_t kPoolBlockRows,
          uint32_t kMaxRanges>
CUTLASS_DEVICE K3MxFp8DW13ABSourceBlock
k3_mxfp8_dw13_ab_source_block(
        uint32_t compact_block_ordinal,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t* value_prefix,
        const uint32_t* physical_range_prefix) {
    static_assert(kPoolBlockRows == 192u);
    const uint32_t compact_row =
        compact_block_ordinal * kPoolBlockRows;
    const uint32_t expert =
        detail::k3_mxfp8_find_prefix_owner<
            kNumExperts, kMaxRanges>(value_prefix, compact_row);
    DG_DEVICE_ASSERT(
        expert < kNumExperts &&
        compact_row >= value_prefix[expert] &&
        compact_row < value_prefix[expert + 1u] &&
        value_prefix[expert] % kPoolBlockRows == 0u &&
        value_prefix[expert + 1u] % kPoolBlockRows == 0u);
    uint32_t expert_local_block =
        (compact_row - value_prefix[expert]) / kPoolBlockRows;
    const uint32_t original_expert_local_block = expert_local_block;
    #pragma unroll
    for (uint32_t reverse_iteration = 0u;
         reverse_iteration < kMaxRanges; ++reverse_iteration) {
        if (reverse_iteration >= backward_ranges.num_ranges)
            break;
        const uint32_t range_index =
            backward_ranges.reverse_range_index(reverse_iteration);
        const auto& range = backward_ranges.ranges[range_index];
        const uint32_t count = static_cast<uint32_t>(__ldg(
            expert_counts + backward_ranges.expert_counts_begin(
                range_index, kNumExperts) + expert));
        const uint32_t range_blocks =
            math::ceil_div(count, kPoolBlockRows);
        if (expert_local_block >= range_blocks) {
            expert_local_block -= range_blocks;
            continue;
        }
        const uint32_t range_source_row = physical_range_prefix[
            range_index * (kNumExperts + 1u) + expert];
        DG_DEVICE_ASSERT(
            range_source_row >= range.pool_row_begin &&
            range_source_row % kPoolBlockRows == 0u &&
            range_source_row + range_blocks * kPoolBlockRows <=
                range.pool_row_begin + range.num_pool_rows);
        const uint32_t block_row =
            expert_local_block * kPoolBlockRows;
        const uint32_t remaining = count - block_row;
        return {
            expert, range_index, compact_block_ordinal,
            original_expert_local_block,
            range_source_row / kPoolBlockRows + expert_local_block,
            remaining < kPoolBlockRows ? remaining : kPoolBlockRows};
    }
    DG_DEVICE_ASSERT(false);
    return {
        static_cast<uint32_t>(-1), static_cast<uint32_t>(-1),
        compact_block_ordinal, static_cast<uint32_t>(-1),
        static_cast<uint32_t>(-1), 0u};
}

/** Initialize a persistent canonical-pool stage once per producer role. */
template <uint32_t kProducerThreads, uint32_t kNamedBarrier>
CUTLASS_DEVICE void k3_mxfp8_initialize_dw13b_gather_stage(
        K3MxFp8DW13BGatherStage* stage,
        uint32_t producer_thread) {
    DG_DEVICE_ASSERT(producer_thread < kProducerThreads);
    if (producer_thread == 0u) {
        stage->load_barrier.init(1u);
        cutlass::arch::fence_barrier_init();
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
}

/** Invalidate a persistent canonical-pool stage after its final TMA. */
template <uint32_t kProducerThreads, uint32_t kNamedBarrier>
CUTLASS_DEVICE void k3_mxfp8_release_dw13b_gather_stage(
        K3MxFp8DW13BGatherStage* stage,
        uint32_t producer_thread) {
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
    if (producer_thread == 0u) {
        using Barrier = cutlass::arch::ClusterTransactionBarrier;
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(
                &stage->load_barrier));
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
}

/** Publish one canonical pool-resident A or B group into its rolling ring.
 *
 * `source_map` names a contiguous BF16 physical pool (`grad_gate_up_output`
 * for A or `x_pool_output` for B).  The caller has resolved the compact
 * expert-major ordinal to one physical block and provides the packed per-block
 * GU/X completion word described above.
 * One descriptor-matched 32x128 TMA load replaces 32 symmetric remote row
 * gathers.  Invalid tail rows are explicitly zeroed in shared memory before
 * quantization, including an entire group when it lies beyond `valid_rows`.
 * The existing exact-two-term converter writes group-32 E4M3
 * primary values, BF16-rounded residual values, and one byte in the compact
 * UTCCP scale word.  Every value and scale-byte writer fences before the
 * leader release-publishes the ticket; a K128 consumer acquires all four
 * group tickets before loading their shared packed scale word.
 */
template <uint32_t kLogicalWidth, uint32_t kPoolBlockRows,
          uint32_t kReaderTarget, uint32_t kProducerThreads,
          uint32_t kNamedBarrier
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
          , typename DiagnosticRing
#endif
          >
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_dw13_ab_group_from_pool(
        const K3MxFp8EpiloguePanelRingView<
            3584u, kLogicalWidth, kPoolBlockRows>& ring,
        const cute::TmaDescriptor* source_map,
        const uint32_t* source_block_ready,
        const K3MxFp8DW13ABSourceBlock& source_block,
        uint32_t expert_value_begin,
        uint32_t expert_value_end,
        uint32_t expert_scale_begin,
        uint32_t logical_feature_begin,
        uint32_t group_in_block,
        uint32_t* primary_packed_scales,
        uint32_t* residual_packed_scales,
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
        const DiagnosticRing* diagnostic_ring,
#endif
        K3MxFp8DW13BGatherStage* stage,
        uint32_t& load_phase,
        uint32_t producer_thread,
        bool source_ready_acquired) {
    using Ring = K3MxFp8EpiloguePanelRingView<
        3584u, kLogicalWidth, kPoolBlockRows>;
    using Layout = typename Ring::Layout;
    constexpr uint32_t kRows = kK3MxFp8EpilogueRowsPerGroup;
    constexpr uint32_t kFeatures = kK3MxFp8EpilogueFeaturePanel;
    constexpr uint32_t kSourceBytes =
        kRows * kFeatures * sizeof(cutlass::bfloat16_t);
    static_assert(kPoolBlockRows == 192u);
    static_assert(kProducerThreads == kFeatures);
    static_assert(kLogicalWidth % kFeatures == 0u);
    static_assert(
        kReaderTarget == kK3MxFp8DW13AReaderTarget ||
        kReaderTarget == kK3MxFp8DW13BReaderTarget);

    DG_DEVICE_ASSERT(
        producer_thread < kProducerThreads && source_map != nullptr &&
        source_block_ready != nullptr &&
        source_block.compact_block_ordinal < ring.total_pool_blocks &&
        source_block.physical_pool_block < ring.total_pool_blocks &&
        source_block.valid_rows != 0u &&
        source_block.valid_rows <= kPoolBlockRows &&
        expert_value_begin % kPoolBlockRows == 0u &&
        expert_value_end > expert_value_begin &&
        expert_value_end % kPoolBlockRows == 0u &&
        group_in_block < Layout::kGroupsPerBlock &&
        logical_feature_begin % kFeatures == 0u &&
        logical_feature_begin + kFeatures <= kLogicalWidth);

    if (!source_ready_acquired) {
        if (producer_thread == 0u) {
            uint32_t observed = ptx::ld_acq(
                source_block_ready + source_block.physical_pool_block);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            const uint64_t wait_start = clock64();
#endif
            while (!k3_mxfp8_dw13_ab_block_ready(observed)) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                k3_mxfp8_exact_ring_watchdog<1u>(
                    wait_start, observed, kK3MxFp8DW13ABBlockReadyWord,
                    source_block.physical_pool_block,
                    blockIdx.x == 132u && (threadIdx.x & 31u) == 0u);
#endif
                __nanosleep(64);
                observed = ptx::ld_acq(
                    source_block_ready + source_block.physical_pool_block);
            }
        }
        cutlass::arch::NamedBarrier::sync(
            kProducerThreads, kNamedBarrier);
        // Carry this generic acquire into TMA's async proxy once per claim.
        asm volatile("fence.proxy.async.global;" ::: "memory");
    }

    const uint32_t production_ordinal =
        source_block.compact_block_ordinal;
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t feature_panel =
        logical_feature_begin / kFeatures;
    const uint32_t row_in_block = group_in_block * kRows;
    const uint32_t valid_rows =
        k3_mxfp8_dw13_ab_group_valid_rows(
            source_block.valid_rows, group_in_block);
    const uint32_t physical_row_begin =
        source_block.physical_pool_block * kPoolBlockRows +
        row_in_block;
    const uint32_t slot_row_begin =
        ring.slot_row_begin(slot) + row_in_block;
    const uint32_t expert_local_group =
        source_block.expert_local_block * Layout::kGroupsPerBlock +
        group_in_block;
    const uint32_t expert_num_groups =
        (expert_value_end - expert_value_begin) / kRows;
    const uint32_t packed_scale_row =
        expert_scale_begin +
        expert_local_group / kK3MxFp8EpilogueGroupsPerScaleWord;
    const uint32_t packed_scale_byte =
        expert_local_group % kK3MxFp8EpilogueGroupsPerScaleWord;
    const bool final_expert_group =
        expert_local_group + 1u == expert_num_groups;

    if (producer_thread == 0u) {
        detail::k3_mxfp8_epilogue_ring_ensure_slot_open(
            ring, production_ordinal,
            source_block.physical_pool_block,
            source_block.expert, source_block.valid_rows
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            ,
            diagnostic_ring);
#else
            );
#endif
        detail::k3_mxfp8_epilogue_ring_prepare_ticket(
            ring, production_ordinal,
            source_block.physical_pool_block,
            source_block.expert, feature_panel,
            group_in_block, kReaderTarget);
        if (valid_rows != 0u) {
            tma::copy<kFeatures, kRows, 0u, cutlass::bfloat16_t>(
                source_map, &stage->load_barrier, stage->source,
                logical_feature_begin, physical_row_begin);
            stage->load_barrier.arrive_and_expect_tx(kSourceBytes);
        }
    }
    if (valid_rows != 0u) {
        stage->load_barrier.wait(load_phase);
        load_phase ^= 1u;
    }

    // `grad_gate_up_output` does not promise that rows skipped by the SiTU
    // epilogue are globally zero until a later cleanup phase.  X currently
    // zero-stores its tail, but applying the same mask to both operands makes
    // the publisher independent of that incidental producer behavior.  One
    // feature thread owns one column, so these stores are race-free.  The
    // named barrier is the visibility edge into the quantizer below.
    #pragma unroll
    for (uint32_t row = valid_rows; row < kRows; ++row) {
        stage->source[row * kFeatures + producer_thread] =
            cutlass::bfloat16_t(0.0f);
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);

    k3_mxfp8_quantize_exact2_epilogue_panel<kLogicalWidth, false>(
        stage->source, kFeatures, 0u, valid_rows,
        logical_feature_begin, 0u,
        ring.primary_values, ring.residual_values,
        Layout::kRowBytes, slot_row_begin,
        primary_packed_scales, residual_packed_scales,
        packed_scale_row, packed_scale_byte,
        final_expert_group, producer_thread);

    // Values and the selected compact-scale byte are generic stores.  The CTA
    // barrier joins all 128 disjoint writers; the elected leader then performs
    // one cumulative device/proxy publication before release-storing the
    // ticket.  Consumers acquire the ticket before issuing their TMA reads.
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
    if (producer_thread == 0u) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        auto* const ticket = ring.ticket(
            slot, Layout::ticket_index(
                feature_panel, group_in_block));
        detail::k3_mxfp8_store_ticket_key_release_gpu(
            ticket, ring.epoch, sequence);
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
}

/** Publish all six group tickets for one feature-panel bundle. */
template <uint32_t kLogicalWidth, uint32_t kPoolBlockRows,
          uint32_t kReaderTarget, uint32_t kProducerThreads,
          uint32_t kNamedBarrier
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
          , typename DiagnosticRing
#endif
          >
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_dw13_ab_panel_bundle_from_pool(
        const K3MxFp8EpiloguePanelRingView<
            3584u, kLogicalWidth, kPoolBlockRows>& ring,
        const cute::TmaDescriptor* source_map,
        const uint32_t* source_block_ready,
        bool source_ready_acquired,
        const K3MxFp8DW13ABSourceBlock& source_block,
        uint32_t expert_value_begin,
        uint32_t expert_value_end,
        uint32_t expert_scale_begin,
        uint32_t logical_feature_begin,
        uint32_t* primary_packed_scales,
        uint32_t* residual_packed_scales,
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
        const DiagnosticRing* diagnostic_ring,
#endif
        K3MxFp8DW13BGatherStage* stage,
        uint32_t& load_phase,
        uint32_t producer_thread) {
    #pragma unroll 1
    for (uint32_t group_in_block = 0u;
         group_in_block <
             K3MxFp8DW13ABBlockMajorSchedule::kGroupsPerPanel;
         ++group_in_block) {
        k3_mxfp8_publish_dw13_ab_group_from_pool<
            kLogicalWidth, kPoolBlockRows, kReaderTarget,
            kProducerThreads, kNamedBarrier>(
                ring, source_map, source_block_ready, source_block,
                expert_value_begin, expert_value_end, expert_scale_begin,
                logical_feature_begin, group_in_block,
                primary_packed_scales, residual_packed_scales,
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                diagnostic_ring,
#endif
                stage, load_phase, producer_thread,
                source_ready_acquired || group_in_block != 0u);
    }
}

/** Pointer-stable arguments shared by every A/B producer engine. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kPoolBlockRows>
struct alignas(128) K3MxFp8DW13ABBackgroundContext {
    K3MxFp8EpiloguePanelRingView<
        kHidden, 2u * kIntermediateHidden, kPoolBlockRows> a_ring;
    K3MxFp8EpiloguePanelRingView<
        kHidden, kHidden, kPoolBlockRows> b_ring;
    const cute::TmaDescriptor* grad_gate_up_source_map;
    const cute::TmaDescriptor* x_pool_source_map;
    const uint32_t* source_block_ready;
    const int* expert_counts;
    const K3BackwardRangeSet* backward_ranges;
    const uint32_t* value_prefix;
    const uint32_t* scale_prefix;
    const uint32_t* physical_range_prefix;
    uint32_t* a_primary_packed_scales;
    uint32_t* a_residual_packed_scales;
    uint32_t* b_primary_packed_scales;
    uint32_t* b_residual_packed_scales;
    uint32_t* ab_block_cursor;
    K3MxFp8DW13BGatherStage* stages;
};
/** Allocation-free decoupled producer for both dW13 input rings.
 *
 * A caller-owned atomic cursor walks
 * `(compact block, A panels, B panels)` in that exact order. Each 128-thread
 * engine claims four feature panels, resolves reverse-range varlen source
 * coordinates once, acquires canonical-pool readiness once, and publishes
 * the panels' 24 group-32 value/scale tickets. Engines may overlap adjacent
 * claims, but no engine can claim a later block until all 19 claims of the
 * current block have been handed out. Slot opening supplies the rolling
 * credit: a wrapped generation cannot overwrite its predecessor until every
 * grouped consumer has retired it.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kPoolBlockRows,
          uint32_t kMaxRanges, uint32_t kProducerThreads,
          uint32_t kNamedBarrier>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_run_dw13_ab_block_major_producer(
        const K3MxFp8DW13ABBackgroundContext<
            kHidden, kIntermediateHidden, kPoolBlockRows>* context,
        K3MxFp8DW13BGatherStage* stage,
        uint32_t producer_thread) {
    using Schedule = K3MxFp8DW13ABBlockMajorSchedule;
    constexpr uint32_t kAWidth = 2u * kIntermediateHidden;
    static_assert(kHidden == 3584u && kIntermediateHidden == 3072u);
    static_assert(kPoolBlockRows == 192u);
    static_assert(kProducerThreads == 128u);
    static_assert(kMaxRanges == kK3MaxBackwardRanges);

    DG_DEVICE_ASSERT(
        context != nullptr &&
        context->a_ring.total_pool_blocks ==
            context->b_ring.total_pool_blocks &&
        context->a_ring.depth == context->b_ring.depth &&
        context->a_ring.total_pool_blocks != 0u &&
        context->value_prefix[kNumExperts] ==
            context->a_ring.total_pool_blocks * kPoolBlockRows &&
        context->value_prefix[kNumExperts] % kPoolBlockRows == 0u &&
        context->ab_block_cursor != nullptr &&
        context->source_block_ready != nullptr &&
        context->grad_gate_up_source_map != nullptr &&
        context->x_pool_source_map != nullptr &&
        producer_thread < kProducerThreads);

    if (producer_thread == 0u) {
        cute::prefetch_tma_descriptor(
            context->grad_gate_up_source_map);
        cute::prefetch_tma_descriptor(context->x_pool_source_map);
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);

    uint32_t load_phase = 0u;
    const uint32_t total_bundles =
        Schedule::num_bundles(context->a_ring.total_pool_blocks);
    while (true) {
        if (producer_thread == 0u) {
            stage->work[0] = atomicAdd(
                context->ab_block_cursor,
                Schedule::kPanelBundlesPerClaim);
            if (stage->work[0] < total_bundles) {
                const auto source_block =
                    k3_mxfp8_dw13_ab_source_block<
                        kNumExperts, kPoolBlockRows, kMaxRanges>(
                            Schedule::bundle_production_ordinal(
                                stage->work[0]),
                            context->expert_counts,
                            *context->backward_ranges,
                            context->value_prefix,
                            context->physical_range_prefix);
                stage->work[1] = source_block.expert;
                stage->work[2] = source_block.range_index;
                stage->work[3] = source_block.expert_local_block;
                stage->work[4] = source_block.physical_pool_block;
                stage->work[5] = source_block.valid_rows;
            }
        }
        cutlass::arch::NamedBarrier::sync(
            kProducerThreads, kNamedBarrier);
        const uint32_t base_bundle = stage->work[0];
        if (base_bundle >= total_bundles)
            break;

        const K3MxFp8DW13ABSourceBlock source_block{
            stage->work[1], stage->work[2],
            Schedule::bundle_production_ordinal(base_bundle),
            stage->work[3], stage->work[4], stage->work[5]};
        #pragma unroll 1
        for (uint32_t bundle_offset = 0u;
             bundle_offset < Schedule::kPanelBundlesPerClaim;
             ++bundle_offset) {
            const uint32_t bundle = base_bundle + bundle_offset;
            DG_DEVICE_ASSERT(
                bundle < total_bundles &&
                Schedule::bundle_production_ordinal(bundle) ==
                    source_block.compact_block_ordinal);
            const bool is_b = Schedule::bundle_is_b(bundle);
            const uint32_t feature_panel =
                Schedule::bundle_feature_panel(bundle);
            const uint32_t logical_feature_begin =
                feature_panel * kK3MxFp8EpilogueFeaturePanel;
            const uint32_t expert = source_block.expert;

            if (is_b) {
                DG_DEVICE_ASSERT(
                    feature_panel <
                        K3MxFp8ExactDW13BEpilogueRingLayout::
                            kFeaturePanels);
                k3_mxfp8_publish_dw13_ab_panel_bundle_from_pool<
                    kHidden, kPoolBlockRows,
                    kK3MxFp8DW13BReaderTarget,
                    kProducerThreads, kNamedBarrier>(
                        context->b_ring, context->x_pool_source_map,
                        context->source_block_ready,
                        bundle_offset != 0u, source_block,
                        context->value_prefix[expert],
                        context->value_prefix[expert + 1u],
                        context->scale_prefix[expert],
                        logical_feature_begin,
                        context->b_primary_packed_scales,
                        context->b_residual_packed_scales,
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                        &context->a_ring,
#endif
                        stage, load_phase, producer_thread);
            } else {
                DG_DEVICE_ASSERT(
                    feature_panel <
                        K3MxFp8ExactDW13AEpilogueRingLayout::
                            kFeaturePanels);
                k3_mxfp8_publish_dw13_ab_panel_bundle_from_pool<
                    kAWidth, kPoolBlockRows,
                    kK3MxFp8DW13AReaderTarget,
                    kProducerThreads, kNamedBarrier>(
                        context->a_ring,
                        context->grad_gate_up_source_map,
                        context->source_block_ready,
                        bundle_offset != 0u, source_block,
                        context->value_prefix[expert],
                        context->value_prefix[expert + 1u],
                        context->scale_prefix[expert],
                        logical_feature_begin,
                        context->a_primary_packed_scales,
                        context->a_residual_packed_scales,
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                        &context->b_ring,
#endif
                        stage, load_phase, producer_thread);
            }
        }
    }
    cutlass::arch::NamedBarrier::sync(
        kProducerThreads, kNamedBarrier);
}

/** Grouped dW13-B consumer over the planes-four/five rolling pair.
 *
 * The output task owns one 256-M cluster tile, so `reader_panel` is
 * `base_m / 256` (0..23), not either of the two 128-M accumulator panels.
 * One K128 B load may cross physical 192-row blocks; four independent group-32
 * tickets are acquired before their TMA reads and retired once after P01.
 */
template <uint32_t kNumExperts, uint32_t kMaxRanges,
          uint32_t kHidden, uint32_t kPoolBlockRows,
          uint32_t kConsumerBlockM = 256u>
struct K3MxFp8DW13BGroupedConsumerLifecycle {
    static constexpr bool kEnabled = true;
    static constexpr bool kAEnabled = false;
    static constexpr bool kBEnabled = true;
    using Ring = K3MxFp8EpiloguePanelRingView<
        kHidden, kHidden, kPoolBlockRows>;
    using Layout = typename Ring::Layout;

    Ring ring;
    const int* expert_counts;
    const K3BackwardRangeSet* backward_ranges;
    const uint32_t* value_prefix;
    const uint32_t* scale_prefix;
    const uint32_t* physical_range_prefix;

    CUTLASS_DEVICE K3MxFp8EpiloguePanelConsumerCoordinate coordinate(
            uint32_t expert, uint32_t feature_panel,
            uint32_t reader_panel, uint32_t compact_k_row) const {
        static_assert(kConsumerBlockM == 256u);
        DG_DEVICE_ASSERT(
            expert < kNumExperts &&
            reader_panel < kK3MxFp8DW13BReaderTarget &&
            compact_k_row >= value_prefix[expert] &&
            compact_k_row < value_prefix[expert + 1u]);
        const uint32_t expert_local_k =
            compact_k_row - value_prefix[expert];
        const uint32_t physical_pool_row =
            detail::k3_mxfp8_expert_source_pool_row<
                kNumExperts, kPoolBlockRows, kMaxRanges>(
                    expert, expert_local_k,
                    expert_counts, *backward_ranges,
                    physical_range_prefix);
        DG_DEVICE_ASSERT(
            physical_pool_row != static_cast<uint32_t>(-1) &&
            physical_pool_row % kK3MxFp8EpilogueRowsPerGroup == 0u);
        const uint32_t expert_local_group =
            expert_local_k / kK3MxFp8EpilogueRowsPerGroup;
        return Layout::consumer_coordinate(
            expert, feature_panel, reader_panel,
            compact_k_row / kPoolBlockRows,
            physical_pool_row,
            scale_prefix[expert] + expert_local_group /
                kK3MxFp8EpilogueGroupsPerScaleWord,
            expert_local_group % kK3MxFp8EpilogueGroupsPerScaleWord,
            ring.depth);
    }

    CUTLASS_DEVICE void acquire_coordinate(
            const K3MxFp8EpiloguePanelConsumerCoordinate& coord) const {
        auto* const ticket = k3_mxfp8_epilogue_ring_acquire_ticket(
            ring, coord.production_ordinal,
            coord.physical_pool_block,
            coord.feature_panel, coord.group_in_block);
        DG_DEVICE_ASSERT(
            ticket->expert == coord.expert &&
            ticket->reader_target == kK3MxFp8DW13BReaderTarget);
    }

    CUTLASS_DEVICE void retire_coordinate_after_p01(
            const K3MxFp8EpiloguePanelConsumerCoordinate& coord) const {
        auto* const ticket = ring.ticket(
            coord.slot,
            Layout::ticket_index(
                coord.feature_panel, coord.group_in_block));
        DG_DEVICE_ASSERT(
            coord.sequence != 0u &&
            ticket->production_ordinal == coord.production_ordinal &&
            ticket->expert == coord.expert &&
            coord.reader_panel < ticket->reader_target);
        k3_mxfp8_epilogue_ring_retire_known_sequence_after_p01(
            ring, ticket, coord.sequence);
    }

    CUTLASS_DEVICE void acquire(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        DG_DEVICE_ASSERT(
            base_m % kConsumerBlockM == 0u &&
            base_n % kK3MxFp8EpilogueFeaturePanel == 0u &&
            valid_k != 0u && valid_k <= 128u &&
            valid_k % kK3MxFp8EpilogueRowsPerGroup == 0u);
        const uint32_t feature_panel =
            base_n / kK3MxFp8EpilogueFeaturePanel;
        const uint32_t reader_panel = base_m / kConsumerBlockM;
        #pragma unroll
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                break;
            acquire_coordinate(coordinate(
                expert, feature_panel, reader_panel,
                compact_k_begin +
                    group * kK3MxFp8EpilogueRowsPerGroup));
        }
    }

    /** Gather one non-contiguous or partial B panel with group-32 TMAs. */
    CUTLASS_DEVICE __noinline__ void load_k32_fallback(
            uint32_t expert, uint32_t feature_begin,
            uint32_t reader_panel, uint32_t compact_k_begin,
            uint32_t valid_k,
            const cute::TmaDescriptor* value_map,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value) const {
        #pragma unroll 1
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                break;
            const auto coord = coordinate(
                expert,
                feature_begin / kK3MxFp8EpilogueFeaturePanel,
                reader_panel,
                compact_k_begin +
                    group * kK3MxFp8EpilogueRowsPerGroup);
            k3_mxfp8_load_mn_major_k32_fragment(
                value_map, full_barrier, smem_value,
                feature_begin, coord.slot_row_begin, group, 2u);
        }
    }

    /** Load B with two K128 TMAs when four published groups are adjacent. */
    CUTLASS_DEVICE void load_k128_stage(
            uint32_t expert, uint32_t base_m,
            uint32_t feature_begin, uint32_t compact_k_begin,
            uint32_t valid_k,
            const cute::TmaDescriptor* value_map,
            const cute::TmaDescriptor* k128_value_map,
            const cute::TmaDescriptor* scale_map,
            uint32_t compact_scale_row,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value, uint32_t* smem_scale,
            uint32_t& expected_bytes) const {
        constexpr uint32_t kGroupValueBytes =
            kK3MxFp8EpilogueFeaturePanel *
            kK3MxFp8EpilogueRowsPerGroup;
        const uint32_t reader_panel = base_m / kConsumerBlockM;
        DG_DEVICE_ASSERT(
            feature_begin % kK3MxFp8EpilogueFeaturePanel == 0u &&
            valid_k != 0u && valid_k <= 128u &&
            valid_k % kK3MxFp8EpilogueRowsPerGroup == 0u &&
            k128_value_map != nullptr);
        bool contiguous_k128 = valid_k == 128u;
        uint32_t first_slot_row_begin = 0u;
        #pragma unroll 1
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                break;
            const auto coord = coordinate(
                expert,
                feature_begin / kK3MxFp8EpilogueFeaturePanel,
                reader_panel,
                compact_k_begin +
                    group * kK3MxFp8EpilogueRowsPerGroup);
            acquire_coordinate(coord);
            if (group == 0u) {
                first_slot_row_begin = coord.slot_row_begin;
            } else {
                contiguous_k128 &= coord.slot_row_begin ==
                    first_slot_row_begin +
                        group * kK3MxFp8EpilogueRowsPerGroup;
            }
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        if (contiguous_k128) {
            k3_mxfp8_load_mn_major_k128_contiguous(
                k128_value_map, full_barrier, smem_value,
                feature_begin, first_slot_row_begin, 2u);
        } else {
            load_k32_fallback(
                expert, feature_begin, reader_panel,
                compact_k_begin, valid_k, value_map,
                full_barrier, smem_value);
        }
        expected_bytes +=
            (valid_k / kK3MxFp8EpilogueRowsPerGroup) * kGroupValueBytes;
        tma::copy<128u, 1u, 0u>(
            scale_map, full_barrier, smem_scale,
            feature_begin, compact_scale_row, 2u);
        expected_bytes += 128u * sizeof(uint32_t);
    }

    /** Operand-tagged adapter used by the dual A/B grouped-body hook. */
    CUTLASS_DEVICE __noinline__ void load_b_k128_stage(
            uint32_t expert, uint32_t base_m,
            uint32_t feature_begin, uint32_t compact_k_begin,
            uint32_t valid_k, bool residual,
            const cute::TmaDescriptor* value_map,
            const cute::TmaDescriptor* k128_value_map,
            const cute::TmaDescriptor* scale_map,
            uint32_t compact_scale_row,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value, uint32_t* smem_scale,
            uint32_t& expected_bytes) const {
        (void)residual;
        load_k128_stage(
            expert, base_m, feature_begin, compact_k_begin, valid_k,
            value_map, k128_value_map, scale_map, compact_scale_row,
            full_barrier, smem_value, smem_scale, expected_bytes);
    }

    CUTLASS_DEVICE void retire_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        const uint32_t feature_panel =
            base_n / kK3MxFp8EpilogueFeaturePanel;
        const uint32_t reader_panel = base_m / kConsumerBlockM;
        #pragma unroll 1
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                break;
            retire_coordinate_after_p01(coordinate(
                expert, feature_panel, reader_panel,
                compact_k_begin +
                    group * kK3MxFp8EpilogueRowsPerGroup));
        }
    }

    /** Operand-tagged adapter used by the dual A/B grouped-body hook. */
    CUTLASS_DEVICE __noinline__ void retire_b_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        // One adapter invocation represents one N128 B panel. The sole P01
        // leader invokes it for both adjacent cluster-rank panels, so even and
        // odd B panels each receive one retirement from every M256 reader.
        retire_k128_after_p01(
            expert, base_m, base_n, compact_k_begin, valid_k);
    }
};

/** One pointer-stable lifecycle object for both rolling dW13 operands.
 *
 * The grouped body invokes A and B hooks independently at their original TMA
 * sites and retires both ticket sets from the leader CTA after P01.  Keeping
 * the two concrete lifecycles side by side avoids virtual dispatch, process
 * globals, and any new launch argument.
 */
template <typename ALifecycle, typename BLifecycle>
struct K3MxFp8DW13ABGroupedConsumerLifecycle {
    static constexpr bool kAEnabled = ALifecycle::kAEnabled;
    static constexpr bool kBEnabled = BLifecycle::kBEnabled;
    static constexpr bool kEnabled = kAEnabled || kBEnabled;

    // Both operands share all compact/physical prefix inputs. Store the A
    // lifecycle once and only B's distinct ring view; this keeps the complete
    // object inside the parent's existing 120-byte control-tail gap.
    ALifecycle a;
    typename BLifecycle::Ring b_ring;

    CUTLASS_DEVICE BLifecycle b_lifecycle() const {
        return {
            b_ring, a.expert_counts, a.backward_ranges,
            a.value_prefix, a.scale_prefix,
            a.physical_range_prefix};
    }

    template <typename... Args>
    CUTLASS_DEVICE void load_a_k128_stage(Args&&... args) const {
        a.load_a_k128_stage(static_cast<Args&&>(args)...);
    }

    template <typename... Args>
    CUTLASS_DEVICE void load_b_k128_stage(Args&&... args) const {
        b_lifecycle().load_b_k128_stage(
            static_cast<Args&&>(args)...);
    }

    CUTLASS_DEVICE void retire_a_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        a.retire_a_k128_after_p01(
            expert, base_m, base_n, compact_k_begin, valid_k);
    }

    CUTLASS_DEVICE void retire_b_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        b_lifecycle().retire_b_k128_after_p01(
            expert, base_m, base_n, compact_k_begin, valid_k);
    }
};

constexpr uint32_t kK3MxFp8DW13ABProducerFirstWarp = 8u;
constexpr uint32_t kK3MxFp8DW13ABProducerWarpsPerEngine = 4u;
constexpr uint32_t kK3MxFp8DW13ABProducerNumEngines = 6u;
constexpr uint32_t kK3MxFp8DW13ABProducerThreads =
    kK3MxFp8DW13ABProducerWarpsPerEngine * 32u;
constexpr uint32_t kK3MxFp8DW13ABProducerFirstNamedBarrier = 1u;

static_assert(kK3MxFp8DW13ABProducerThreads == 128u);
static_assert(
    kK3MxFp8DW13ABProducerFirstWarp +
        kK3MxFp8DW13ABProducerNumEngines *
            kK3MxFp8DW13ABProducerWarpsPerEngine <=
    32u);
static_assert(
    kK3MxFp8DW13ABProducerFirstNamedBarrier +
        kK3MxFp8DW13ABProducerNumEngines <=
    cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers -
        cutlass::arch::NamedBarrier::ReservedNamedBarrierCount);


/** Run one of six allocation-free A/B publishers on idle body warps. */
template <uint32_t kEngine, uint32_t kHidden,
          uint32_t kIntermediateHidden, uint32_t kNumExperts,
          uint32_t kPoolBlockRows, uint32_t kMaxRanges>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_run_dw13_ab_background_engine(
        uint32_t background_warp_idx,
        uint32_t background_lane_idx,
        const K3MxFp8DW13ABBackgroundContext<
            kHidden, kIntermediateHidden, kPoolBlockRows>* context) {
    static_assert(kEngine < kK3MxFp8DW13ABProducerNumEngines);
    constexpr uint32_t kFirstWarp =
        kK3MxFp8DW13ABProducerFirstWarp +
        kEngine * kK3MxFp8DW13ABProducerWarpsPerEngine;
    constexpr uint32_t kNamedBarrier =
        kK3MxFp8DW13ABProducerFirstNamedBarrier + kEngine;
    if (background_warp_idx < kFirstWarp ||
        background_warp_idx >=
            kFirstWarp + kK3MxFp8DW13ABProducerWarpsPerEngine) {
        return;
    }
    const uint32_t producer_thread =
        (background_warp_idx - kFirstWarp) * 32u +
        background_lane_idx;
    auto* const stage = context->stages + kEngine;
    k3_mxfp8_initialize_dw13b_gather_stage<
        kK3MxFp8DW13ABProducerThreads, kNamedBarrier>(
            stage, producer_thread);
    k3_mxfp8_run_dw13_ab_block_major_producer<
        kHidden, kIntermediateHidden, kNumExperts,
        kPoolBlockRows, kMaxRanges,
        kK3MxFp8DW13ABProducerThreads, kNamedBarrier>(
            context, stage, producer_thread);
    k3_mxfp8_release_dw13b_gather_stage<
        kK3MxFp8DW13ABProducerThreads, kNamedBarrier>(
            stage, producer_thread);
}

/** Tiny callback object passed through the grouped body. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kPoolBlockRows,
          uint32_t kMaxRanges>
struct K3MxFp8DW13ABBackgroundProducer {
    const K3MxFp8DW13ABBackgroundContext<
        kHidden, kIntermediateHidden, kPoolBlockRows>* context;

    CUTLASS_DEVICE __noinline__ void operator()(
            uint32_t background_warp_idx,
            uint32_t background_lane_idx) const {
        k3_mxfp8_run_dw13_ab_background_engine<
            0u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
        k3_mxfp8_run_dw13_ab_background_engine<
            1u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
        k3_mxfp8_run_dw13_ab_background_engine<
            2u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
        k3_mxfp8_run_dw13_ab_background_engine<
            3u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
        k3_mxfp8_run_dw13_ab_background_engine<
            4u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
        k3_mxfp8_run_dw13_ab_background_engine<
            5u, kHidden, kIntermediateHidden, kNumExperts,
            kPoolBlockRows, kMaxRanges>(
                background_warp_idx, background_lane_idx, context);
    }
};

/** Early Option-A dW13 consumer/producer phase for the fused parent.
 *
 * This helper is deliberately ordinary rather than ready-first: each
 * cluster may join late, immediately claims grouped M256xN256 work through
 * the existing dynamic stream, and blocks only on the exact A/B ring tickets
 * needed by its K128 load. A disjoint block-major cursor feeds six
 * 128-thread background producer engines. The callback performs no legacy
 * feature-major production, fixed-top-k reduction, or NVLink rendezvous.
 *
 * The caller must have reset `kDW13Cursor`, `kDW13ABBlockCursor`, and every
 * dW13 mailbox before release-publishing `expected_epoch`.  It must also own
 * an empty base-zero TMEM allocation: this phase initializes and invalidates
 * its mbarriers but neither allocates nor frees parent TMEM.  On return, this
 * CTA has observed terminal retirement of every local A and B ring slot; the
 * parent still owns the subsequent cross-rank direct-dX rendezvous.
 * `w13_output` is also the former W13 BF16 dequantization arena.  The selected
 * parent uses inline W13 dequantization, so this helper may preserve the public
 * clear-empty-output contract while active experts are written by grouped TMA.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kPoolBlockRows,
          uint32_t kNumSMs, uint32_t kNumThreads,
          bool kAccumulateWgrad, uint32_t kBatchTasks>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_run_early_dw13_ab_ring(
        uint32_t* state,
        uint32_t expected_epoch,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        const cute::TmaDescriptor& tensor_map_d,
        uint8_t* smem_buffer,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        uint32_t scale_capacity_rows,
        const cutlass::float_e4m3_t* scale_arena_source,
        cutlass::bfloat16_t* w13_output,
        bool clear_empty_outputs,
        cutlass::bfloat16_t* combine_plane_zero,
        uint32_t combine_capacity_rows) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    using Config = Sm100K3MxFp8ThreeTermWgradConfig<
        2u * kIntermediateHidden, kHidden,
        kNumExperts, kNumSMs, kAccumulateWgrad>;
    using Provider = sched::ExternalKGroupedDynamicRangeProvider<
        Config::kBlockM, Config::kBlockN,
        Config::kNumMulticast, Config::kIsMulticastOnA,
        kNumSMs, 2u * kIntermediateHidden, kHidden,
        kPoolBlockRows, Config::kScaleKSpan,
        Overlap::kPoolBlockPrefix, Overlap::kActiveExperts,
        4u, 4u, 16u,
        false, 0u,
        Prefix::kValuePrefix, Prefix::kScalePrefix, true,
        0u, Overlap::kPoolBlockPrefix,
        Config::kKAlignment, Config::kBlockK, false>;
    using RetainParentTmem = Sm100K3MxFp8WgradBatchResourceHooks<
        true, true, true, false, false>;
    using ALifecycle = K3MxFp8EpilogueGroupedConsumerLifecycle<
        kNumExperts, kK3MaxBackwardRanges,
        kHidden, 2u * kIntermediateHidden, kPoolBlockRows>;
    using BLifecycle = K3MxFp8DW13BGroupedConsumerLifecycle<
        kNumExperts, kK3MaxBackwardRanges,
        kHidden, kPoolBlockRows, Config::kBlockM>;
    using Lifecycle = K3MxFp8DW13ABGroupedConsumerLifecycle<
        ALifecycle, BLifecycle>;
    using ProducerContext = K3MxFp8DW13ABBackgroundContext<
        kHidden, kIntermediateHidden, kPoolBlockRows>;
    using BackgroundProducer = K3MxFp8DW13ABBackgroundProducer<
        kHidden, kIntermediateHidden, kNumExperts,
        kPoolBlockRows, kK3MaxBackwardRanges>;

    constexpr uint32_t kLifecycleOffset = math::constexpr_align(
        kK3MxFp8DW13QuantBodySmemBytes, 8u);
    constexpr uint32_t kGroupedTmemPtrOffset =
        sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset<Config>();
    constexpr uint32_t kProducerContextOffset = math::constexpr_align(
        kLifecycleOffset + static_cast<uint32_t>(sizeof(Lifecycle)),
        128u);
    constexpr uint32_t kProducerStageBegin = math::constexpr_align(
        kProducerContextOffset +
            static_cast<uint32_t>(sizeof(ProducerContext)),
        128u);
    constexpr uint32_t kProducerStageEnd =
        kProducerStageBegin +
        kK3MxFp8DW13ABProducerNumEngines *
            sizeof(K3MxFp8DW13BGatherStage);
    static_assert(
        kHidden == 3584u && kIntermediateHidden == 3072u &&
        kNumExperts == 112u && kPoolBlockRows == 192u &&
        kNumSMs == 148u && kNumThreads == 1024u &&
        kBatchTasks != 0u && kBatchTasks % 4u == 0u);
    static_assert(
        !Provider::kTaskReadyFirstTaskClaim &&
        Provider::kTaskRetirementBias == 0u &&
        Provider::kCompleteAcquireMask ==
            Overlap::kExactSchedulerRoleMask &&
        Provider::kNumClusterTasksPerGroup ==
            Overlap::kDW13ClusterTasksPerExpert);
    static_assert(
        RetainParentTmem::kInitializeBatchResources &&
        RetainParentTmem::kReleaseBatchResources &&
        !RetainParentTmem::kAllocateTmem &&
        !RetainParentTmem::kFreeTmem);
    static_assert(sizeof(Lifecycle) <= 120u);
    static_assert(sizeof(ProducerContext) <= 256u);
    static_assert(
        kGroupedTmemPtrOffset + sizeof(uint32_t) <= kLifecycleOffset);
    static_assert(kLifecycleOffset == 153736u);
    static_assert(kProducerContextOffset == 153856u);
    static_assert(kProducerStageBegin == 154112u);
    static_assert(kProducerStageEnd == 204032u);
    static_assert(
        kProducerStageEnd <= kK3MxFp8WgradStreamingSmemBytes);

    if (threadIdx.x == 0u) {
        while (ptx::ld_acq(state + Overlap::kDW13Epoch) !=
               expected_epoch) {
            __nanosleep(64);
        }
        // The embedded grouped body retains the parent's live TMEM allocation
        // (`kAllocateTmem == false`) but still reads its private shared-memory
        // pointer word.  It is distinct from the parent body's pointer slot,
        // so initialize it deterministically before the grouped-body cluster
        // synchronization can make it visible to either CTA.
        *reinterpret_cast<uint32_t*>(
            smem_buffer + kGroupedTmemPtrOffset) = 0u;
    }
    __syncthreads();

    // The dynamic predecessor cleared empty dW13 experts before entering its
    // grouped body.  Preserve that observable contract here: the active-only
    // scheduler never writes those slices, and the full persistent grid gives
    // every vector one unique owner even though tail CTAs reach this phase
    // before the W2 prefix.  Active and empty experts are disjoint, so this
    // needs neither a whole-grid join nor a delay in ring production.
    if (clear_empty_outputs) {
        constexpr uint64_t kBF16PerVector =
            sizeof(uint4) / sizeof(cutlass::bfloat16_t);
        constexpr uint64_t kW13VectorsPerExpert =
            static_cast<uint64_t>(2u * kIntermediateHidden) * kHidden /
            kBF16PerVector;
        const uint64_t global_thread =
            static_cast<uint64_t>(blockIdx.x) * kNumThreads + threadIdx.x;
        constexpr uint64_t kGlobalThreads =
            static_cast<uint64_t>(kNumSMs) * kNumThreads;
        auto* const w13_vectors = reinterpret_cast<uint4*>(w13_output);
        const uint4 zero = {0u, 0u, 0u, 0u};
        #pragma unroll 1
        for (uint32_t expert = 0u; expert < kNumExperts; ++expert) {
            bool active_in_union = false;
            #pragma unroll 1
            for (uint32_t range_iteration = 0u;
                 range_iteration < backward_ranges.num_ranges;
                 ++range_iteration) {
                const uint32_t range_index =
                    backward_ranges.reverse_range_index(range_iteration);
                active_in_union |= __ldg(
                    expert_counts + backward_ranges.expert_counts_begin(
                        range_index, kNumExperts) + expert) != 0;
            }
            if (active_in_union)
                continue;
            for (uint64_t linear = global_thread;
                 linear < kW13VectorsPerExpert;
                 linear += kGlobalThreads) {
                w13_vectors[
                    static_cast<uint64_t>(expert) *
                        kW13VectorsPerExpert + linear] = zero;
            }
        }
    }

    const uint32_t total_pool_blocks =
        backward_ranges.total_pool_rows / kPoolBlockRows;
    DG_DEVICE_ASSERT(
        backward_ranges.total_pool_rows % kPoolBlockRows == 0u &&
        state[Overlap::kTotalPoolBlocks] == total_pool_blocks &&
        state[Prefix::kValuePrefix + kNumExperts] ==
            total_pool_blocks * kPoolBlockRows &&
        combine_capacity_rows > kK3MxFp8RingReservedRows);

    const uint32_t ring_epoch = expected_epoch ^ 0x80000000u;
    const auto a_ring = k3_mxfp8_make_epilogue_panel_ring<
        kHidden, 2u * kIntermediateHidden, kPoolBlockRows>(
            reinterpret_cast<uint8_t*>(combine_plane_zero),
            combine_capacity_rows, total_pool_blocks, ring_epoch);
    const auto b_ring = k3_mxfp8_make_dw13b_epilogue_panel_ring<
        kHidden, kPoolBlockRows>(
            reinterpret_cast<uint8_t*>(combine_plane_zero),
            combine_capacity_rows, total_pool_blocks, ring_epoch);
    DG_DEVICE_ASSERT(
        a_ring.depth == b_ring.depth && a_ring.depth != 0u);

    auto* const lifecycle = reinterpret_cast<Lifecycle*>(
        smem_buffer + kLifecycleOffset);
    if (threadIdx.x == 0u) {
        *lifecycle = {
            {a_ring, expert_counts, &backward_ranges,
             state + Prefix::kValuePrefix,
             state + Prefix::kScalePrefix,
             state + Prefix::kPhysicalRangePrefix},
            b_ring};
    }
    __syncthreads();

    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / Config::kNumMulticast;
    auto* const mailbox = state + Overlap::kDW13Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    const sched::ExternalKGroupedRangeStream stream{
        state,
        0u, 0u,
        state + Overlap::kDW13Cursor,
        0u,
        mailbox,
        kBatchTasks,
        Overlap::kDW13ClusterTasksPerExpert,
        nullptr,
        0u,
        state + Overlap::kDW13Epoch,
        expected_epoch,
        state + Overlap::kDW13Tasks,
    };

    const auto* const maps = tensor_map_pack->maps;
    auto* const ab_block_cursor =
        state + Overlap::kDW13ABBlockCursor;
    DG_DEVICE_ASSERT(ab_block_cursor != stream.task_cursor);
    auto* const producer_context = reinterpret_cast<ProducerContext*>(
        smem_buffer + kProducerContextOffset);
    if (threadIdx.x == 0u) {
        *producer_context = {
            a_ring,
            b_ring,
            maps + kK3MxFp8DW13ProducerSourceAMap,
            maps + kK3MxFp8DW13ProducerSourceBMap,
            state + Overlap::kNumWords,
            expert_counts,
            &backward_ranges,
            state + Prefix::kValuePrefix,
            state + Prefix::kScalePrefix,
            state + Prefix::kPhysicalRangePrefix,
            k3_mxfp8_exact_epilogue_dw13a_packed_scale_alias<
                kHidden, kIntermediateHidden, kPoolBlockRows, false>(
                    scale_arena_source, scale_capacity_rows),
            k3_mxfp8_exact_epilogue_dw13a_packed_scale_alias<
                kHidden, kIntermediateHidden, kPoolBlockRows, true>(
                    scale_arena_source, scale_capacity_rows),
            k3_mxfp8_exact_epilogue_dw13b_packed_scale_alias<
                kHidden, kIntermediateHidden, kPoolBlockRows, false>(
                    scale_arena_source, scale_capacity_rows),
            k3_mxfp8_exact_epilogue_dw13b_packed_scale_alias<
                kHidden, kIntermediateHidden, kPoolBlockRows, true>(
                    scale_arena_source, scale_capacity_rows),
            ab_block_cursor,
            reinterpret_cast<K3MxFp8DW13BGatherStage*>(
                smem_buffer + kProducerStageBegin),
        };
    }
    __syncthreads();
    const BackgroundProducer background_producer{producer_context};

    Sm100K3MxFp8NoInputTileRetired no_input_tile_retired;
    sm100_k3_mxfp8_three_term_grouped_wgrad_body<
        Config, Provider, RetainParentTmem,
        decltype(background_producer),
        Sm100K3MxFp8NoInputTileRetired, Lifecycle>(
            reinterpret_cast<int*>(
                const_cast<sched::ExternalKGroupedRangeStream*>(&stream)),
            state[Prefix::kValuePrefix + kNumExperts],
            maps[kK3MxFp8DW13RingValueAPrimaryMap],
            maps[kK3MxFp8DW13RingValueAResidualMap],
            maps[kK3MxFp8DW13RingValueBPrimaryMap],
            maps[kK3MxFp8DW13RingValueBResidualMap],
            maps[kK3MxFp8DW13ScaleAPrimaryMap],
            maps[kK3MxFp8DW13ScaleAResidualMap],
            maps[kK3MxFp8DW13ScaleBPrimaryMap],
            maps[kK3MxFp8DW13ScaleBResidualMap],
            tensor_map_d, smem_buffer, false,
            background_producer, no_input_tile_retired,
            nullptr, nullptr, lifecycle);

    // Tail CTAs and later prefix CTAs all enter this identical helper. Spread
    // the terminal checks across that eventual full-grid cohort; the parent's
    // following converged edge orders reuse after every assigned check.
    const uint32_t global_thread =
        static_cast<uint32_t>(blockIdx.x) * kNumThreads + threadIdx.x;
    constexpr uint32_t kGlobalThreads = kNumSMs * kNumThreads;
    for (uint32_t slot = global_thread;
         slot < a_ring.depth; slot += kGlobalThreads) {
        k3_mxfp8_epilogue_ring_wait_terminal_retirement(a_ring, slot);
    }
    for (uint32_t slot = global_thread;
         slot < b_ring.depth; slot += kGlobalThreads) {
        k3_mxfp8_epilogue_ring_wait_terminal_retirement(b_ring, slot);
    }
    __syncthreads();
}

/** Wait for all locally produced A/B slots before fixed-top-k reuse.
 *
 * This is only the local drain edge.  The caller MUST next execute the existing
 * cross-rank `prepare_direct_grad_x_planes` NVLink rendezvous while the full
 * grid is converged.  Only after both edges may W13 dgrad clear or remotely
 * scatter into any fixed-top-k plane.  Local ticket retirement alone cannot
 * prove that another rank has stopped reading planes two through five.
 */
template <typename ARing, typename BRing>
CUTLASS_DEVICE void k3_mxfp8_wait_dw13_ab_rings_before_direct_dx(
        const ARing& a_ring, const BRing& b_ring,
        uint32_t global_thread, uint32_t global_threads) {
    for (uint32_t slot = global_thread;
         slot < a_ring.depth; slot += global_threads) {
        k3_mxfp8_epilogue_ring_wait_terminal_retirement(a_ring, slot);
    }
    for (uint32_t slot = global_thread;
         slot < b_ring.depth; slot += global_threads) {
        k3_mxfp8_epilogue_ring_wait_terminal_retirement(b_ring, slot);
    }
}

/** Pure contract used by host/source tests for the mandatory reuse order. */
struct K3MxFp8RingDirectDXLifetimeContract {
    static constexpr bool may_reuse_fixed_topk_planes(
            bool a_locally_drained, bool b_locally_drained,
            bool rank_rendezvous_complete) {
        return a_locally_drained && b_locally_drained &&
            rank_rendezvous_complete;
    }
};

static_assert(!K3MxFp8RingDirectDXLifetimeContract::
    may_reuse_fixed_topk_planes(true, true, false));
static_assert(K3MxFp8RingDirectDXLifetimeContract::
    may_reuse_fixed_topk_planes(true, true, true));

}  // namespace deep_gemm
