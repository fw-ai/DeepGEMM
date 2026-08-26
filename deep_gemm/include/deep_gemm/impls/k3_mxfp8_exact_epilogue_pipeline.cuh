#pragma once

#include <deep_gemm/impls/k3_mxfp8_exact_epilogue_ring.hpp>
// This prototype deliberately builds on the exact producer's numerical
// primitives and UTCCP-native scale layout.  It is kept in a separate header
// so the production parent can adopt the panel ABI without changing the
// validated terminal grouped-UMMA body in the first slice.
#include <deep_gemm/impls/sm100_mxfp8_three_term_grouped_wgrad.cuh>

#ifndef DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
#define DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG 0
#endif

namespace deep_gemm {

#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
/** Trace-build-only deadline for localizing a stalled exact-ring edge.
 *
 * The production specialization compiles this helper out completely.  A trace
 * specialization gives the selected diagnostic lane roughly one second to
 * make progress, prints the first-class wait site and observed coordinates,
 * then traps so a malformed ring cannot occupy a shared GPU indefinitely.
 */
constexpr uint64_t kK3MxFp8ExactRingWatchdogCycles = 2000000000ull;

template <uint32_t kSite>
CUTLASS_DEVICE void k3_mxfp8_exact_ring_watchdog(
        uint64_t wait_start, uint32_t observed,
        uint32_t expected, uint32_t coordinate,
        bool diagnostic_lane) {
    if (clock64() - wait_start <= kK3MxFp8ExactRingWatchdogCycles ||
        !diagnostic_lane)
        return;
    printf(
        "K3_EXACT_RING_WATCHDOG site=%u block=%u thread=%u "
        "observed=0x%08x expected=0x%08x coordinate=%u\n",
        kSite, static_cast<uint32_t>(blockIdx.x),
        static_cast<uint32_t>(threadIdx.x), observed, expected,
        coordinate);
    asm volatile("trap;");
}
#endif

enum class K3MxFp8EpilogueOperand : uint32_t {
    DW2A = 0u,
    DW2B = 1u,
    DW13A = 2u,
    DW13B = 3u,
};
constexpr uint32_t kK3MxFp8EpilogueValueBytesPerGroup =
    kK3MxFp8EpilogueRowsPerGroup * kK3MxFp8EpilogueFeaturePanel;
constexpr uint32_t kK3MxFp8EpilogueScaleBytesPerPackedRow =
    kK3MxFp8EpilogueFeaturePanel * sizeof(uint32_t);

constexpr uint32_t k3_mxfp8_epilogue_operand_group(
        K3MxFp8EpilogueOperand operand, uint32_t group_in_scale_row) {
    return (static_cast<uint32_t>(operand) << 8) |
        (group_in_scale_row & 0xffu);
}

constexpr K3MxFp8EpilogueOperand k3_mxfp8_epilogue_operand(
        uint32_t operand_and_group) {
    return static_cast<K3MxFp8EpilogueOperand>(operand_and_group >> 8);
}

constexpr uint32_t k3_mxfp8_epilogue_group(uint32_t operand_and_group) {
    return operand_and_group & 0xffu;
}

struct K3MxFp8EpilogueCompactScaleBlock {
    uint32_t expert_scale_prefix;
    uint32_t expert_local_group_begin;
    uint32_t expert_num_groups;
    uint32_t production_ordinal;
};

/** Resolve a physical range-local block into the compact scale ABI.
 *
 * SiTU runs before the terminal exact metadata builder, but the immutable
 * expert counts and reverse-range order already define that metadata.  One
 * elected epilogue lane computes this coordinate once per physical block;
 * all 128 scale owners then publish directly into the ordinary compact,
 * UTCCP-native dW13-A scale arena.  Consequently the grouped consumer can
 * retain its proven 256-word scale TMA and needs no scalar gather.
 */
template <uint32_t kNumExperts, uint32_t kPoolBlockRows,
          uint32_t kMaxRanges>
CUTLASS_DEVICE __noinline__ K3MxFp8EpilogueCompactScaleBlock
k3_mxfp8_epilogue_compact_scale_block(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        uint32_t expert, uint32_t range_index,
        uint32_t expert_block_in_range) {
    constexpr uint32_t kGroupsPerBlock =
        kPoolBlockRows / kK3MxFp8EpilogueRowsPerGroup;
    static_assert(
        kPoolBlockRows % kK3MxFp8EpilogueRowsPerGroup == 0u);
    DG_DEVICE_ASSERT(
        expert < kNumExperts &&
        range_index < backward_ranges.num_ranges);

    uint32_t expert_scale_prefix = 0u;
    uint32_t prior_expert_blocks = 0u;
    #pragma unroll 1
    for (uint32_t prior_expert = 0u;
         prior_expert < expert; ++prior_expert) {
        uint32_t prior_groups = 0u;
        #pragma unroll
        for (uint32_t reverse_iteration = 0u;
             reverse_iteration < kMaxRanges; ++reverse_iteration) {
            if (reverse_iteration >= backward_ranges.num_ranges)
                break;
            const uint32_t prior_range =
                backward_ranges.reverse_range_index(reverse_iteration);
            const uint32_t count = static_cast<uint32_t>(__ldg(
                expert_counts + backward_ranges.expert_counts_begin(
                    prior_range, kNumExperts) + prior_expert));
            prior_groups +=
                math::ceil_div(count, kPoolBlockRows) * kGroupsPerBlock;
        }
        expert_scale_prefix += math::ceil_div(
            prior_groups, kK3MxFp8EpilogueGroupsPerScaleWord);
        prior_expert_blocks += prior_groups / kGroupsPerBlock;
    }

    uint32_t expert_local_group_begin = 0u;
    uint32_t expert_num_groups = 0u;
    bool found_range = false;
    #pragma unroll
    for (uint32_t reverse_iteration = 0u;
         reverse_iteration < kMaxRanges; ++reverse_iteration) {
        if (reverse_iteration >= backward_ranges.num_ranges)
            break;
        const uint32_t current_range =
            backward_ranges.reverse_range_index(reverse_iteration);
        const uint32_t count = static_cast<uint32_t>(__ldg(
            expert_counts + backward_ranges.expert_counts_begin(
                current_range, kNumExperts) + expert));
        const uint32_t range_groups =
            math::ceil_div(count, kPoolBlockRows) * kGroupsPerBlock;
        expert_num_groups += range_groups;
        if (!found_range) {
            if (current_range == range_index)
                found_range = true;
            else
                expert_local_group_begin += range_groups;
        }
    }
    DG_DEVICE_ASSERT(found_range);
    expert_local_group_begin +=
        expert_block_in_range * kGroupsPerBlock;
    return {
        expert_scale_prefix,
        expert_local_group_begin,
        expert_num_groups,
        prior_expert_blocks +
            expert_local_group_begin / kGroupsPerBlock};
}

/** Runtime view over two already-allocated symmetric fixed-top-k planes. */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
struct K3MxFp8EpiloguePanelRingView {
    using Layout = K3MxFp8EpiloguePanelRingLayout<
        kHidden, kLogicalWidth, kPoolBlockRows>;

    uint8_t* primary_values;
    uint8_t* residual_values;
    uint32_t capacity_rows;
    uint32_t depth;
    uint32_t total_pool_blocks;
    uint32_t epoch;

    CUTLASS_HOST_DEVICE uint32_t slot_row_begin(uint32_t slot) const {
        return slot * kPoolBlockRows;
    }

    CUTLASS_DEVICE K3MxFp8EpiloguePanelTicket* ticket(
            uint32_t slot, uint32_t ticket_index) const {
        const uint32_t row = slot_row_begin(slot) +
            Layout::ticket_tail_row(ticket_index);
        return reinterpret_cast<K3MxFp8EpiloguePanelTicket*>(
            primary_values + static_cast<uint64_t>(row) *
                Layout::kRowBytes + Layout::kValueBytesPerRow +
            Layout::ticket_tail_byte(ticket_index));
    }

    CUTLASS_DEVICE K3MxFp8EpiloguePanelSlotState* slot_state(
            uint32_t slot) const {
        const uint32_t row = slot_row_begin(slot) +
            Layout::kSlotStateTailRow;
        return reinterpret_cast<K3MxFp8EpiloguePanelSlotState*>(
            primary_values + static_cast<uint64_t>(row) *
                Layout::kRowBytes + Layout::kValueBytesPerRow);
    }
};

/** Producer-only aliases stored in an already-reserved shared gap.
 *
 * The consumer reconstructs its own ring view from existing suffix arguments;
 * this producer copy is dead before the later exact-quantizer scratch aliases
 * the same shared-memory region.
 */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
struct alignas(16) K3MxFp8EpilogueParentSharedContext {
    K3MxFp8EpiloguePanelRingView<
        kHidden, kLogicalWidth, kPoolBlockRows> producer_ring;
    uint32_t* primary_packed_scales;
    uint32_t* residual_packed_scales;
    K3MxFp8EpilogueCompactScaleBlock producer_scale_block;
    uint32_t physical_pool_block;
    uint32_t expert;
    uint32_t range_index;
    uint32_t expert_block_in_range;
    uint32_t block_valid_rows;
    uint32_t feature_block;
};

template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows, uint32_t kPrimaryPlane,
          uint32_t kResidualPlane>
CUTLASS_HOST_DEVICE K3MxFp8EpiloguePanelRingView<
    kHidden, kLogicalWidth, kPoolBlockRows>
k3_mxfp8_make_epilogue_panel_ring_at_planes(
        uint8_t* combine_plane_zero, uint32_t capacity_rows,
        uint32_t total_pool_blocks, uint32_t epoch) {
    using Layout = K3MxFp8EpiloguePanelRingLayout<
        kHidden, kLogicalWidth, kPoolBlockRows>;
    const uint64_t plane_bytes =
        static_cast<uint64_t>(capacity_rows) * Layout::kRowBytes;
    const uint64_t row_base_bytes =
        static_cast<uint64_t>(kK3MxFp8RingReservedRows) *
        Layout::kRowBytes;
    return {
        combine_plane_zero +
            kPrimaryPlane * plane_bytes + row_base_bytes,
        combine_plane_zero +
            kResidualPlane * plane_bytes + row_base_bytes,
        capacity_rows,
        Layout::ring_depth(capacity_rows),
        total_pool_blocks,
        epoch,
    };
}

/** Bind the dW13-A pair in fixed-top-k planes two and three. */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
CUTLASS_HOST_DEVICE K3MxFp8EpiloguePanelRingView<
    kHidden, kLogicalWidth, kPoolBlockRows>
k3_mxfp8_make_epilogue_panel_ring(
        uint8_t* combine_plane_zero, uint32_t capacity_rows,
        uint32_t total_pool_blocks, uint32_t epoch) {
    return k3_mxfp8_make_epilogue_panel_ring_at_planes<
        kHidden, kLogicalWidth, kPoolBlockRows,
        kK3MxFp8EpilogueScratchPrimaryPlane,
        kK3MxFp8EpilogueScratchResidualPlane>(
            combine_plane_zero, capacity_rows, total_pool_blocks, epoch);
}

/** Bind the dW13-B pair in fixed-top-k planes four and five. */
template <uint32_t kHidden, uint32_t kPoolBlockRows>
CUTLASS_HOST_DEVICE K3MxFp8EpiloguePanelRingView<
    kHidden, kHidden, kPoolBlockRows>
k3_mxfp8_make_dw13b_epilogue_panel_ring(
        uint8_t* combine_plane_zero, uint32_t capacity_rows,
        uint32_t total_pool_blocks, uint32_t epoch) {
    return k3_mxfp8_make_epilogue_panel_ring_at_planes<
        kHidden, kHidden, kPoolBlockRows,
        kK3MxFp8DW13BScratchPrimaryPlane,
        kK3MxFp8DW13BScratchResidualPlane>(
            combine_plane_zero, capacity_rows, total_pool_blocks, epoch);
}

namespace detail {

CUTLASS_DEVICE void k3_mxfp8_store_release_gpu(
        uint32_t* address, uint32_t value) {
    asm volatile(
        "st.release.gpu.global.u32 [%0], %1;"
        :: "l"(address), "r"(value) : "memory");
}

/** Pack the ticket's aligned epoch/sequence publication key. */
CUTLASS_HOST_DEVICE constexpr uint64_t k3_mxfp8_epilogue_ticket_key(
        uint32_t epoch, uint32_t sequence) {
    return (static_cast<uint64_t>(sequence) << 32u) | epoch;
}

/** Atomically publish or invalidate one aligned ticket key. */
CUTLASS_DEVICE void k3_mxfp8_store_ticket_key_release_gpu(
        K3MxFp8EpiloguePanelTicket* ticket,
        uint32_t epoch, uint32_t sequence) {
    const uint64_t key = k3_mxfp8_epilogue_ticket_key(epoch, sequence);
    auto* const address = reinterpret_cast<uint64_t*>(ticket);
    asm volatile(
        "st.release.gpu.global.u64 [%0], %1;"
        :: "l"(address), "l"(key) : "memory");
}

/** Acquire the same non-torn ticket key used by producer publication. */
CUTLASS_DEVICE uint64_t k3_mxfp8_load_ticket_key_acquire_gpu(
        const K3MxFp8EpiloguePanelTicket* ticket) {
    return ptx::ld_acq_gpu(reinterpret_cast<const uint64_t*>(ticket));
}

#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
/** Dump one stalled slot and its first unretired tickets before trapping.
 *
 * This trace-only helper deliberately runs on one producer lane.  The compact
 * dump distinguishes a missing grouped-consumer retirement from a producer
 * that merely has not finished opening the next generation, without turning
 * the shared GPU into a printf flood.
 */
template <typename Ring, typename DiagnosticRing = Ring>
CUTLASS_DEVICE void k3_mxfp8_exact_ring_watchdog_slot_close(
        uint64_t wait_start, const Ring& ring, uint32_t slot,
        uint32_t production_ordinal, uint32_t required_closed_sequence,
        uint32_t reason, bool diagnostic_lane,
        const DiagnosticRing* diagnostic_ring = nullptr) {
    using Layout = typename Ring::Layout;
    if (!diagnostic_lane ||
        clock64() - wait_start <= kK3MxFp8ExactRingWatchdogCycles)
        return;

    const auto* const state = ring.slot_state(slot);
    printf(
        "K3_EXACT_RING_SLOT_WATCHDOG reason=%u block=%u thread=%u "
        "slot=%u production=%u required_closed=%u "
        "epoch=0x%08x generation=0x%08x retired=%u expected=%u "
        "closed=%u terminal=%u physical=%u valid_rows=%u\n",
        reason, static_cast<uint32_t>(blockIdx.x),
        static_cast<uint32_t>(threadIdx.x), slot, production_ordinal,
        required_closed_sequence, ptx::ld_acq(&state->epoch),
        ptx::ld_acq(&state->generation),
        ptx::ld_acq(&state->retired_tickets),
        ptx::ld_acq(&state->expected_tickets),
        ptx::ld_acq(&state->closed_sequence),
        ptx::ld_acq(&state->terminal_sequence),
        ptx::ld_acq(&state->physical_pool_block),
        ptx::ld_acq(&state->valid_rows));

    uint32_t missing = 0u;
    uint32_t printed = 0u;
    #pragma unroll 1
    for (uint32_t index = 0u; index < Layout::kTicketsPerSlot; ++index) {
        const auto* const ticket = ring.ticket(slot, index);
        const uint32_t retired = ptx::ld_acq(&ticket->retired_sequence);
        if (retired == required_closed_sequence)
            continue;
        ++missing;
        if (printed >= 8u)
            continue;
        const uint64_t key = k3_mxfp8_load_ticket_key_acquire_gpu(ticket);
        printf(
            "K3_EXACT_RING_UNRETIRED slot=%u ticket=%u key_epoch=0x%08x "
            "key_sequence=%u retired=%u arrivals=%u target=%u expert=%u "
            "ordinal=%u feature_group=0x%08x\n",
            slot, index, static_cast<uint32_t>(key),
            static_cast<uint32_t>(key >> 32u), retired,
            ptx::ld_acq(&ticket->reader_arrivals),
            ptx::ld_acq(&ticket->reader_target),
            ptx::ld_acq(&ticket->expert),
            ptx::ld_acq(&ticket->production_ordinal),
            ptx::ld_acq(&ticket->feature_and_group));
        ++printed;
    }
    printf(
        "K3_EXACT_RING_UNRETIRED_SUMMARY slot=%u missing=%u tickets=%u\n",
        slot, missing, Layout::kTicketsPerSlot);

    // Option-A owns disjoint A and B rings.  A retirement remains zero until
    // P01, so the companion dump distinguishes scheduler non-dispatch from a
    // task that acquired A and then parked on a missing B publication.
    if (diagnostic_ring != nullptr) {
        using DiagnosticLayout = typename DiagnosticRing::Layout;
        const uint32_t diagnostic_slot =
            DiagnosticLayout::slot_for_ordinal(
                production_ordinal, diagnostic_ring->depth);
        const auto* const diagnostic_state =
            diagnostic_ring->slot_state(diagnostic_slot);
        printf(
            "K3_EXACT_RING_COMPANION_SLOT slot=%u production=%u "
            "required_closed=%u epoch=0x%08x generation=0x%08x "
            "retired=%u expected=%u closed=%u terminal=%u physical=%u "
            "valid_rows=%u\n",
            diagnostic_slot, production_ordinal, required_closed_sequence,
            ptx::ld_acq(&diagnostic_state->epoch),
            ptx::ld_acq(&diagnostic_state->generation),
            ptx::ld_acq(&diagnostic_state->retired_tickets),
            ptx::ld_acq(&diagnostic_state->expected_tickets),
            ptx::ld_acq(&diagnostic_state->closed_sequence),
            ptx::ld_acq(&diagnostic_state->terminal_sequence),
            ptx::ld_acq(&diagnostic_state->physical_pool_block),
            ptx::ld_acq(&diagnostic_state->valid_rows));
        uint32_t diagnostic_missing = 0u;
        uint32_t diagnostic_printed = 0u;
        #pragma unroll 1
        for (uint32_t index = 0u;
             index < DiagnosticLayout::kTicketsPerSlot; ++index) {
            const auto* const ticket =
                diagnostic_ring->ticket(diagnostic_slot, index);
            const uint32_t retired =
                ptx::ld_acq(&ticket->retired_sequence);
            if (retired == required_closed_sequence)
                continue;
            ++diagnostic_missing;
            if (diagnostic_printed >= 8u)
                continue;
            const uint64_t key =
                k3_mxfp8_load_ticket_key_acquire_gpu(ticket);
            printf(
                "K3_EXACT_RING_COMPANION_UNRETIRED slot=%u ticket=%u "
                "key_epoch=0x%08x key_sequence=%u retired=%u arrivals=%u "
                "target=%u expert=%u ordinal=%u feature_group=0x%08x\n",
                diagnostic_slot, index, static_cast<uint32_t>(key),
                static_cast<uint32_t>(key >> 32u), retired,
                ptx::ld_acq(&ticket->reader_arrivals),
                ptx::ld_acq(&ticket->reader_target),
                ptx::ld_acq(&ticket->expert),
                ptx::ld_acq(&ticket->production_ordinal),
                ptx::ld_acq(&ticket->feature_and_group));
            ++diagnostic_printed;
        }
        printf(
            "K3_EXACT_RING_COMPANION_UNRETIRED_SUMMARY slot=%u "
            "missing=%u tickets=%u\n",
            diagnostic_slot, diagnostic_missing,
            DiagnosticLayout::kTicketsPerSlot);
    }
    asm volatile("trap;");
}
#endif

/** Any producer may claim a slot; one 64-bit CAS elects its initializer. */
template <typename Ring
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
          , typename DiagnosticRing = Ring
#endif
          >
CUTLASS_DEVICE void k3_mxfp8_epilogue_ring_ensure_slot_open(
        const Ring& ring, uint32_t production_ordinal,
        uint32_t physical_pool_block,
        uint32_t expert, uint32_t valid_rows
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
        ,
        const DiagnosticRing* diagnostic_ring = nullptr) {
#else
        ) {
#endif
    using Layout = typename Ring::Layout;
    DG_DEVICE_ASSERT(
        ring.depth != 0u &&
        production_ordinal < ring.total_pool_blocks &&
        physical_pool_block < ring.total_pool_blocks &&
        valid_rows <=
            Layout::kGroupsPerBlock * kK3MxFp8EpilogueRowsPerGroup);
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, ring.depth);
    auto* const state = ring.slot_state(slot);
    auto* const key = reinterpret_cast<unsigned long long*>(state);
    const uint64_t published =
        (static_cast<uint64_t>(sequence) << 32u) | ring.epoch;
    const uint64_t opening =
        (static_cast<uint64_t>(sequence | 0x80000000u) << 32u) |
        ring.epoch;
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
    const uint64_t wait_start = clock64();
#endif
    while (true) {
        const uint64_t observed = ptx::ld_acq_gpu(
            reinterpret_cast<const uint64_t*>(key));
        if (observed == published)
            return;
        if (observed == opening) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            k3_mxfp8_exact_ring_watchdog_slot_close(
                wait_start, ring, slot, production_ordinal,
                sequence - 1u, 0u,
                blockIdx.x == 132u && threadIdx.x == 384u,
                diagnostic_ring);
#endif
            continue;
        }
        if (sequence > 1u &&
            (ptx::ld_acq(&state->epoch) != ring.epoch ||
             ptx::ld_acq(&state->closed_sequence) != sequence - 1u)) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            k3_mxfp8_exact_ring_watchdog_slot_close(
                wait_start, ring, slot, production_ordinal,
                sequence - 1u, 1u,
                blockIdx.x == 132u && threadIdx.x == 384u,
                diagnostic_ring);
#endif
            continue;
        }
        if (atomicCAS(key, observed, opening) != observed)
            continue;
        break;
    }
    state->retired_tickets = 0u;
    state->closed_sequence = sequence - 1u;
    state->physical_pool_block = physical_pool_block;
    state->terminal_sequence = Layout::terminal_sequence_for_slot(
        ring.total_pool_blocks, slot, ring.depth);
    state->valid_rows = valid_rows;
    state->expected_tickets = Layout::kTicketsPerSlot;
    __threadfence();
    asm volatile(
        "st.release.gpu.global.u64 [%0], %1;"
        :: "l"(key), "l"(published) : "memory");
}

template <typename Ring>
CUTLASS_DEVICE K3MxFp8EpiloguePanelTicket*
k3_mxfp8_epilogue_ring_prepare_ticket(
        const Ring& ring, uint32_t production_ordinal,
        uint32_t physical_pool_block,
        uint32_t expert, uint32_t feature_panel,
        uint32_t group_in_block, uint32_t reader_target) {
    using Layout = typename Ring::Layout;
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t index = Layout::ticket_index(
        feature_panel, group_in_block);
    auto* const ticket = ring.ticket(slot, index);
    // Invalidate the complete key before touching metadata.  In particular,
    // never expose a new epoch beside a stale matching sequence from an older
    // launch or wrapped ring generation.
    k3_mxfp8_store_ticket_key_release_gpu(ticket, ring.epoch, 0u);
    ticket->retired_sequence = sequence - 1u;
    ticket->reader_arrivals = 0u;
    ticket->reader_target = reader_target;
    ticket->expert = expert;
    ticket->production_ordinal = production_ordinal;
    ticket->feature_and_group = Layout::feature_and_group(
        feature_panel, group_in_block);
    return ticket;
}

}  // namespace detail

/** Invalidate every physical slot before opening a new launch epoch.
 *
 * After one backward finishes, direct dX deliberately reuses the ring planes
 * as ordinary BF16 storage.  Those arbitrary bytes can alias a future
 * `{epoch, sequence}` publication key.  All persistent CTAs therefore clear
 * the aligned 64-bit ticket and slot-state keys, then the caller carries the
 * writes through a full-grid release barrier before publishing the new epoch.
 * Sequence zero is never a valid ring generation.
 */
template <typename Ring>
CUTLASS_DEVICE void k3_mxfp8_epilogue_ring_clear_publication_keys(
        const Ring& ring, uint32_t global_thread,
        uint32_t global_threads) {
    using Layout = typename Ring::Layout;
    constexpr uint32_t kKeysPerSlot = Layout::kTicketsPerSlot + 1u;
    DG_DEVICE_ASSERT(ring.depth != 0u && global_threads != 0u);
    const uint64_t total_keys =
        static_cast<uint64_t>(ring.depth) * kKeysPerSlot;
    for (uint64_t linear = global_thread; linear < total_keys;
         linear += global_threads) {
        const uint32_t slot = static_cast<uint32_t>(
            linear / kKeysPerSlot);
        const uint32_t key_in_slot = static_cast<uint32_t>(
            linear % kKeysPerSlot);
        auto* const key = key_in_slot < Layout::kTicketsPerSlot
            ? reinterpret_cast<uint64_t*>(
                  ring.ticket(slot, key_in_slot))
            : reinterpret_cast<uint64_t*>(ring.slot_state(slot));
        *key = 0ull;
    }
}

/** Clear both K3 dW13 rings without extending parent-kernel live ranges. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kPoolBlockRows>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_clear_dw13_ab_ring_publication_keys(
        uint8_t* combine_plane_zero, uint32_t capacity_rows,
        uint32_t total_pool_blocks, uint32_t epoch,
        uint32_t global_thread, uint32_t global_threads) {
    const auto a_ring = k3_mxfp8_make_epilogue_panel_ring<
        kHidden, 2u * kIntermediateHidden, kPoolBlockRows>(
            combine_plane_zero, capacity_rows, total_pool_blocks, epoch);
    const auto b_ring = k3_mxfp8_make_dw13b_epilogue_panel_ring<
        kHidden, kPoolBlockRows>(
            combine_plane_zero, capacity_rows, total_pool_blocks, epoch);
    DG_DEVICE_ASSERT(
        a_ring.depth == b_ring.depth && a_ring.depth != 0u);
    k3_mxfp8_epilogue_ring_clear_publication_keys(
        a_ring, global_thread, global_threads);
    k3_mxfp8_epilogue_ring_clear_publication_keys(
        b_ring, global_thread, global_threads);
}

/** Acquire one publication before issuing the ticket's first TMA read. */
template <typename Ring>
CUTLASS_DEVICE K3MxFp8EpiloguePanelTicket*
k3_mxfp8_epilogue_ring_acquire_ticket(
        const Ring& ring, uint32_t production_ordinal,
        uint32_t physical_pool_block,
        uint32_t feature_panel, uint32_t group_in_block) {
    using Layout = typename Ring::Layout;
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, ring.depth);
    auto* const ticket = ring.ticket(
        slot, Layout::ticket_index(feature_panel, group_in_block));
    const uint64_t expected = detail::k3_mxfp8_epilogue_ticket_key(
        ring.epoch, sequence);
    uint64_t observed =
        detail::k3_mxfp8_load_ticket_key_acquire_gpu(ticket);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
    const uint64_t wait_start = clock64();
#endif
    while (observed != expected) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
        k3_mxfp8_exact_ring_watchdog<3u>(
            wait_start, static_cast<uint32_t>(observed),
            ring.epoch, production_ordinal,
            blockIdx.x == 132u && (threadIdx.x & 31u) == 0u);
#endif
        observed = detail::k3_mxfp8_load_ticket_key_acquire_gpu(ticket);
    }
    DG_DEVICE_ASSERT(
        ticket->production_ordinal == production_ordinal &&
        ring.slot_state(slot)->physical_pool_block == physical_pool_block &&
        ticket->feature_and_group == Layout::feature_and_group(
            feature_panel, group_in_block));
    return ticket;
}

/** Retire only after P01's final input read; the last reader closes the slot. */
template <typename Ring>
CUTLASS_DEVICE void k3_mxfp8_epilogue_ring_retire_after_p01(
        const Ring& ring, K3MxFp8EpiloguePanelTicket* ticket) {
    using Layout = typename Ring::Layout;
    const uint64_t key =
        detail::k3_mxfp8_load_ticket_key_acquire_gpu(ticket);
    const uint32_t epoch = static_cast<uint32_t>(key);
    const uint32_t sequence = static_cast<uint32_t>(key >> 32u);
    DG_DEVICE_ASSERT(
        epoch == ring.epoch && sequence != 0u &&
        ticket->reader_target != 0u);
    const uint32_t previous = ptx::atomic_add_acq_rel(
        &ticket->reader_arrivals, 1u);
    DG_DEVICE_ASSERT(previous < ticket->reader_target);
    if (previous + 1u != ticket->reader_target)
        return;

    __threadfence();
    detail::k3_mxfp8_store_release_gpu(
        &ticket->retired_sequence, sequence);
    const uint32_t slot = Layout::slot_for_ordinal(
        ticket->production_ordinal, ring.depth);
    auto* const state = ring.slot_state(slot);
    const uint32_t retired = ptx::atomic_add_acq_rel(
        &state->retired_tickets, 1u);
    DG_DEVICE_ASSERT(retired < Layout::kTicketsPerSlot);
    if (retired + 1u == Layout::kTicketsPerSlot) {
        __threadfence();
        detail::k3_mxfp8_store_release_gpu(
            &state->closed_sequence, sequence);
    }
}

/** Remote W13 writers call this before clearing/overwriting a scratch slot. */
template <typename Ring>
CUTLASS_DEVICE void k3_mxfp8_epilogue_ring_wait_terminal_retirement(
        const Ring& ring, uint32_t slot) {
    using Layout = typename Ring::Layout;
    DG_DEVICE_ASSERT(slot < ring.depth);
    const uint32_t terminal = Layout::terminal_sequence_for_slot(
        ring.total_pool_blocks, slot, ring.depth);
    if (terminal == 0u)
        return;
    const auto* const state = ring.slot_state(slot);
    uint32_t observed_epoch = ptx::ld_acq(&state->epoch);
    uint32_t observed_sequence = ptx::ld_acq(&state->closed_sequence);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
    const uint64_t wait_start = clock64();
#endif
    while (observed_epoch != ring.epoch ||
           observed_sequence != terminal) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
        k3_mxfp8_exact_ring_watchdog<4u>(
            wait_start, observed_sequence, terminal, slot,
            blockIdx.x == 0u && threadIdx.x == 0u);
#endif
        observed_epoch = ptx::ld_acq(&state->epoch);
        observed_sequence = ptx::ld_acq(&state->closed_sequence);
    }
}

/** Grouped-body lifecycle for rolling physical dW13-A panels.
 *
 * The grouped scheduler names a compact, reverse-range-concatenated expert K
 * axis. One K128 load can cross a physical 192-row block boundary, so this
 * lifecycle resolves and acquires four independent group-32 tickets. Each
 * output tile covers two A feature panels and one of the 28 B/output-N
 * panels. Retirement walks the same coordinates only after final P01.
 */
template <uint32_t kNumExperts, uint32_t kMaxRanges,
          uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
struct K3MxFp8EpilogueGroupedConsumerLifecycle {
    static constexpr bool kEnabled = true;
    static constexpr bool kAEnabled = true;
    static constexpr bool kBEnabled = false;
    using Ring = K3MxFp8EpiloguePanelRingView<
        kHidden, kLogicalWidth, kPoolBlockRows>;
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
        DG_DEVICE_ASSERT(
            expert < kNumExperts &&
            compact_k_row >= value_prefix[expert] &&
            compact_k_row < value_prefix[expert + 1u]);
        const uint32_t physical_pool_row =
            detail::k3_mxfp8_expert_source_pool_row<
                kNumExperts, kPoolBlockRows, kMaxRanges>(
                    expert,
                    compact_k_row - value_prefix[expert],
                    expert_counts, *backward_ranges,
                    physical_range_prefix);
        DG_DEVICE_ASSERT(
            physical_pool_row != static_cast<uint32_t>(-1) &&
            physical_pool_row % kK3MxFp8EpilogueRowsPerGroup == 0u);
        const uint32_t expert_local_k =
            compact_k_row - value_prefix[expert];
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
        DG_DEVICE_ASSERT(
            coord.reader_panel < kHidden / 128u &&
            coord.slot < ring.depth);
        auto* const ticket = k3_mxfp8_epilogue_ring_acquire_ticket(
            ring, coord.production_ordinal,
            coord.physical_pool_block,
            coord.feature_panel, coord.group_in_block);
        DG_DEVICE_ASSERT(
            ticket->expert == coord.expert &&
            ticket->reader_target == kHidden / 128u);
    }

    CUTLASS_DEVICE void retire_coordinate_after_p01(
            const K3MxFp8EpiloguePanelConsumerCoordinate& coord) const {
        auto* const ticket = ring.ticket(
            coord.slot,
            Layout::ticket_index(
                coord.feature_panel, coord.group_in_block));
        DG_DEVICE_ASSERT(
            detail::k3_mxfp8_load_ticket_key_acquire_gpu(ticket) ==
                detail::k3_mxfp8_epilogue_ticket_key(
                    ring.epoch, coord.sequence) &&
            ticket->expert == coord.expert &&
            coord.reader_panel < ticket->reader_target);
        k3_mxfp8_epilogue_ring_retire_after_p01(ring, ticket);
    }

    CUTLASS_DEVICE void acquire(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        DG_DEVICE_ASSERT(
            base_m % kK3MxFp8EpilogueFeaturePanel == 0u &&
            base_n % 128u == 0u && valid_k != 0u &&
            valid_k <= 128u &&
            valid_k % kK3MxFp8EpilogueRowsPerGroup == 0u);
        const uint32_t first_feature_panel =
            base_m / kK3MxFp8EpilogueFeaturePanel;
        const uint32_t reader_panel = base_n / 128u;
        #pragma unroll
        for (uint32_t panel = 0u; panel < 2u; ++panel) {
            #pragma unroll
            for (uint32_t group = 0u; group < 4u; ++group) {
                if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                    break;
                acquire_coordinate(coordinate(
                    expert, first_feature_panel + panel, reader_panel,
                    compact_k_begin +
                        group * kK3MxFp8EpilogueRowsPerGroup));
            }
        }
    }

    /** Four physical value fragments plus one native compact scale TMA. */
    CUTLASS_DEVICE void load_k128_stage(
            uint32_t expert, uint32_t value_feature_begin,
            uint32_t scale_feature_begin, uint32_t reader_panel,
            uint32_t compact_k_begin, uint32_t valid_k,
            bool residual,
            const cute::TmaDescriptor* value_map,
            const cute::TmaDescriptor* scale_map,
            uint32_t compact_scale_row,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value, uint32_t* smem_scale,
            uint32_t& expected_bytes) const {
        DG_DEVICE_ASSERT(
            valid_k != 0u && valid_k <= 128u &&
            valid_k % kK3MxFp8EpilogueRowsPerGroup == 0u &&
            value_feature_begin % 128u == 0u &&
            scale_feature_begin % 256u == 0u);
        (void)residual;
        constexpr uint32_t kGroupValueBytes = 128u * 32u;
        #pragma unroll
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * 32u >= valid_k)
                break;
            const auto local_coord = coordinate(
                expert, value_feature_begin / 128u, reader_panel,
                compact_k_begin + group * 32u);
            acquire_coordinate(local_coord);
            // Scale staging spans both 128-feature panels even though this
            // CTA's value TMA owns one panel. Acquire the sibling publication
            // before reading its packed-scale bytes as well.
            acquire_coordinate(coordinate(
                expert, scale_feature_begin / 128u +
                    static_cast<uint32_t>(
                        value_feature_begin == scale_feature_begin),
                reader_panel, compact_k_begin + group * 32u));
            asm volatile("fence.proxy.async.global;" ::: "memory");
            tma::copy<128u, 32u, 64u, uint8_t>(
                value_map, full_barrier,
                smem_value + group * kGroupValueBytes,
                value_feature_begin, local_coord.slot_row_begin, 2u);
            expected_bytes += kGroupValueBytes;
        }

        // The SiTU producer writes its bytes directly into the compact
        // expert scale rows in UTCCP-native order. Keep the established
        // descriptor-matched 256-word copy and its transaction accounting.
        tma::copy<256u, 1u, 0u>(
            scale_map, full_barrier, smem_scale,
            scale_feature_begin, compact_scale_row, 2u);
        expected_bytes += 256u * sizeof(uint32_t);
    }

    /** Operand-tagged adapter used by the dual A/B grouped-body hook. */
    CUTLASS_DEVICE __noinline__ void load_a_k128_stage(
            uint32_t expert, uint32_t value_feature_begin,
            uint32_t scale_feature_begin, uint32_t reader_panel,
            uint32_t compact_k_begin, uint32_t valid_k,
            bool residual,
            const cute::TmaDescriptor* value_map,
            const cute::TmaDescriptor* scale_map,
            uint32_t compact_scale_row,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value, uint32_t* smem_scale,
            uint32_t& expected_bytes) const {
        load_k128_stage(
            expert, value_feature_begin, scale_feature_begin,
            reader_panel, compact_k_begin, valid_k, residual,
            value_map, scale_map, compact_scale_row, full_barrier,
            smem_value, smem_scale, expected_bytes);
    }

    CUTLASS_DEVICE void retire_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        DG_DEVICE_ASSERT(
            base_m % kK3MxFp8EpilogueFeaturePanel == 0u &&
            base_n % 128u == 0u &&
            valid_k != 0u && valid_k <= 128u &&
            valid_k % kK3MxFp8EpilogueRowsPerGroup == 0u);
        const uint32_t first_feature_panel =
            base_m / kK3MxFp8EpilogueFeaturePanel;
        const uint32_t reader_panel = base_n / 128u;
        #pragma unroll
        for (uint32_t group = 0u; group < 4u; ++group) {
            if (group * kK3MxFp8EpilogueRowsPerGroup >= valid_k)
                break;
            #pragma unroll
            for (uint32_t panel = 0u; panel < 2u; ++panel) {
                retire_coordinate_after_p01(coordinate(
                    expert, first_feature_panel + panel, reader_panel,
                    compact_k_begin +
                        group * kK3MxFp8EpilogueRowsPerGroup));
            }
        }
    }

    /** Operand-tagged adapter used by the dual A/B grouped-body hook. */
    CUTLASS_DEVICE __noinline__ void retire_a_k128_after_p01(
            uint32_t expert, uint32_t base_m,
            uint32_t base_n, uint32_t compact_k_begin,
            uint32_t valid_k) const {
        // One adapter invocation represents one N128 reader of the cluster's
        // shared M256 A tile. The sole P01 leader invokes this adapter for
        // both adjacent N128 CTA identities; each invocation retires two M128
        // feature panels times four group-32 tickets.
        retire_k128_after_p01(
            expert, base_m, base_n, compact_k_begin, valid_k);
    }
};

/** Four-fragment K128 gather into the grouped body's existing A/SFA stage.
 *
 * This bounded compile slice makes the non-contiguous physical-ring problem
 * explicit without changing UMMA or TMEM. The elected loader acquires and
 * issues four 128x32 FP8 TMA copies into offsets 0/32/64/96 of one contiguous
 * 128x128 shared A stage. The 128 feature owners gather one selected byte from
 * each physical packed-scale row into the existing 128-word UTCCP SFA stage.
 * Retirement is a separate edge and must be called only after
 * the consuming P01 has released this operand stage.
 */
template <uint32_t kNumExperts, uint32_t kMaxRanges,
          uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
struct K3MxFp8EpilogueBoundedK128Gather {
    using Lifecycle = K3MxFp8EpilogueGroupedConsumerLifecycle<
        kNumExperts, kMaxRanges, kHidden, kLogicalWidth,
        kPoolBlockRows>;
    using Coordinate = K3MxFp8EpiloguePanelConsumerCoordinate;
    static constexpr uint32_t kGroups = 4u;
    static constexpr uint32_t kPanelFeatures = 128u;
    static constexpr uint32_t kRowsPerGroup = 32u;
    static constexpr uint32_t kValueBytes =
        kPanelFeatures * kRowsPerGroup;
    static constexpr uint32_t kExpectedValueBytes =
        kGroups * kValueBytes;

    CUTLASS_DEVICE static void load(
            const Lifecycle& lifecycle,
            uint32_t expert,
            uint32_t feature_begin,
            uint32_t reader_panel,
            uint32_t compact_k_begin,
            const cute::TmaDescriptor& value_map,
            const uint32_t* packed_scales,
            cutlass::arch::ClusterTransactionBarrier* full_barrier,
            uint8_t* smem_value,
            uint32_t* smem_scale,
            Coordinate* smem_coordinates,
            uint32_t thread_idx) {
        DG_DEVICE_ASSERT(
            feature_begin % kPanelFeatures == 0u &&
            thread_idx < kPanelFeatures);
        if (thread_idx == 0u) {
            full_barrier->init(1u);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        if (thread_idx == 0u) {
            #pragma unroll
            for (uint32_t group = 0u; group < kGroups; ++group) {
                const auto coord = lifecycle.coordinate(
                    expert, feature_begin / kPanelFeatures,
                    reader_panel,
                    compact_k_begin + group * kRowsPerGroup);
                lifecycle.acquire_coordinate(coord);
                smem_coordinates[group] = coord;
                // One descriptor names the physical rolling value plane.
                // Four bounded copies may cross a slot or range boundary but
                // always land contiguously in the established K128 stage.
                tma::copy<
                    kPanelFeatures, kRowsPerGroup, 64u, uint8_t>(
                        &value_map, full_barrier,
                        smem_value + group * kValueBytes,
                        feature_begin, coord.slot_row_begin, 1u);
            }
            full_barrier->arrive_and_expect_tx(kExpectedValueBytes);
        }
        __syncthreads();

        // The elected device acquires are published through the CTA barrier
        // before feature owners gather the four physical scale bytes.
        uint32_t gathered_scale = 0u;
        // `thread_idx` names the physical/native slot in the established
        // 256-word SFA stage.  The producer already permuted logical feature
        // ownership when publishing its packed words, so indexing the ring by
        // this native slot preserves the exact UTCCP stage layout.
        const uint32_t native_feature = feature_begin + thread_idx;
        #pragma unroll
        for (uint32_t group = 0u; group < kGroups; ++group) {
            const auto& coord = smem_coordinates[group];
            const uint32_t physical_word = packed_scales[
                static_cast<uint64_t>(coord.packed_scale_row) *
                    kLogicalWidth + native_feature];
            gathered_scale |=
                ((physical_word >> (coord.packed_scale_byte * 8u)) &
                 0xffu) << (group * 8u);
        }
        smem_scale[thread_idx] = gathered_scale;
        __syncthreads();
        if (thread_idx == 0u)
            full_barrier->wait(0u);
        __syncthreads();
    }

    CUTLASS_DEVICE static void retire_after_final_p01(
            const Lifecycle& lifecycle,
            const Coordinate* smem_coordinates,
            uint32_t thread_idx) {
        if (thread_idx == 0u) {
            #pragma unroll
            for (uint32_t group = 0u; group < kGroups; ++group)
                lifecycle.retire_coordinate_after_p01(
                    smem_coordinates[group]);
        }
        __syncthreads();
    }
};

/** Convert one producer-owned 32x128 BF16 panel to exact two-term MXFP8.
 *
 * Exactly 128 participating threads call this function with distinct
 * `panel_thread` values.  One thread owns one feature and therefore computes
 * both group-32 scales without a cross-thread reduction.  The residual is
 * rounded to BF16 before its amax and E4M3 conversion, matching the terminal
 * producer byte-for-byte.  Values are written token-major; the established
 * TMA descriptors expose the transposed UMMA view.  Scale words are updated
 * directly in UTCCP-native order, so no later software transpose is needed.
 *
 * The caller owns publication.  It must join all 128 writers, execute the
 * generic/async proxy fences appropriate for the destination, and only then
 * release-store the complete ticket key.  This separation makes the primitive
 * usable from both the SiTU epilogue and reverse-dispatch producer without
 * embedding either parent's barrier policy here.
 */
template <uint32_t kLogicalWidth, bool kSourceInterleaved = false>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_quantize_exact2_epilogue_panel(
        const cutlass::bfloat16_t* source,
        uint32_t source_row_stride,
        uint32_t source_row_begin,
        uint32_t valid_rows,
        uint32_t logical_feature_begin,
        uint32_t source_feature_begin,
        uint8_t* primary_values,
        uint8_t* residual_values,
        uint32_t destination_row_stride,
        uint32_t compact_row_begin,
        uint32_t* primary_packed_scales,
        uint32_t* residual_packed_scales,
        uint32_t packed_scale_row,
        uint32_t group_in_scale_row,
        bool final_expert_group,
        uint32_t panel_thread) {
    constexpr uint32_t kRows = kK3MxFp8EpilogueRowsPerGroup;
    constexpr uint32_t kFeatures = kK3MxFp8EpilogueFeaturePanel;
    static_assert(kLogicalWidth % kFeatures == 0u);

    DG_DEVICE_ASSERT(
        panel_thread < kFeatures && valid_rows <= kRows &&
        group_in_scale_row < kK3MxFp8EpilogueGroupsPerScaleWord &&
        logical_feature_begin + kFeatures <= kLogicalWidth);

    const uint32_t logical_feature =
        logical_feature_begin + panel_thread;
    uint32_t source_feature = source_feature_begin + panel_thread;
    if constexpr (kSourceInterleaved) {
        static_assert(kLogicalWidth % 2u == 0u);
        constexpr uint32_t kBranchWidth = kLogicalWidth / 2u;
        const uint32_t branch_feature = logical_feature % kBranchWidth;
        source_feature = (branch_feature / 8u) * 16u +
            (logical_feature >= kBranchWidth ? 8u : 0u) +
            (branch_feature & 7u);
    }

    float primary_amax = 0.0f;
    #pragma unroll
    for (uint32_t row = 0u; row < kRows; ++row) {
        const float value = row < valid_rows
            ? static_cast<float>(source[
                  static_cast<uint64_t>(source_row_begin + row) *
                      source_row_stride + source_feature])
            : 0.0f;
        primary_amax = cute::max(primary_amax, cute::abs(value));
    }

    float primary_scale = 1.0f;
    float primary_scale_inv = 1.0f;
    uint32_t primary_scale_byte = 0x7fu;
    detail::k3_mxfp8_scale_pair(
        primary_amax, primary_scale, primary_scale_inv,
        primary_scale_byte);

    // Sixteen packed BF16 pairs retain the exact rounded residual without a
    // shared-memory allocation and avoid a third read of the producer output.
    uint32_t residual_pairs[kRows / 2u];
    float residual_amax = 0.0f;
    #pragma unroll 1
    for (uint32_t row = 0u; row < kRows; row += 4u) {
        float values[4];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            values[i] = row + i < valid_rows
                ? static_cast<float>(source[
                      static_cast<uint64_t>(source_row_begin + row + i) *
                          source_row_stride + source_feature])
                : 0.0f;
        }
        const auto primary = __nv_fp8x4_e4m3(make_float4(
            values[0] * primary_scale_inv,
            values[1] * primary_scale_inv,
            values[2] * primary_scale_inv,
            values[3] * primary_scale_inv));
        const float4 primary_float = static_cast<float4>(primary);
        const __nv_bfloat162 residual01 = __float22bfloat162_rn(make_float2(
            values[0] - primary_float.x * primary_scale,
            values[1] - primary_float.y * primary_scale));
        const __nv_bfloat162 residual23 = __float22bfloat162_rn(make_float2(
            values[2] - primary_float.z * primary_scale,
            values[3] - primary_float.w * primary_scale));
        residual_pairs[row / 2u] =
            *reinterpret_cast<const uint32_t*>(&residual01);
        residual_pairs[row / 2u + 1u] =
            *reinterpret_cast<const uint32_t*>(&residual23);
        const float2 rounded01 = __bfloat1622float2(residual01);
        const float2 rounded23 = __bfloat1622float2(residual23);
        residual_amax = cute::max(residual_amax, cute::abs(rounded01.x));
        residual_amax = cute::max(residual_amax, cute::abs(rounded01.y));
        residual_amax = cute::max(residual_amax, cute::abs(rounded23.x));
        residual_amax = cute::max(residual_amax, cute::abs(rounded23.y));

        const uint32_t primary_bits = primary.__x;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            primary_values[
                static_cast<uint64_t>(compact_row_begin + row + i) *
                    destination_row_stride + logical_feature] =
                static_cast<uint8_t>(primary_bits >> (i * 8u));
        }
    }

    float residual_scale = 1.0f;
    float residual_scale_inv = 1.0f;
    uint32_t residual_scale_byte = 0x7fu;
    detail::k3_mxfp8_scale_pair(
        residual_amax, residual_scale, residual_scale_inv,
        residual_scale_byte);
    #pragma unroll 1
    for (uint32_t row = 0u; row < kRows; row += 4u) {
        const float2 residual01 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(
                residual_pairs + row / 2u));
        const float2 residual23 = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(
                residual_pairs + row / 2u + 1u));
        const auto residual = __nv_fp8x4_e4m3(make_float4(
            residual01.x * residual_scale_inv,
            residual01.y * residual_scale_inv,
            residual23.x * residual_scale_inv,
            residual23.y * residual_scale_inv));
        const uint32_t residual_bits = residual.__x;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            residual_values[
                static_cast<uint64_t>(compact_row_begin + row + i) *
                    destination_row_stride + logical_feature] =
                static_cast<uint8_t>(residual_bits >> (i * 8u));
        }
    }

    const uint32_t native_feature =
        k3_mxfp8_utccp_scale_feature(logical_feature);
    const uint64_t packed_byte_offset =
        (static_cast<uint64_t>(packed_scale_row) * kLogicalWidth +
         native_feature) * kK3MxFp8EpilogueGroupsPerScaleWord +
        group_in_scale_row;
    // Neighboring physical blocks may be produced by different CTAs while
    // contributing distinct bytes to the same compact scale word.  Publish
    // disjoint byte locations rather than racing whole-word read/modify/write
    // stores.  An expert owns its final word, so its final group also fills
    // any tail bytes with the MX zero-scale encoding.
    auto* const primary_scale_bytes =
        reinterpret_cast<uint8_t*>(primary_packed_scales);
    auto* const residual_scale_bytes =
        reinterpret_cast<uint8_t*>(residual_packed_scales);
    primary_scale_bytes[packed_byte_offset] =
        static_cast<uint8_t>(primary_scale_byte);
    residual_scale_bytes[packed_byte_offset] =
        static_cast<uint8_t>(residual_scale_byte);
    if (final_expert_group) {
        #pragma unroll
        for (uint32_t byte = group_in_scale_row + 1u;
             byte < kK3MxFp8EpilogueGroupsPerScaleWord; ++byte) {
            const uint64_t tail_offset = packed_byte_offset +
                byte - group_in_scale_row;
            primary_scale_bytes[tail_offset] = 0x7fu;
            residual_scale_bytes[tail_offset] = 0x7fu;
        }
    }
}

/** SiTU epilogue adapter for one completed group-32/feature-128 panel.
 *
 * Every epilogue thread calls this adapter. The named barrier rendezvous the
 * epilogue role, including writers outside the first 128 threads.  The first
 * rendezvous therefore publishes the just-written BF16 derivative before the
 * 128-thread exact quantizer rereads it.  The second publishes all value and
 * UTCCP scale writers before lane zero release-publishes the ticket.
 */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows, uint32_t kReaderTarget,
          bool kSourceInterleaved, uint32_t kBarrierThreads,
          uint32_t kBarrierId>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_situ_dw13a_epilogue_group(
        const K3MxFp8EpiloguePanelRingView<
            kHidden, kLogicalWidth, kPoolBlockRows>& ring,
        const cutlass::bfloat16_t* source,
        uint32_t source_row_stride,
        uint32_t production_ordinal,
        uint32_t physical_pool_block,
        uint32_t expert,
        uint32_t block_valid_rows,
        uint32_t logical_feature_begin,
        uint32_t group_in_block,
        uint32_t packed_scale_row,
        uint32_t packed_scale_byte,
        bool final_expert_group,
        uint32_t* primary_packed_scales,
        uint32_t* residual_packed_scales,
        uint32_t epilogue_thread) {
    using Layout = K3MxFp8EpiloguePanelRingLayout<
        kHidden, kLogicalWidth, kPoolBlockRows>;
    static_assert(kReaderTarget == kHidden / 128u);
    DG_DEVICE_ASSERT(
        ring.depth != 0u &&
        physical_pool_block < ring.total_pool_blocks &&
        logical_feature_begin % kK3MxFp8EpilogueFeaturePanel == 0u &&
        logical_feature_begin + kK3MxFp8EpilogueFeaturePanel <=
            kLogicalWidth &&
        group_in_block < Layout::kGroupsPerBlock &&
        packed_scale_byte <
            kK3MxFp8EpilogueGroupsPerScaleWord &&
        block_valid_rows <= kPoolBlockRows);

    const uint32_t feature_panel =
        logical_feature_begin / kK3MxFp8EpilogueFeaturePanel;
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, ring.depth);
    const uint32_t source_row_begin =
        physical_pool_block * kPoolBlockRows +
        group_in_block * kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t slot_row_begin =
        ring.slot_row_begin(slot) +
        group_in_block * kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t row_begin =
        group_in_block * kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t remaining_rows = row_begin >= block_valid_rows
        ? 0u : block_valid_rows - row_begin;
    const uint32_t valid_rows =
        remaining_rows < kK3MxFp8EpilogueRowsPerGroup
        ? remaining_rows : kK3MxFp8EpilogueRowsPerGroup;

    // Every BF16 writer publishes its own generic store before the role join.
    __threadfence();
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);
    if (epilogue_thread == 0u) {
        detail::k3_mxfp8_epilogue_ring_ensure_slot_open(
            ring, production_ordinal, physical_pool_block,
            expert, block_valid_rows);
        detail::k3_mxfp8_epilogue_ring_prepare_ticket(
            ring, production_ordinal, physical_pool_block,
            expert, feature_panel,
            group_in_block, kReaderTarget);
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);

    if (epilogue_thread < kK3MxFp8EpilogueFeaturePanel) {
        k3_mxfp8_quantize_exact2_epilogue_panel<
            kLogicalWidth, kSourceInterleaved>(
                source, source_row_stride,
                source_row_begin, valid_rows,
                logical_feature_begin, logical_feature_begin,
                ring.primary_values, ring.residual_values,
                Layout::kRowBytes, slot_row_begin,
                primary_packed_scales, residual_packed_scales,
                packed_scale_row,
                packed_scale_byte,
                final_expert_group,
                epilogue_thread);
        // The consumer uses TMA's async proxy. Every generic writer fences
        // its own bytes before the elected publication lane releases them.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);
    if (epilogue_thread == 0u) {
        auto* const ticket = ring.ticket(
            slot, Layout::ticket_index(
                feature_panel, group_in_block));
        detail::k3_mxfp8_store_ticket_key_release_gpu(
            ticket, ring.epoch, sequence);
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);
}

/** Publish both canonical dW13-A panels from one staged BF16 group-32. */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows, uint32_t kReaderTarget,
          uint32_t kBarrierThreads, uint32_t kBarrierId>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_situ_dw13a_staged_group(
        K3MxFp8EpilogueParentSharedContext<
            kHidden, kLogicalWidth, kPoolBlockRows>* context,
        const cutlass::bfloat16_t* staged_gate,
        const cutlass::bfloat16_t* staged_up,
        uint32_t group_in_block,
        uint32_t epilogue_thread) {
    using Layout = K3MxFp8EpiloguePanelRingLayout<
        kHidden, kLogicalWidth, kPoolBlockRows>;
    const uint32_t expert_local_group =
        context->producer_scale_block.expert_local_group_begin +
        group_in_block;
    const uint32_t packed_scale_row =
        context->producer_scale_block.expert_scale_prefix +
        expert_local_group / kK3MxFp8EpilogueGroupsPerScaleWord;
    const uint32_t packed_scale_byte =
        expert_local_group % kK3MxFp8EpilogueGroupsPerScaleWord;
    const bool final_expert_group =
        expert_local_group + 1u ==
        context->producer_scale_block.expert_num_groups;
    const uint32_t production_ordinal =
        context->producer_scale_block.production_ordinal;
    const uint32_t slot = Layout::slot_for_ordinal(
        production_ordinal, context->producer_ring.depth);
    const uint32_t sequence = Layout::sequence_for_ordinal(
        production_ordinal, context->producer_ring.depth);
    const uint32_t row_begin =
        group_in_block * kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t remaining_rows = row_begin >= context->block_valid_rows
        ? 0u : context->block_valid_rows - row_begin;
    const uint32_t valid_rows = remaining_rows <
            kK3MxFp8EpilogueRowsPerGroup
        ? remaining_rows : kK3MxFp8EpilogueRowsPerGroup;
    const uint32_t slot_row_begin =
        context->producer_ring.slot_row_begin(slot) + row_begin;

    if (epilogue_thread == 0u) {
        detail::k3_mxfp8_epilogue_ring_ensure_slot_open(
            context->producer_ring, production_ordinal,
            context->physical_pool_block,
            context->expert, context->block_valid_rows);
    }
    if (epilogue_thread < 2u) {
        const uint32_t feature_panel =
            (context->feature_block + epilogue_thread *
                (kLogicalWidth / 2u)) /
            kK3MxFp8EpilogueFeaturePanel;
        detail::k3_mxfp8_epilogue_ring_prepare_ticket(
            context->producer_ring, production_ordinal,
            context->physical_pool_block,
            context->expert, feature_panel, group_in_block,
            kReaderTarget);
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);

    if (epilogue_thread < 2u * kK3MxFp8EpilogueFeaturePanel) {
        const uint32_t branch = epilogue_thread /
            kK3MxFp8EpilogueFeaturePanel;
        const uint32_t panel_thread = epilogue_thread %
            kK3MxFp8EpilogueFeaturePanel;
        k3_mxfp8_quantize_exact2_epilogue_panel<kLogicalWidth, false>(
            branch == 0u ? staged_gate : staged_up,
            kK3MxFp8EpilogueFeaturePanel, 0u, valid_rows,
            branch * (kLogicalWidth / 2u) + context->feature_block,
            0u,
            context->producer_ring.primary_values,
            context->producer_ring.residual_values,
            Layout::kRowBytes, slot_row_begin,
            context->primary_packed_scales,
            context->residual_packed_scales,
            packed_scale_row, packed_scale_byte,
            final_expert_group, panel_thread);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);
    if (epilogue_thread < 2u) {
        const uint32_t feature_panel =
            (context->feature_block + epilogue_thread *
                (kLogicalWidth / 2u)) /
            kK3MxFp8EpilogueFeaturePanel;
        auto* const ticket = context->producer_ring.ticket(
            slot, Layout::ticket_index(feature_panel, group_in_block));
        detail::k3_mxfp8_store_ticket_key_release_gpu(
            ticket, context->producer_ring.epoch, sequence);
    }
    cutlass::arch::NamedBarrier::sync(kBarrierThreads, kBarrierId);
}

/** Derive one existing second-phase dW13-A packed-scale alias out of line. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kPoolBlockRows, bool kResidual>
CUTLASS_DEVICE __noinline__ uint32_t*
k3_mxfp8_exact_epilogue_dw13a_packed_scale_alias(
        const void* scale_arena_source,
        uint32_t scale_capacity_rows) {
    const auto dw2_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden,
        scale_capacity_rows, kPoolBlockRows);
    const auto dw13_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden,
        scale_capacity_rows, kPoolBlockRows);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    auto* const packed_base = scale_arena +
        k3_mxfp8_wgrad_next_scale_phase_offset(dw2_layout) +
        dw13_layout.raw_bytes;
    return reinterpret_cast<uint32_t*>(
        packed_base + (kResidual
            ? dw13_layout.packed_a_residual
            : dw13_layout.packed_a_primary));
}

/** Derive one existing second-phase dW13-B packed-scale alias out of line.
 *
 * The A/B rolling producer writes directly into the packed scale storage
 * already reserved for the grouped dW13 body.  Keeping this arithmetic in an
 * out-of-line device helper avoids extending the parent launch ABI or holding
 * arena pointers in process-global state.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kPoolBlockRows, bool kResidual>
CUTLASS_DEVICE __noinline__ uint32_t*
k3_mxfp8_exact_epilogue_dw13b_packed_scale_alias(
        const void* scale_arena_source,
        uint32_t scale_capacity_rows) {
    const auto dw2_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden,
        scale_capacity_rows, kPoolBlockRows);
    const auto dw13_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden,
        scale_capacity_rows, kPoolBlockRows);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    auto* const packed_base = scale_arena +
        k3_mxfp8_wgrad_next_scale_phase_offset(dw2_layout) +
        dw13_layout.raw_bytes;
    return reinterpret_cast<uint32_t*>(
        packed_base + (kResidual
            ? dw13_layout.packed_b_residual
            : dw13_layout.packed_b_primary));
}

}  // namespace deep_gemm
