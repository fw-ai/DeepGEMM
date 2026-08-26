#pragma once

#include <cstddef>
#include <cstdint>

namespace deep_gemm {

/** Coordinate-only state for one group-32 by feature-128 ring ticket.
 *
 * Pointer values never enter symmetric memory.  The aligned first eight
 * bytes form one atomic publication key: epoch in the low word and sequence
 * in the high word.  A producer first release-stores `{epoch, 0}` to make the
 * ticket unambiguously unpublished, then release-stores `{epoch, sequence}`
 * only after its values and scales are visible.  Exactly `reader_target`
 * output-tile consumers contribute to
 * `reader_arrivals` after their final P01 input read, and the last consumer
 * publishes `retired_sequence` and contributes one slot-retirement credit.
 */
struct alignas(32) K3MxFp8EpiloguePanelTicket {
    uint32_t epoch;
    uint32_t sequence;
    uint32_t retired_sequence;
    uint32_t reader_arrivals;
    uint32_t reader_target;
    uint32_t expert;
    uint32_t production_ordinal;
    uint32_t feature_and_group;
};

/** One generation controller for a physical ring slot. */
struct alignas(32) K3MxFp8EpiloguePanelSlotState {
    uint32_t epoch;
    uint32_t generation;
    uint32_t retired_tickets;
    uint32_t closed_sequence;
    uint32_t physical_pool_block;
    uint32_t terminal_sequence;
    uint32_t valid_rows;
    uint32_t expected_tickets;
};

/** Pointer-free coordinates for one grouped-consumer A panel.
 *
 * The grouped scheduler's compact expert K coordinate is deliberately not
 * stored here: a compact K128 tile may straddle two physical 192-row pool
 * blocks.  The consumer resolves it into group-32 coordinates first, then
 * acquires one ticket per coordinate before issuing that group's value/scale
 * reads.  `reader_panel` is the dW13 output-N panel (0..27), i.e. the unique
 * identity contributing one of the ticket's 28 retirement arrivals.
 */
struct K3MxFp8EpiloguePanelConsumerCoordinate {
    uint32_t expert;
    uint32_t feature_panel;
    uint32_t reader_panel;
    uint32_t production_ordinal;
    uint32_t physical_pool_block;
    uint32_t group_in_block;
    uint32_t slot;
    uint32_t sequence;
    uint32_t slot_row_begin;
    uint32_t packed_scale_row;
    uint32_t packed_scale_byte;
};

static_assert(sizeof(K3MxFp8EpiloguePanelTicket) == 32u);
static_assert(alignof(K3MxFp8EpiloguePanelTicket) == 32u);
static_assert(offsetof(K3MxFp8EpiloguePanelTicket, epoch) == 0u);
static_assert(offsetof(K3MxFp8EpiloguePanelTicket, sequence) == 4u);
static_assert(sizeof(K3MxFp8EpiloguePanelSlotState) == 32u);
static_assert(alignof(K3MxFp8EpiloguePanelSlotState) == 32u);
static_assert(offsetof(K3MxFp8EpiloguePanelSlotState, epoch) == 0u);
static_assert(offsetof(K3MxFp8EpiloguePanelSlotState, generation) == 4u);

constexpr uint32_t kK3MxFp8EpilogueRowsPerGroup = 32u;
constexpr uint32_t kK3MxFp8EpilogueFeaturePanel = 128u;
constexpr uint32_t kK3MxFp8EpilogueGroupsPerScaleWord = 4u;
constexpr uint32_t kK3MxFp8LiveGradYPlane = 0u;
constexpr uint32_t kK3MxFp8LiveExactXPlane = 1u;
constexpr uint32_t kK3MxFp8EpilogueScratchPrimaryPlane = 2u;
constexpr uint32_t kK3MxFp8EpilogueScratchResidualPlane = 3u;
constexpr uint32_t kK3MxFp8DW13BScratchPrimaryPlane = 4u;
constexpr uint32_t kK3MxFp8DW13BScratchResidualPlane = 5u;
constexpr uint32_t kK3MxFp8RingNumTopK = 16u;
constexpr uint32_t kK3MxFp8RingReservedRows = 192u;
constexpr uint32_t kK3MxFp8DW13AReaderTarget = 3584u / 128u;
// dW13-B is multicast once to each 256-M cluster task.  The two 128-M
// accumulator panels are not independent B readers.
constexpr uint32_t kK3MxFp8DW13BReaderTarget = 6144u / 256u;
static_assert(kK3MxFp8DW13AReaderTarget == 28u);
static_assert(kK3MxFp8DW13BReaderTarget == 24u);
static_assert(
    kK3MxFp8LiveGradYPlane < kK3MxFp8LiveExactXPlane &&
    kK3MxFp8LiveExactXPlane < kK3MxFp8EpilogueScratchPrimaryPlane &&
    kK3MxFp8EpilogueScratchPrimaryPlane <
        kK3MxFp8EpilogueScratchResidualPlane &&
    kK3MxFp8EpilogueScratchResidualPlane <
        kK3MxFp8DW13BScratchPrimaryPlane &&
    kK3MxFp8DW13BScratchPrimaryPlane <
        kK3MxFp8DW13BScratchResidualPlane &&
    kK3MxFp8DW13BScratchResidualPlane < kK3MxFp8RingNumTopK);

/** Allocation-free physical layout of the K3 dW13-A panel ring.
 *
 * Combine planes zero and one remain live grad-y and exact-X inputs during
 * SiTU.  The ring therefore uses otherwise-idle fixed-top-k planes two and
 * three.  Each BF16 H row is 7168 bytes.  Its first 6144 bytes hold one FP8
 * 2I value row; the 1024-byte tail holds 32 coordinate tickets.  A 192-row
 * slot needs 48 feature panels by six group-32 panels = 288 tickets (nine
 * tails) and one slot state (the tenth tail).  No allocation or saved tensor
 * is added.
 */
template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows>
struct K3MxFp8EpiloguePanelRingLayout {
    static constexpr uint32_t kRowBytes =
        kHidden * sizeof(uint16_t);
    static constexpr uint32_t kValueBytesPerRow = kLogicalWidth;
    static constexpr uint32_t kTailBytesPerRow =
        kRowBytes - kValueBytesPerRow;
    static constexpr uint32_t kFeaturePanels =
        kLogicalWidth / kK3MxFp8EpilogueFeaturePanel;
    static constexpr uint32_t kGroupsPerBlock =
        kPoolBlockRows / kK3MxFp8EpilogueRowsPerGroup;
    static constexpr uint32_t kTicketsPerSlot =
        kFeaturePanels * kGroupsPerBlock;
    static constexpr uint32_t kTicketsPerTailRow =
        kTailBytesPerRow / sizeof(K3MxFp8EpiloguePanelTicket);
    static constexpr uint32_t kTicketTailRows =
        (kTicketsPerSlot + kTicketsPerTailRow - 1u) /
        kTicketsPerTailRow;
    static constexpr uint32_t kSlotStateTailRow = kTicketTailRows;
    static constexpr uint32_t kMetadataTailRows =
        kSlotStateTailRow + 1u;
    static constexpr uint32_t kValueBytesPerGroup =
        kK3MxFp8EpilogueRowsPerGroup *
        kK3MxFp8EpilogueFeaturePanel;
    static constexpr uint32_t kScaleBytesPerPackedRow =
        kK3MxFp8EpilogueFeaturePanel * sizeof(uint32_t);

    static_assert(kLogicalWidth % kK3MxFp8EpilogueFeaturePanel == 0u);
    static_assert(kPoolBlockRows % kK3MxFp8EpilogueRowsPerGroup == 0u);
    static_assert(kRowBytes > kValueBytesPerRow);
    static_assert(
        kTailBytesPerRow % sizeof(K3MxFp8EpiloguePanelTicket) == 0u);
    static_assert(kMetadataTailRows <= kPoolBlockRows);

    static constexpr uint32_t ring_depth(uint32_t capacity_rows) {
        return capacity_rows <= kK3MxFp8RingReservedRows
            ? 0u
            : (capacity_rows - kK3MxFp8RingReservedRows) /
                kPoolBlockRows;
    }

    static constexpr uint32_t slot_for_ordinal(
            uint32_t production_ordinal, uint32_t depth) {
        return production_ordinal % depth;
    }

    static constexpr uint32_t sequence_for_ordinal(
            uint32_t production_ordinal, uint32_t depth) {
        return production_ordinal / depth + 1u;
    }

    static constexpr uint32_t terminal_sequence_for_slot(
            uint32_t total_pool_blocks, uint32_t slot,
            uint32_t depth) {
        return slot >= total_pool_blocks
            ? 0u
            : (total_pool_blocks - 1u - slot) / depth + 1u;
    }

    static constexpr uint32_t ticket_index(
            uint32_t feature_panel, uint32_t group_in_block) {
        return feature_panel * kGroupsPerBlock + group_in_block;
    }

    static constexpr uint32_t feature_and_group(
            uint32_t feature_panel, uint32_t group_in_block) {
        return (feature_panel << 8u) | (group_in_block & 0xffu);
    }

    static constexpr uint32_t ticket_tail_row(uint32_t ticket) {
        return ticket / kTicketsPerTailRow;
    }

    static constexpr uint32_t ticket_tail_byte(uint32_t ticket) {
        return (ticket % kTicketsPerTailRow) *
            sizeof(K3MxFp8EpiloguePanelTicket);
    }

    static constexpr K3MxFp8EpiloguePanelConsumerCoordinate
    consumer_coordinate(
            uint32_t expert, uint32_t feature_panel,
            uint32_t reader_panel, uint32_t production_ordinal,
            uint32_t physical_pool_row, uint32_t packed_scale_row,
            uint32_t packed_scale_byte, uint32_t depth) {
        const uint32_t physical_pool_block =
            physical_pool_row / kPoolBlockRows;
        const uint32_t row_in_block =
            physical_pool_row % kPoolBlockRows;
        const uint32_t group_in_block =
            row_in_block / kK3MxFp8EpilogueRowsPerGroup;
        const uint32_t slot = slot_for_ordinal(
            production_ordinal, depth);
        return {
            expert,
            feature_panel,
            reader_panel,
            production_ordinal,
            physical_pool_block,
            group_in_block,
            slot,
            sequence_for_ordinal(production_ordinal, depth),
            slot * kPoolBlockRows +
                group_in_block * kK3MxFp8EpilogueRowsPerGroup,
            packed_scale_row,
            packed_scale_byte,
        };
    }
};

using K3MxFp8ExactDW13AEpilogueRingLayout =
    K3MxFp8EpiloguePanelRingLayout<3584u, 6144u, 192u>;
using K3MxFp8ExactDW13BEpilogueRingLayout =
    K3MxFp8EpiloguePanelRingLayout<3584u, 3584u, 192u>;

static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kRowBytes == 7168u);
static_assert(
    K3MxFp8ExactDW13AEpilogueRingLayout::kValueBytesPerRow == 6144u);
static_assert(
    K3MxFp8ExactDW13AEpilogueRingLayout::kTailBytesPerRow == 1024u);
static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kFeaturePanels == 48u);
static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kGroupsPerBlock == 6u);
static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kTicketsPerSlot == 288u);
static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kTicketTailRows == 9u);
static_assert(K3MxFp8ExactDW13AEpilogueRingLayout::kMetadataTailRows == 10u);
constexpr auto kK3MxFp8ExactDW13AConsumerCoordinateProof =
    K3MxFp8ExactDW13AEpilogueRingLayout::consumer_coordinate(
        3u, 17u, 9u, 5u, 71u * 192u + 4u * 32u,
        41u, 2u, 2u);
static_assert(
    kK3MxFp8ExactDW13AConsumerCoordinateProof.production_ordinal == 5u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.physical_pool_block == 71u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.group_in_block == 4u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.slot == 1u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.sequence == 3u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.slot_row_begin == 320u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.packed_scale_row == 41u &&
    kK3MxFp8ExactDW13AConsumerCoordinateProof.packed_scale_byte == 2u);

static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kRowBytes == 7168u);
static_assert(
    K3MxFp8ExactDW13BEpilogueRingLayout::kValueBytesPerRow == 3584u);
static_assert(
    K3MxFp8ExactDW13BEpilogueRingLayout::kTailBytesPerRow == 3584u);
static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kFeaturePanels == 28u);
static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kGroupsPerBlock == 6u);
static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kTicketsPerSlot == 168u);
static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kTicketTailRows == 2u);
static_assert(K3MxFp8ExactDW13BEpilogueRingLayout::kMetadataTailRows == 3u);

/** Host-side, pointer-free capacity proof shared with source contracts. */
struct K3MxFp8EpiloguePanelRingHostProof {
    uint64_t plane_bytes;
    uint64_t ring_bytes;
    uint64_t combine_bytes;
    uint64_t primary_plane_offset;
    uint64_t residual_plane_offset;
    uint32_t capacity_rows;
    uint32_t ring_row_base;
    uint32_t usable_rows;
    uint32_t ring_depth;
    uint32_t num_topk;
    bool capacity_ok;
    bool planes_disjoint;
    bool metadata_fits;
};

template <uint32_t kHidden, uint32_t kLogicalWidth,
          uint32_t kPoolBlockRows,
          uint32_t kPrimaryPlane = kK3MxFp8EpilogueScratchPrimaryPlane,
          uint32_t kResidualPlane = kK3MxFp8EpilogueScratchResidualPlane>
constexpr K3MxFp8EpiloguePanelRingHostProof
k3_mxfp8_epilogue_panel_ring_host_proof(
        uint32_t capacity_rows, uint32_t num_topk) {
    using Layout = K3MxFp8EpiloguePanelRingLayout<
        kHidden, kLogicalWidth, kPoolBlockRows>;
    const uint64_t plane_bytes =
        static_cast<uint64_t>(capacity_rows) * Layout::kRowBytes;
    const uint64_t row_base_bytes =
        static_cast<uint64_t>(kK3MxFp8RingReservedRows) *
        Layout::kRowBytes;
    const uint64_t primary_offset =
        kPrimaryPlane * plane_bytes + row_base_bytes;
    const uint64_t residual_offset =
        kResidualPlane * plane_bytes + row_base_bytes;
    const uint64_t combine_bytes =
        static_cast<uint64_t>(num_topk) * plane_bytes;
    const uint32_t depth = Layout::ring_depth(capacity_rows);
    const uint64_t ring_bytes =
        static_cast<uint64_t>(depth) * kPoolBlockRows *
        Layout::kRowBytes;
    return {
        plane_bytes,
        ring_bytes,
        combine_bytes,
        primary_offset,
        residual_offset,
        capacity_rows,
        kK3MxFp8RingReservedRows,
        capacity_rows > kK3MxFp8RingReservedRows
            ? capacity_rows - kK3MxFp8RingReservedRows : 0u,
        depth,
        num_topk,
        capacity_rows >= kK3MxFp8RingReservedRows + kPoolBlockRows &&
            depth != 0u &&
            num_topk > kResidualPlane,
        primary_offset + ring_bytes <= residual_offset &&
            residual_offset + ring_bytes <= combine_bytes,
        Layout::kMetadataTailRows <= kPoolBlockRows,
    };
}

/** Same byte offset on every symmetric base is the mapping invariant. */
constexpr uint64_t k3_mxfp8_epilogue_ring_mapped_address(
        uint64_t peer_base, uint64_t combine_offset,
        uint64_t plane_offset) {
    return peer_base + combine_offset + plane_offset;
}

}  // namespace deep_gemm
