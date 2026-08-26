#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace deep_gemm {

// Kimi K3 needs at most three backward ranges at the 64-K-token/rank launch
// boundary (the true-variable-length EP=8 maximum is 87,382 local tokens).
// Three compact 32-byte descriptors preserve the former 112-byte RangeSet
// ABI, so the auxiliary TensorMap slots do not grow.
constexpr uint32_t kK3MaxBackwardRanges = 3;

// The exact K3 MXFP8 wgrad path consumes two independently quantized operand
// planes per logical GEMM operand.  Keeping all address-specialized TensorMaps
// in one grid-constant pack avoids both device-side descriptor mutation and a
// per-thread local-memory frame when the fused parent calls outlined UMMA/TMA
// phases.  The template keeps the host CUtensorMap and device TmaDescriptor
// representations ABI-identical without coupling this lightweight header to
// either CUDA driver or CUTE declarations.
constexpr uint32_t kK3MxFp8WgradMapsPerPhase = 8u;
constexpr uint32_t kK3MxFp8WgradNumConsumerTensorMaps =
    2u * kK3MxFp8WgradMapsPerPhase;
constexpr uint32_t kK3MxFp8WgradNumProducerTensorMaps = 12u;
constexpr uint32_t kK3MxFp8WgradNumTensorMaps =
    kK3MxFp8WgradNumConsumerTensorMaps +
    kK3MxFp8WgradNumProducerTensorMaps;
// Four streaming engines occupy the tail of the compact fused wgrad body.
// Keep this host/device-visible launch requirement next to the TensorMap ABI
// so the JIT cannot silently launch the exact specialization with only the
// smaller serial-wgrad arena.
constexpr uint32_t kK3MxFp8WgradStreamingSmemBytes = 224000u;
// The exact-body BF16 epilogue and its host-built D TensorMaps must share one
// row-slice height. Two 32-row stages halve TMEM/TMA epilogue iterations while
// retaining the proven double-buffered store pipeline.
constexpr uint32_t kK3MxFp8WgradStoreBlockM = 32u;
constexpr uint32_t kK3MxFp8WgradNumTmaStoreStages = 2u;
constexpr uint32_t kK3MxFp8DW13QuantFeatureTile = 128u;
constexpr uint32_t kK3MxFp8DW2QuantFeatureTile = 64u;
constexpr uint32_t kK3MxFp8DW13QuantRowsPerGroup = 32u;

enum K3MxFp8WgradTensorMapSlot : uint32_t {
    kK3MxFp8DW2ValueAPrimaryMap = 0u,
    kK3MxFp8DW2ValueAResidualMap = 1u,
    kK3MxFp8DW2ValueBPrimaryMap = 2u,
    kK3MxFp8DW2ValueBResidualMap = 3u,
    kK3MxFp8DW2ScaleAPrimaryMap = 4u,
    kK3MxFp8DW2ScaleAResidualMap = 5u,
    kK3MxFp8DW2ScaleBPrimaryMap = 6u,
    kK3MxFp8DW2ScaleBResidualMap = 7u,
    kK3MxFp8DW13ValueAPrimaryMap = 8u,
    kK3MxFp8DW13ValueAResidualMap = 9u,
    kK3MxFp8DW13ValueBPrimaryMap = 10u,
    kK3MxFp8DW13ValueBResidualMap = 11u,
    kK3MxFp8DW13ScaleAPrimaryMap = 12u,
    kK3MxFp8DW13ScaleAResidualMap = 13u,
    kK3MxFp8DW13ScaleBPrimaryMap = 14u,
    kK3MxFp8DW13ScaleBResidualMap = 15u,
    // The streaming dW13 quantizer uses compact 32x128, no-swizzle TensorMaps.
    // Consumer maps above remain 128x128-swizzled for the UMMA mainloop.
    kK3MxFp8DW13ProducerSourceAMap = 16u,
    kK3MxFp8DW13ProducerSourceBMap = 17u,
    kK3MxFp8DW13ProducerValueAPrimaryMap = 18u,
    kK3MxFp8DW13ProducerValueAResidualMap = 19u,
    kK3MxFp8DW13ProducerValueBPrimaryMap = 20u,
    kK3MxFp8DW13ProducerValueBResidualMap = 21u,
    // The elastic dW2 producer borrows one W13 warpgroup and therefore uses
    // compact 32x64, no-swizzle TensorMaps.  One source load plus two value
    // stores replaces the former thirty-two row loads and sixty-four row
    // stores for each exact group-32 conversion.
    kK3MxFp8DW2ProducerSourceAMap = 22u,
    kK3MxFp8DW2ProducerSourceBMap = 23u,
    kK3MxFp8DW2ProducerValueAPrimaryMap = 24u,
    kK3MxFp8DW2ProducerValueAResidualMap = 25u,
    kK3MxFp8DW2ProducerValueBPrimaryMap = 26u,
    kK3MxFp8DW2ProducerValueBResidualMap = 27u,
    // Ring mode removes the streaming dW13-A value producer. Rebind its two
    // now-dead immutable value-store slots to the physical BF16-H row pitch
    // of symmetric combine planes 2/3. The selected grouped consumer sees
    // the ring maps only after the producer's complete lifetime, while the
    // ordinary path retains the original compact store descriptors. This
    // specialization-time alias keeps the grid-constant pack at 28 maps.
    kK3MxFp8DW13RingValueAPrimaryMap =
        kK3MxFp8DW13ProducerValueAPrimaryMap,
    kK3MxFp8DW13RingValueAResidualMap =
        kK3MxFp8DW13ProducerValueAResidualMap,
    // The allocation-free B producer gathers source X through raw-pointer TMA
    // and writes planes 4/5 directly. Its compact value-store descriptors are
    // therefore dead in the selected specialization and can name the B ring
    // without increasing the 28-map grid-constant pack.
    kK3MxFp8DW13RingValueBPrimaryMap =
        kK3MxFp8DW13ProducerValueBPrimaryMap,
    kK3MxFp8DW13RingValueBResidualMap =
        kK3MxFp8DW13ProducerValueBResidualMap,
};

template <typename TensorMap>
struct alignas(128) K3MxFp8WgradTensorMapPack {
    TensorMap maps[kK3MxFp8WgradNumTensorMaps];
};

template <typename TensorMap>
constexpr bool k3_mxfp8_wgrad_tensor_map_pack_abi() {
    using Pack = K3MxFp8WgradTensorMapPack<TensorMap>;
    return sizeof(TensorMap) == 128u &&
        sizeof(Pack) == kK3MxFp8WgradNumTensorMaps * 128u &&
        alignof(Pack) == 128u && offsetof(Pack, maps) == 0u &&
        std::is_standard_layout_v<Pack> &&
        std::is_trivially_copyable_v<Pack> &&
        kK3MxFp8DW13ValueAPrimaryMap ==
            kK3MxFp8WgradMapsPerPhase;
}

#if defined(__CUDACC__)
#define DG_K3_MXFP8_HOST_DEVICE __host__ __device__
#else
#define DG_K3_MXFP8_HOST_DEVICE
#endif

/** Byte layout for the raw and packed scales of two MXFP8 operands.
 *
 * This layout is consumed both by the host TensorMap builder and by the fused
 * device producer. Keeping the arithmetic in this lightweight ABI header
 * prevents the two sides from binding different offsets after a JIT rebuild.
 */
struct K3MxFp8WgradScaleArenaLayout {
    uint32_t raw_row_capacity;
    uint32_t packed_row_capacity;
    uint64_t raw_a_primary;
    uint64_t raw_a_residual;
    uint64_t raw_b_primary;
    uint64_t raw_b_residual;
    uint64_t raw_bytes;
    uint64_t packed_a_primary;
    uint64_t packed_a_residual;
    uint64_t packed_b_primary;
    uint64_t packed_b_residual;
    uint64_t packed_bytes;
};

DG_K3_MXFP8_HOST_DEVICE constexpr uint32_t
k3_mxfp8_wgrad_ceil_div(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

/** Number of temporary group-scale rows emitted by 128-row value tiles. */
DG_K3_MXFP8_HOST_DEVICE constexpr uint32_t
k3_mxfp8_wgrad_raw_scale_rows(uint32_t k_capacity) {
    return 4u * k3_mxfp8_wgrad_ceil_div(k_capacity, 128u);
}

DG_K3_MXFP8_HOST_DEVICE constexpr uint64_t
k3_mxfp8_wgrad_raw_scale_bytes(
        uint32_t feature, uint32_t k_capacity) {
    return static_cast<uint64_t>(feature) *
        k3_mxfp8_wgrad_raw_scale_rows(k_capacity);
}

DG_K3_MXFP8_HOST_DEVICE constexpr uint64_t
k3_mxfp8_wgrad_packed_scale_bytes(
        uint32_t feature, uint32_t scale_row_capacity) {
    return static_cast<uint64_t>(feature) * scale_row_capacity *
        sizeof(uint32_t);
}

DG_K3_MXFP8_HOST_DEVICE constexpr uint32_t
k3_mxfp8_wgrad_scale_capacity(
        uint32_t k_capacity, uint32_t pool_block_rows) {
    return k3_mxfp8_wgrad_ceil_div(k_capacity, pool_block_rows) *
        k3_mxfp8_wgrad_ceil_div(pool_block_rows, 128u);
}

DG_K3_MXFP8_HOST_DEVICE constexpr K3MxFp8WgradScaleArenaLayout
k3_mxfp8_wgrad_scale_arena_layout(
        uint32_t feature_a, uint32_t feature_b,
        uint32_t k_capacity, uint32_t pool_block_rows) {
    const uint32_t raw_rows =
        k3_mxfp8_wgrad_raw_scale_rows(k_capacity);
    const uint32_t packed_rows =
        k3_mxfp8_wgrad_scale_capacity(k_capacity, pool_block_rows);
    const uint64_t raw_a_bytes =
        static_cast<uint64_t>(raw_rows) * feature_a;
    const uint64_t raw_b_bytes =
        static_cast<uint64_t>(raw_rows) * feature_b;
    const uint64_t packed_a_bytes =
        static_cast<uint64_t>(packed_rows) * feature_a * sizeof(uint32_t);
    const uint64_t packed_b_bytes =
        static_cast<uint64_t>(packed_rows) * feature_b * sizeof(uint32_t);
    return {
        raw_rows,
        packed_rows,
        0u,
        raw_a_bytes,
        2u * raw_a_bytes,
        2u * raw_a_bytes + raw_b_bytes,
        2u * (raw_a_bytes + raw_b_bytes),
        0u,
        packed_a_bytes,
        2u * packed_a_bytes,
        2u * packed_a_bytes + packed_b_bytes,
        2u * (packed_a_bytes + packed_b_bytes),
    };
}

DG_K3_MXFP8_HOST_DEVICE constexpr uint64_t
k3_mxfp8_wgrad_next_scale_phase_offset(
        const K3MxFp8WgradScaleArenaLayout& first_phase) {
    const uint64_t value = first_phase.raw_bytes + first_phase.packed_bytes;
    return (value + 127u) / 128u * 128u;
}

DG_K3_MXFP8_HOST_DEVICE constexpr uint64_t
k3_mxfp8_wgrad_two_phase_scale_bytes(
        const K3MxFp8WgradScaleArenaLayout& first_phase,
        const K3MxFp8WgradScaleArenaLayout& second_phase) {
    return k3_mxfp8_wgrad_next_scale_phase_offset(first_phase) +
        second_phase.raw_bytes + second_phase.packed_bytes;
}

/** Returns whether two half-open physical-row intervals alias. */
DG_K3_MXFP8_HOST_DEVICE constexpr bool
k3_mxfp8_wgrad_row_intervals_overlap(
        uint32_t lhs_begin, uint32_t lhs_end,
        uint32_t rhs_begin, uint32_t rhs_end) {
    return lhs_begin < lhs_end && rhs_begin < rhs_end &&
        lhs_begin < rhs_end && rhs_begin < lhs_end;
}

static_assert(k3_mxfp8_wgrad_raw_scale_rows(0u) == 0u);
static_assert(k3_mxfp8_wgrad_raw_scale_rows(128u) == 4u);
static_assert(k3_mxfp8_wgrad_raw_scale_rows(192u) == 8u);
static_assert(k3_mxfp8_wgrad_raw_scale_rows(384u) == 12u);
static_assert(k3_mxfp8_wgrad_row_intervals_overlap(0u, 192u, 191u, 384u));
static_assert(!k3_mxfp8_wgrad_row_intervals_overlap(0u, 192u, 192u, 384u));
static_assert(!k3_mxfp8_wgrad_row_intervals_overlap(64u, 64u, 0u, 128u));
static_assert(!k3_mxfp8_wgrad_row_intervals_overlap(0u, 128u, 64u, 64u));

#undef DG_K3_MXFP8_HOST_DEVICE

#if defined(__CUDACC__)
#define DG_K3_MULTIRANGE_HOST_DEVICE __host__ __device__
#else
#define DG_K3_MULTIRANGE_HOST_DEVICE
#endif

// Offsets are expressed in elements (or rows), never bytes. Token, expert-
// count, metadata, and epoch offsets are derived from the range index. Token
// offsets advance by rank-uniform capacity rather than active count, so every
// rank maps a remote token to the same symmetric-plane row for variable length.
struct alignas(16) K3BackwardRangeDesc {
    uint32_t num_tokens = 0;
    uint32_t max_tokens_per_rank = 0;
    uint32_t pool_row_begin = 0;
    uint32_t num_pool_rows = 0;
    uint32_t acts_row_begin = 0;
    uint32_t num_acts_rows = 0;
    uint32_t sf_pool_row_begin = 0;
    uint32_t num_sf_pool_rows = 0;
};

struct alignas(16) K3BackwardRangeSet {
    uint32_t num_ranges = 0;
    uint32_t total_backward_tokens = 0;
    uint32_t total_pool_rows = 0;
    uint32_t total_acts_rows = 0;
    K3BackwardRangeDesc ranges[kK3MaxBackwardRanges]{};

    DG_K3_MULTIRANGE_HOST_DEVICE static constexpr K3BackwardRangeSet
    single_range(
        const uint32_t num_tokens,
        const uint32_t max_tokens_per_rank,
        const uint32_t num_pool_rows,
        const uint32_t num_acts_rows,
        const uint32_t num_sf_pool_rows) {
        K3BackwardRangeSet result{};
        result.num_ranges = 1;
        result.total_backward_tokens = num_tokens;
        result.total_pool_rows = num_pool_rows;
        result.total_acts_rows = num_acts_rows;
        result.ranges[0] = {
            num_tokens, max_tokens_per_rank,
            0u, num_pool_rows,
            0u, num_acts_rows,
            0u, num_sf_pool_rows};
        return result;
    }

    DG_K3_MULTIRANGE_HOST_DEVICE constexpr uint32_t token_begin(
        const uint32_t range_idx) const {
        uint32_t result = 0u;
        for (uint32_t idx = 0u; idx < range_idx; ++idx)
            result += ranges[idx].max_tokens_per_rank;
        return result;
    }

    DG_K3_MULTIRANGE_HOST_DEVICE static constexpr uint32_t
    expert_counts_begin(
        const uint32_t range_idx, const uint32_t num_experts) {
        return range_idx * num_experts;
    }

    DG_K3_MULTIRANGE_HOST_DEVICE constexpr uint32_t metadata_row_begin(
        const uint32_t range_idx) const {
        return ranges[range_idx].pool_row_begin;
    }

    DG_K3_MULTIRANGE_HOST_DEVICE static constexpr uint32_t epoch_seed(
        const uint32_t range_idx) {
        return range_idx + 1u;
    }

    DG_K3_MULTIRANGE_HOST_DEVICE constexpr bool is_packed(
        const uint32_t num_experts) const {
        if (num_ranges == 0 || num_ranges > kK3MaxBackwardRanges)
            return false;

        (void)num_experts;
        uint32_t token_count = 0;
        uint32_t pool_end = 0;
        uint32_t acts_end = 0;
        uint32_t sf_pool_end = 0;
        for (uint32_t range_idx = 0; range_idx < num_ranges; ++range_idx) {
            const auto& range = ranges[range_idx];
            if (range.pool_row_begin != pool_end ||
                range.acts_row_begin != acts_end ||
                range.sf_pool_row_begin != sf_pool_end ||
                range.num_tokens > range.max_tokens_per_rank ||
                range.num_acts_rows > range.num_pool_rows) {
                return false;
            }
            token_count += range.num_tokens;
            pool_end += range.num_pool_rows;
            acts_end += range.num_acts_rows;
            sf_pool_end += range.num_sf_pool_rows;
        }
        return token_count == total_backward_tokens &&
            pool_end == total_pool_rows &&
            acts_end == total_acts_rows;
    }

    template <uint32_t kNumExperts>
    DG_K3_MULTIRANGE_HOST_DEVICE constexpr bool is_packed() const {
        return is_packed(kNumExperts);
    }

    DG_K3_MULTIRANGE_HOST_DEVICE constexpr uint32_t reverse_range_index(
        const uint32_t iteration) const {
        return num_ranges - 1u - iteration;
    }

    // The kernel binds every operand to one full-arena TMA map, so activation
    // and metadata rows use the same physical coordinates. Later variants may
    // relax this if they carry independent descriptor bases through W13.
    template <uint32_t kNumExperts, uint32_t kBlockM>
    DG_K3_MULTIRANGE_HOST_DEVICE constexpr bool
    is_full_arena_compatible() const {
        if (!is_packed<kNumExperts>() || total_pool_rows != total_acts_rows ||
            total_pool_rows % kBlockM != 0) {
            return false;
        }
        for (uint32_t range_idx = 0; range_idx < num_ranges; ++range_idx) {
            const auto& range = ranges[range_idx];
            if (range.pool_row_begin != range.acts_row_begin ||
                range.num_pool_rows != range.num_acts_rows ||
                range.pool_row_begin % kBlockM != 0 ||
                range.num_pool_rows % kBlockM != 0) {
                return false;
            }
        }
        return true;
    }
};

// Host-facing sizes for one physical range. The pure constexpr packer keeps
// offset derivation shared by the JIT wrapper and unit tests without exposing
// a device allocation or a process-global descriptor table.
struct K3BackwardRangeShape {
    uint32_t num_tokens = 0;
    uint32_t max_tokens_per_rank = 0;
    uint32_t num_pool_rows = 0;
    uint32_t num_acts_rows = 0;
    uint32_t num_sf_pool_rows = 0;
};

DG_K3_MULTIRANGE_HOST_DEVICE constexpr K3BackwardRangeSet
pack_k3_backward_ranges(
    const K3BackwardRangeShape* shapes,
    const uint32_t num_ranges,
    const uint32_t num_experts) {
    K3BackwardRangeSet result{};
    if (shapes == nullptr || num_ranges == 0u ||
        num_ranges > kK3MaxBackwardRanges) {
        return result;
    }
    (void)num_experts;
    result.num_ranges = num_ranges;
    uint32_t pool_row_begin = 0u;
    uint32_t acts_row_begin = 0u;
    uint32_t sf_pool_row_begin = 0u;
    for (uint32_t range_idx = 0u;
         range_idx < num_ranges; ++range_idx) {
        const auto& shape = shapes[range_idx];
        result.ranges[range_idx] = {
            shape.num_tokens,
            shape.max_tokens_per_rank,
            pool_row_begin,
            shape.num_pool_rows,
            acts_row_begin,
            shape.num_acts_rows,
            sf_pool_row_begin,
            shape.num_sf_pool_rows};
        result.total_backward_tokens += shape.num_tokens;
        result.total_pool_rows += shape.num_pool_rows;
        result.total_acts_rows += shape.num_acts_rows;
        pool_row_begin += shape.num_pool_rows;
        acts_row_begin += shape.num_acts_rows;
        sf_pool_row_begin += shape.num_sf_pool_rows;
    }
    return result;
}

static_assert(std::is_standard_layout_v<K3BackwardRangeDesc>);
static_assert(std::is_trivially_copyable_v<K3BackwardRangeDesc>);
static_assert(sizeof(K3BackwardRangeDesc) == 32);
static_assert(alignof(K3BackwardRangeDesc) == 16);
static_assert(std::is_standard_layout_v<K3BackwardRangeSet>);
static_assert(std::is_trivially_copyable_v<K3BackwardRangeSet>);
static_assert(sizeof(K3BackwardRangeSet) == 112);
static_assert(alignof(K3BackwardRangeSet) == 16);

/** Compact exact-suffix arguments stored in one legacy TensorMap slot.
 *
 * Thirteen pointers followed by six 32-bit scalars consume exactly 128 bytes,
 * so exact K3 training can reuse one existing descriptor parameter without
 * growing the kernel ABI. The expected epoch is deliberately absent here: it
 * is copied from each CTA's local parent state into the shared handoff so the
 * suffix can wait for CTA zero's release publication without an ABI field.
 */
template <typename BFloat16, typename FloatE4M3>
struct alignas(8) K3MxFp8WgradExactArgs {
    const int* expert_counts;
    BFloat16* grad_ye_output;
    BFloat16* h_weighted_output;
    BFloat16* grad_gate_up_output;
    BFloat16* x_pool_output;
    BFloat16* grad_y_unweighted_output;
    const BFloat16* down_unweighted_output;
    const FloatE4M3* scale_arena_source;
    uint32_t* state;
    BFloat16* w2_output;
    BFloat16* w13_output;
    BFloat16* backward_grad_x_output;
    const BFloat16* backward_grad_y;
    uint32_t k_capacity;
    uint32_t num_backward_tokens;
    uint32_t first_range_tokens;
    uint32_t second_range_begin;
    uint32_t num_topk;
    uint32_t clear_empty_outputs;
};

/** ABI-preserving interpretation of one legacy 128-byte TensorMap argument.
 *
 * Legacy kernels use `legacy_map`. Exact K3 overlap uses the first four of the
 * six existing slots as dW2 D, dW13 D, range-set, and compact-args payloads.
 * Keeping six separate slot parameters preserves their byte positions and
 * avoids the compiler regressions caused by a single 3584-byte aggregate.
 */
template <typename TensorMap, typename BFloat16, typename FloatE4M3>
union alignas(128) K3MxFp8WgradAuxSlot {
    TensorMap legacy_map;
    TensorMap exact_output_map;
    K3BackwardRangeSet exact_ranges;
    K3MxFp8WgradExactArgs<BFloat16, FloatE4M3> exact_args;
};

/** CTA-local handoff for the no-argument exact-wgrad suffix boundary.
 *
 * The parent publishes addresses of the existing exact TensorMap pack,
 * symmetric-buffer/workspace views, and four repurposed 128-byte auxiliary
 * slots in otherwise-unused shared memory. The noinline suffix reloads those
 * addresses after the parent pipeline has retired, avoiding a per-thread
 * outgoing call frame without adding a kernel parameter or global allocation.
 */
template <typename TensorMapPack, typename SymBuffer, typename Workspace,
          typename AuxSlot>
struct alignas(8) K3MxFp8WgradSuffixHandoff {
    const TensorMapPack* tensor_map_pack;
    const SymBuffer* backward_sym_buffer;
    const Workspace* backward_workspace;
    const AuxSlot* dw2_output_slot;
    const AuxSlot* dw13_output_slot;
    const AuxSlot* ranges_slot;
    const AuxSlot* args_slot;
    uint32_t expected_launch_epoch;
    uint32_t reserved;
};

template <typename TensorMapPack, typename SymBuffer, typename Workspace,
          typename AuxSlot>
constexpr bool k3_mxfp8_wgrad_suffix_handoff_abi() {
    using Handoff = K3MxFp8WgradSuffixHandoff<
        TensorMapPack, SymBuffer, Workspace, AuxSlot>;
    return sizeof(Handoff) == 64u && alignof(Handoff) == 8u &&
        offsetof(Handoff, tensor_map_pack) == 0u &&
        offsetof(Handoff, backward_sym_buffer) == 8u &&
        offsetof(Handoff, backward_workspace) == 16u &&
        offsetof(Handoff, dw2_output_slot) == 24u &&
        offsetof(Handoff, dw13_output_slot) == 32u &&
        offsetof(Handoff, ranges_slot) == 40u &&
        offsetof(Handoff, args_slot) == 48u &&
        offsetof(Handoff, expected_launch_epoch) == 56u &&
        offsetof(Handoff, reserved) == 60u &&
        std::is_standard_layout_v<Handoff> &&
        std::is_trivially_copyable_v<Handoff>;
}

template <typename TensorMap, typename BFloat16, typename FloatE4M3>
constexpr bool k3_mxfp8_wgrad_aux_slot_abi() {
    using ExactArgs = K3MxFp8WgradExactArgs<BFloat16, FloatE4M3>;
    using Slot = K3MxFp8WgradAuxSlot<
        TensorMap, BFloat16, FloatE4M3>;
    return sizeof(TensorMap) == 128u && sizeof(ExactArgs) == 128u &&
        alignof(ExactArgs) == 8u && sizeof(Slot) == 128u &&
        alignof(Slot) == 128u &&
        offsetof(ExactArgs, expert_counts) == 0u &&
        offsetof(ExactArgs, backward_grad_y) == 96u &&
        offsetof(ExactArgs, k_capacity) == 104u &&
        offsetof(ExactArgs, clear_empty_outputs) == 124u &&
        std::is_standard_layout_v<ExactArgs> &&
        std::is_trivially_copyable_v<ExactArgs> &&
        std::is_standard_layout_v<Slot> &&
        std::is_trivially_copyable_v<Slot>;
}

static_assert(std::is_standard_layout_v<K3BackwardRangeShape>);
static_assert(std::is_trivially_copyable_v<K3BackwardRangeShape>);
static_assert(sizeof(K3BackwardRangeShape) == 20);

// Map a caller-visible packed active-token row to its physical symmetric
// plane. Active outputs remain contiguous while rank-uniform capacity gaps are
// skipped between ranges.
DG_K3_MULTIRANGE_HOST_DEVICE constexpr uint32_t
k3_multirange_physical_token_index(
    const K3BackwardRangeSet& ranges,
    const uint32_t logical_token_idx) {
    uint32_t logical_begin = 0u;
    for (uint32_t range_idx = 0u;
         range_idx < ranges.num_ranges; ++range_idx) {
        const auto& range = ranges.ranges[range_idx];
        if (logical_token_idx < logical_begin + range.num_tokens) {
            return ranges.token_begin(range_idx) +
                logical_token_idx - logical_begin;
        }
        logical_begin += range.num_tokens;
    }
    return static_cast<uint32_t>(-1);
}

#undef DG_K3_MULTIRANGE_HOST_DEVICE

// Direct dX readiness is indexed by the physical symmetric-plane token, not
// by packed logical order. One counter per token is deliberate: grouping even
// a few source tokens makes their top-k routes wait for the slowest unrelated
// expert block and defeats producer/consumer overlap. The backing dead
// source-index tail already reserves one word per physical token, so this
// granularity changes neither allocation size nor the public workspace ABI.
struct K3DirectGradXReadyContract {
    static constexpr uint32_t kTokensPerCounter = 1u;

    static constexpr uint32_t counter_index(
        const uint32_t physical_token_idx) {
        return physical_token_idx;
    }

    static constexpr uint32_t num_counters(
        const uint32_t physical_token_capacity) {
        return physical_token_capacity;
    }
};

static_assert(K3DirectGradXReadyContract::kTokensPerCounter == 1u);

// Phase values are also the high bits of workspace epochs. Range-local state
// therefore cannot observe a stale ready bit from an earlier phase (ABA).
enum class K3MultiRangeBackwardPhase : uint32_t {
    InitWeights = 1,
    W2Dgrad = 2,
    W2CacheRetired = 3,
    W2Wgrad = 4,
    W13Dgrad = 5,
    GradXPublished = 6,
    SymmetricPlanesReusable = 7,
    W13CacheRetired = 8,
    // Value 9 is reserved. Exact-X publication is row-local and overlaps W2
    // dgrad, so it is modeled by K3W2RangeStorageEvent rather than as a
    // whole-grid phase.
    W13Wgrad = 10,
    Finalize = 11,
};

constexpr uint32_t k3_multirange_epoch(
    const K3MultiRangeBackwardPhase phase,
    const uint32_t range_epoch_seed) {
    return (static_cast<uint32_t>(phase) << 16) |
        (range_epoch_seed & 0xffffu);
}

constexpr bool k3_phase_before(
    const K3MultiRangeBackwardPhase lhs,
    const K3MultiRangeBackwardPhase rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
}

// This contract describes the first implementation body. N=256 is the
// measured-fast two-CTA/two-accumulator implementation. Paired-N is excluded
// until profiling demonstrates that its additional spill traffic is gone.
struct K3MultiRangeWgradBodyContract {
    static constexpr uint32_t kLogicalBlockN = 256;
    static constexpr uint32_t kClusterCtas = 2;
    static constexpr uint32_t kAccumulatorStages = 2;
    static constexpr bool kPairAdjacentNTiles = false;
};

constexpr bool k3_multirange_wgrad_accumulates(
    const uint32_t reverse_iteration) {
    return reverse_iteration != 0u;
}

static_assert(K3MultiRangeWgradBodyContract::kLogicalBlockN == 256);
static_assert(K3MultiRangeWgradBodyContract::kClusterCtas == 2);
static_assert(K3MultiRangeWgradBodyContract::kAccumulatorStages == 2);
static_assert(!K3MultiRangeWgradBodyContract::kPairAdjacentNTiles);
static_assert(!k3_multirange_wgrad_accumulates(0u));
static_assert(k3_multirange_wgrad_accumulates(1u));

// Compile-time lifetime gates for allocations that change role in-place.
// dW buffers hold immutable BF16 weights through all corresponding dgrad work
// and become writable gradient outputs only after the cache-retirement fence.
constexpr bool k3_w2_cache_is_live(
    const K3MultiRangeBackwardPhase phase) {
    return !k3_phase_before(
        K3MultiRangeBackwardPhase::InitWeights, phase) ||
        phase == K3MultiRangeBackwardPhase::W2Dgrad;
}

constexpr bool k3_dw2_is_writable(
    const K3MultiRangeBackwardPhase phase) {
    return !k3_phase_before(
        phase, K3MultiRangeBackwardPhase::W2Wgrad);
}

constexpr bool k3_w13_cache_is_live(
    const K3MultiRangeBackwardPhase phase) {
    return k3_phase_before(
        phase, K3MultiRangeBackwardPhase::W13CacheRetired);
}

constexpr bool k3_dw13_is_writable(
    const K3MultiRangeBackwardPhase phase) {
    return !k3_phase_before(
        phase, K3MultiRangeBackwardPhase::W13Wgrad);
}

static_assert(k3_w2_cache_is_live(K3MultiRangeBackwardPhase::W2Dgrad));
static_assert(!k3_dw2_is_writable(K3MultiRangeBackwardPhase::W2Dgrad));
static_assert(!k3_w2_cache_is_live(K3MultiRangeBackwardPhase::W2Wgrad));
static_assert(k3_dw2_is_writable(K3MultiRangeBackwardPhase::W2Wgrad));
static_assert(k3_w13_cache_is_live(K3MultiRangeBackwardPhase::W13Dgrad));
static_assert(!k3_dw13_is_writable(K3MultiRangeBackwardPhase::W13Dgrad));
static_assert(!k3_w13_cache_is_live(K3MultiRangeBackwardPhase::W13Wgrad));
static_assert(k3_dw13_is_writable(K3MultiRangeBackwardPhase::W13Wgrad));
static_assert(k3_phase_before(
    K3MultiRangeBackwardPhase::GradXPublished,
    K3MultiRangeBackwardPhase::SymmetricPlanesReusable));

enum class K3W2RangeStorageEvent : uint32_t {
    ReadSavedDownForRouteDot = 1,
    WriteExactXToSavedDown = 2,
    ReadGradHForSiTU = 3,
    WriteHActToGradH = 4,
    WriteGradGateUpToSavedGateUp = 5,
    ReadGradGateUpForW13 = 6,
};

constexpr bool k3_storage_event_before(
    const K3W2RangeStorageEvent lhs,
    const K3W2RangeStorageEvent rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
}

static_assert(k3_storage_event_before(
    K3W2RangeStorageEvent::ReadSavedDownForRouteDot,
    K3W2RangeStorageEvent::WriteExactXToSavedDown));
static_assert(k3_storage_event_before(
    K3W2RangeStorageEvent::ReadGradHForSiTU,
    K3W2RangeStorageEvent::WriteHActToGradH));
static_assert(k3_storage_event_before(
    K3W2RangeStorageEvent::WriteGradGateUpToSavedGateUp,
    K3W2RangeStorageEvent::ReadGradGateUpForW13));

}  // namespace deep_gemm
