#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define DG_EXTERNAL_KGROUPED_HOST_DEVICE __host__ __device__
#else
#define DG_EXTERNAL_KGROUPED_HOST_DEVICE
#endif

namespace deep_gemm::sched {

struct ExternalKGroupedRangeDecodedTask {
    std::uint32_t group_idx;
    std::uint32_t m_block_idx;
    std::uint32_t n_block_idx;
    std::uint32_t shape_k;
    std::uint32_t k_cumsum;
    std::uint32_t sf_k_cumsum;
    std::uint32_t num_blocks_in_swizzle_group;
};

/** Four 128-feature operand panels consumed by one cluster output task. */
struct ExternalKGroupedFeaturePanels {
    std::uint32_t a_first;
    std::uint32_t a_second;
    std::uint32_t b_first;
    std::uint32_t b_second;
};

/** Decode Kimi-K3 dW2's fixed cluster task into its operand panels.
 *
 * K3 dW2 has fourteen 256-wide A tiles and twenty-four 128-wide B tiles.
 * The first scheduler swizzle group covers B0..B15 with eight cluster tasks
 * per A tile; the second covers B16..B23 with four tasks per A tile.  This
 * compact decoder is intentionally independent of prefix state so a dynamic
 * scheduler can check two acquire-loaded readiness masks per expert.
 */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr ExternalKGroupedFeaturePanels
external_k_grouped_k3_dw2_feature_panels(
        const std::uint32_t local_task) {
    constexpr std::uint32_t kFirstSwizzleTasks = 14u * 8u;
    constexpr std::uint32_t kNumAFeaturePanels = 28u;
    const bool second_swizzle = local_task >= kFirstSwizzleTasks;
    const std::uint32_t task = second_swizzle
        ? local_task - kFirstSwizzleTasks : local_task;
    const std::uint32_t tasks_per_a_tile = second_swizzle ? 4u : 8u;
    const std::uint32_t a_first =
        (task / tasks_per_a_tile) * 2u;
    const std::uint32_t b_first =
        kNumAFeaturePanels + (second_swizzle ? 16u : 0u) +
        (task % tasks_per_a_tile) * 2u;
    return {a_first, a_first + 1u, b_first, b_first + 1u};
}

/** Decode Kimi-K3 dW13's fixed cluster task into its operand panels.
 *
 * The first swizzle group has 24 A-tile pairs by eight B-panel pairs.  The
 * second has the same 24 A-tile pairs by the remaining six B-panel pairs.
 * A panels occupy bits/counters 0..47 and B panels 48..75.
 */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr ExternalKGroupedFeaturePanels
external_k_grouped_k3_dw13_feature_panels(
        const std::uint32_t local_task) {
    constexpr std::uint32_t kFirstSwizzleTasks = 24u * 8u;
    constexpr std::uint32_t kNumAFeaturePanels = 48u;
    const bool second_swizzle = local_task >= kFirstSwizzleTasks;
    const std::uint32_t task = second_swizzle
        ? local_task - kFirstSwizzleTasks : local_task;
    const std::uint32_t tasks_per_a_tile = second_swizzle ? 6u : 8u;
    const std::uint32_t a_first =
        (task / tasks_per_a_tile) * 2u;
    const std::uint32_t b_first =
        kNumAFeaturePanels + (second_swizzle ? 16u : 0u) +
        (task % tasks_per_a_tile) * 2u;
    return {a_first, a_first + 1u, b_first, b_first + 1u};
}

/** Map feature-major production order to physical A-then-B panel order.
 *
 * The first dW13 four-task quantum consumes A0/A1 and B0..B7.  Publishing
 * those panels first minimizes time-to-first-UMMA; every later panel follows
 * its first consumer in the fixed K3 swizzle.
 */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t
external_k_grouped_first_consumer_feature_panel(
        const std::uint32_t feature_ordinal,
        const std::uint32_t num_a_panels,
        const std::uint32_t num_b_panels,
        const std::uint32_t first_a_panels = 2u,
        const std::uint32_t first_b_panels = 16u) {
    if (feature_ordinal >= num_a_panels + num_b_panels)
        return num_a_panels + num_b_panels;
    if (feature_ordinal < first_a_panels)
        return feature_ordinal;
    if (feature_ordinal < first_a_panels + first_b_panels)
        return num_a_panels + feature_ordinal - first_a_panels;
    if (feature_ordinal < num_a_panels + first_b_panels)
        return feature_ordinal - first_b_panels;
    return feature_ordinal;
}

/** Test whether both 32-bit words contain all four panels for a dW2 task. */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr bool external_k_grouped_k3_dw2_task_is_feature_ready(
        const std::uint32_t local_task,
        const std::uint32_t ready_lo,
        const std::uint32_t ready_hi) {
    const auto panels = external_k_grouped_k3_dw2_feature_panels(local_task);
    const auto is_ready = [=](const std::uint32_t panel) {
        return panel < 32u
            ? (ready_lo & (1u << panel)) != 0u
            : (ready_hi & (1u << (panel - 32u))) != 0u;
    };
    return is_ready(panels.a_first) && is_ready(panels.a_second) &&
        is_ready(panels.b_first) && is_ready(panels.b_second);
}

// A two-segment task preserves the ordinary output-tile decode while exposing
// both disjoint physical K slices. ``output_task.shape_k`` is their logical
// concatenation, and ``output_task.k_cumsum`` remains the first slice's base.
// Keeping the first-slice length explicit lets a provider map every flattened
// K tile without materializing a per-tile descriptor list.
struct ExternalKGroupedTwoSegmentRangeDecodedTask {
    ExternalKGroupedRangeDecodedTask output_task;
    std::uint32_t first_segment_shape_k;
    std::uint32_t second_segment_shape_k;
    std::uint32_t first_segment_scale_rows;
    std::uint32_t second_segment_scale_rows;
    std::uint32_t second_segment_k_cumsum;
    std::uint32_t second_segment_sf_k_cumsum;
};

/** One output tile reduced across three immutable physical K segments.
 *
 * ``output_task.shape_k`` is the logical concatenation of all segments while
 * its physical prefix still names segment zero.  The remaining extents and
 * absolute prefixes let a descriptor-free terminal provider redirect each
 * aligned TMA K tile without materializing or copying concatenated operands.
 */
struct ExternalKGroupedThreeSegmentRangeDecodedTask {
    ExternalKGroupedRangeDecodedTask output_task;
    std::uint32_t first_segment_shape_k;
    std::uint32_t second_segment_shape_k;
    std::uint32_t third_segment_shape_k;
    std::uint32_t first_segment_scale_rows;
    std::uint32_t second_segment_scale_rows;
    std::uint32_t third_segment_scale_rows;
    std::uint32_t second_segment_k_cumsum;
    std::uint32_t third_segment_k_cumsum;
    std::uint32_t second_segment_sf_k_cumsum;
    std::uint32_t third_segment_sf_k_cumsum;
};

// Map one logical element (or scale-factor row) from the concatenated K axis
// into either physical segment. The caller supplies all values in the same
// units: elements for K, scale rows for SF_K.
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_two_segment_physical_index(
        const std::uint32_t logical_index,
        const std::uint32_t first_segment_size,
        const std::uint32_t first_segment_begin,
        const std::uint32_t second_segment_begin) {
    return logical_index < first_segment_size
        ? first_segment_begin + logical_index
        : second_segment_begin + logical_index - first_segment_size;
}

/** Map one logical K element (or scale row) into three physical segments. */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_three_segment_physical_index(
        const std::uint32_t logical_index,
        const std::uint32_t first_segment_size,
        const std::uint32_t second_segment_size,
        const std::uint32_t first_segment_begin,
        const std::uint32_t second_segment_begin,
        const std::uint32_t third_segment_begin) {
    if (logical_index < first_segment_size)
        return first_segment_begin + logical_index;
    if (logical_index < first_segment_size + second_segment_size) {
        return second_segment_begin + logical_index - first_segment_size;
    }
    return third_segment_begin + logical_index - first_segment_size -
        second_segment_size;
}

// A flattened TMA tile may change physical arenas only between complete K
// tiles.  A mid-tile segment boundary cannot be represented by changing the
// tile's starting coordinate: doing so either reads past the first segment or
// skips the beginning of the second.  Producers that use the two-segment
// provider must therefore zero-pad both logical segments to this alignment.
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr bool external_k_grouped_two_segment_is_tile_aligned(
        const std::uint32_t first_segment_size,
        const std::uint32_t second_segment_size,
        const std::uint32_t tile_size) {
    return tile_size != 0u &&
        first_segment_size % tile_size == 0u &&
        second_segment_size % tile_size == 0u;
}

/** Return whether every three-segment boundary lies on a complete TMA tile. */
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr bool external_k_grouped_three_segment_is_tile_aligned(
        const std::uint32_t first_segment_size,
        const std::uint32_t second_segment_size,
        const std::uint32_t third_segment_size,
        const std::uint32_t tile_size) {
    return tile_size != 0u &&
        first_segment_size % tile_size == 0u &&
        second_segment_size % tile_size == 0u &&
        third_segment_size % tile_size == 0u;
}

// Shared arithmetic for readiness counters.  Decode prefixes and readiness
// prefixes intentionally have independent units: exact MXFP8 uses element and
// compact-scale prefixes for TMA, while the parent publishes dgrad retirement
// credits per physical pool block.
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_retirement_target(
        const std::uint32_t first_readiness_units,
        const std::uint32_t second_readiness_units,
        const std::uint32_t retirements_per_unit,
        const std::uint32_t readiness_bias) {
    return (first_readiness_units + second_readiness_units) *
        retirements_per_unit + readiness_bias;
}

// Build the stable expert union consumed by a two-segment decoder. Both input
// prefix arrays must already be complete and immutable. The output list must
// not overlap either prefix; a caller may alias a proven-retired active list,
// but a pipelined caller must reserve a third arena because late roles can
// still acquire either input generation. This helper performs no publication:
// exactly one caller builds the list, initializes task-count/cursor/mailbox
// state, then publishes its generation with a release store. The provider's
// generation acquire and subsequent mailbox release make both prefixes and
// this union visible to all scheduler roles.
template <std::uint32_t kNumGroups,
          std::uint32_t kPoolPrefixWord = 31u,
          std::uint32_t kActiveExpertWord = 144u,
          std::uint32_t kActivityPrefixWord = kPoolPrefixWord>
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_build_two_segment_union(
        const std::uint32_t* first_segment_state_words,
        const std::uint32_t* second_segment_state_words,
        std::uint32_t* reused_first_segment_state_words) {
    std::uint32_t active_count = 0u;
    for (std::uint32_t group_idx = 0u;
         group_idx < kNumGroups; ++group_idx) {
        const bool first_active =
            first_segment_state_words[
                kActivityPrefixWord + group_idx + 1u] !=
            first_segment_state_words[
                kActivityPrefixWord + group_idx];
        const bool second_active =
            second_segment_state_words[
                kActivityPrefixWord + group_idx + 1u] !=
            second_segment_state_words[
                kActivityPrefixWord + group_idx];
        if (first_active or second_active) {
            reused_first_segment_state_words[
                kActiveExpertWord + active_count++] = group_idx;
        }
    }
    return active_count;
}

/** Build the stable expert union consumed by a three-segment decoder.
 *
 * All three input prefixes must be fully published and immutable.  The
 * output arena must be separate from them because it owns the logical first
 * prefix and active-expert union throughout the terminal UMMA/TMA lifetime.
 * Publication remains the caller's responsibility.
 */
template <std::uint32_t kNumGroups,
          std::uint32_t kPoolPrefixWord = 31u,
          std::uint32_t kActiveExpertWord = 144u,
          std::uint32_t kActivityPrefixWord = kPoolPrefixWord>
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_build_three_segment_union(
        const std::uint32_t* first_segment_state_words,
        const std::uint32_t* second_segment_state_words,
        const std::uint32_t* third_segment_state_words,
        std::uint32_t* reused_first_segment_state_words) {
    std::uint32_t active_count = 0u;
    for (std::uint32_t group_idx = 0u;
         group_idx < kNumGroups; ++group_idx) {
        const auto is_active = [=](const std::uint32_t* state_words) {
            return state_words[kActivityPrefixWord + group_idx + 1u] !=
                state_words[kActivityPrefixWord + group_idx];
        };
        if (is_active(first_segment_state_words) ||
            is_active(second_segment_state_words) ||
            is_active(third_segment_state_words)) {
            reused_first_segment_state_words[
                kActiveExpertWord + active_count++] = group_idx;
        }
    }
    return active_count;
}

// Map one persistent multicast cluster and local iteration to a compact
// logical-task index. Four consecutive tasks form one scheduling quantum so
// the embedded BF16 epilogue returns to its initial phase before resource
// release. Keeping the arithmetic host/device-shareable lets the contract test
// prove exact coverage without duplicating the device provider's indexing rule.
DG_EXTERNAL_KGROUPED_HOST_DEVICE
constexpr std::uint32_t external_k_grouped_strided_task_index(
        const std::uint32_t cluster_idx,
        const std::uint32_t iteration,
        const std::uint32_t num_clusters,
        const std::uint32_t tasks_per_quantum = 4u) {
    const auto quantum_iteration = iteration / tasks_per_quantum;
    const auto task_in_quantum = iteration % tasks_per_quantum;
    return (cluster_idx + quantum_iteration * num_clusters) *
        tasks_per_quantum + task_in_quantum;
}

// Host/device-shareable implementation of the compact K3 range decode.  Keep
// this header independent of CUTLASS/Cute so the exhaustive host model and the
// device provider execute the same arithmetic.
template <std::uint32_t BLOCK_M, std::uint32_t BLOCK_N,
          std::uint32_t kNumMulticast, bool kIsMulticastOnA,
          std::uint32_t SHAPE_M, std::uint32_t SHAPE_N,
          std::uint32_t kNum1DBlocksPerGroup,
          std::uint32_t kPoolBlockRows = 192u,
          std::uint32_t kSFKSpan = 64u,
          std::uint32_t kPoolPrefixWord = 31u,
          std::uint32_t kActiveExpertWord = 144u,
          std::uint32_t kValuePrefixWord = kPoolPrefixWord,
          std::uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          bool kPairAdjacentN = false>
struct ExternalKGroupedRangeDecoder {
    static constexpr std::uint32_t ceil_div(
            const std::uint32_t value, const std::uint32_t divisor) {
        return value / divisor + static_cast<std::uint32_t>(
            value % divisor != 0);
    }

    static constexpr std::uint32_t kNumMBlocks =
        ceil_div(SHAPE_M, BLOCK_M);
    static constexpr std::uint32_t kLogicalBlockN =
        BLOCK_N * (kPairAdjacentN ? 2u : 1u);
    static constexpr std::uint32_t kNumNBlocks =
        ceil_div(SHAPE_N, kLogicalBlockN);
    static constexpr std::uint32_t kPrimaryNumBlocks =
        kIsMulticastOnA ? kNumNBlocks : kNumMBlocks;
    static constexpr std::uint32_t kSecondaryNumBlocks =
        kIsMulticastOnA ? kNumMBlocks : kNumNBlocks;
    static constexpr std::uint32_t kNumCTAItemsPerGroup =
        kNumMBlocks * kNumNBlocks;
    static constexpr std::uint32_t kNumClusterTasksPerGroup =
        kNumCTAItemsPerGroup / kNumMulticast;
    static constexpr std::uint32_t kNumBlocksPerSwizzleGroup =
        kSecondaryNumBlocks * kNum1DBlocksPerGroup;

    static_assert(kNumMulticast == 1 or kNumMulticast == 2,
                  "Range decode supports one- or two-CTA clusters");
    static_assert(kNumMBlocks > 0 and kNumNBlocks > 0,
                  "Range decode requires nonempty output geometry");
    static_assert(!kPairAdjacentN || !kIsMulticastOnA,
                  "Paired-N decode requires multicast along M");
    static_assert(!kPairAdjacentN || SHAPE_N % (2u * BLOCK_N) == 0u,
                  "Paired-N decode requires complete adjacent N tiles");
    static_assert(kNumCTAItemsPerGroup % kNumMulticast == 0,
                  "Each expert must contain complete multicast clusters");
    static_assert(kNum1DBlocksPerGroup % kNumMulticast == 0,
                  "Swizzle groups must contain complete multicast clusters");
    static_assert(kPrimaryNumBlocks % kNumMulticast == 0,
                  "The final swizzle group must contain complete clusters");
    static_assert(kExplicitValueAndScalePrefixes ||
                      kPoolBlockRows % kSFKSpan == 0,
                  "Pool blocks must map to integral SF rows");

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedRangeDecodedTask decode_logical_task(
            const std::uint32_t* overlap_state_words,
            const std::uint32_t logical_task,
            const std::uint32_t cluster_rank) {
        const auto active_expert_idx =
            logical_task / kNumClusterTasksPerGroup;
        const auto task_in_group =
            logical_task % kNumClusterTasksPerGroup;
        const auto group_idx =
            overlap_state_words[kActiveExpertWord + active_expert_idx];

        std::uint32_t shape_k;
        std::uint32_t k_cumsum;
        std::uint32_t sf_k_cumsum;
        if constexpr (kExplicitValueAndScalePrefixes) {
            const auto first_value =
                overlap_state_words[kValuePrefixWord + group_idx];
            const auto last_value =
                overlap_state_words[kValuePrefixWord + group_idx + 1u];
            shape_k = last_value - first_value;
            k_cumsum = first_value;
            sf_k_cumsum =
                overlap_state_words[kScalePrefixWord + group_idx];
        } else {
            const auto first_pool_block =
                overlap_state_words[kPoolPrefixWord + group_idx];
            const auto last_pool_block =
                overlap_state_words[kPoolPrefixWord + group_idx + 1u];
            shape_k =
                (last_pool_block - first_pool_block) * kPoolBlockRows;
            k_cumsum = first_pool_block * kPoolBlockRows;
            sf_k_cumsum =
                first_pool_block * (kPoolBlockRows / kSFKSpan);
        }

        const auto cta_task = task_in_group * kNumMulticast + cluster_rank;
        const auto swizzle_group_idx =
            cta_task / kNumBlocksPerSwizzleGroup;
        const auto first_block_idx =
            swizzle_group_idx * kNum1DBlocksPerGroup;
        const auto in_group_idx =
            cta_task % kNumBlocksPerSwizzleGroup;
        const auto remaining_primary_blocks =
            kPrimaryNumBlocks - first_block_idx;
        const auto num_blocks_in_swizzle_group =
            remaining_primary_blocks < kNum1DBlocksPerGroup ?
                remaining_primary_blocks : kNum1DBlocksPerGroup;

        std::uint32_t m_block_idx;
        std::uint32_t n_block_idx;
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_swizzle_group;
            n_block_idx = first_block_idx +
                in_group_idx % num_blocks_in_swizzle_group;
        } else {
            m_block_idx = first_block_idx +
                in_group_idx % num_blocks_in_swizzle_group;
            n_block_idx = in_group_idx / num_blocks_in_swizzle_group;
        }

        return {
            group_idx,
            m_block_idx,
            n_block_idx * (kPairAdjacentN ? 2u : 1u),
            shape_k,
            k_cumsum,
            sf_k_cumsum,
            num_blocks_in_swizzle_group,
        };
    }

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedRangeDecodedTask decode_range_task(
            const std::uint32_t* overlap_state_words,
            const std::uint32_t first_logical_task,
            const std::uint32_t range_idx,
            const std::uint32_t cluster_rank) {
        return decode_logical_task(
            overlap_state_words,
            first_logical_task + range_idx,
            cluster_rank);
    }
};

// Decode one output tile against a union list stored in the first range's
// retired active-expert area and two disjoint pool-prefix arrays. Prefix values
// are absolute pool-block offsets, so either segment may be empty and no
// range-base correction is required. Segment order is semantically visible:
// callers must pass the two states in the accumulation order they require.
template <std::uint32_t BLOCK_M, std::uint32_t BLOCK_N,
          std::uint32_t kNumMulticast, bool kIsMulticastOnA,
          std::uint32_t SHAPE_M, std::uint32_t SHAPE_N,
          std::uint32_t kNum1DBlocksPerGroup,
          std::uint32_t kPoolBlockRows = 192u,
          std::uint32_t kSFKSpan = 64u,
          std::uint32_t kPoolPrefixWord = 31u,
          std::uint32_t kActiveExpertWord = 144u,
          std::uint32_t kValuePrefixWord = kPoolPrefixWord,
          std::uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          bool kPairAdjacentN = false>
struct ExternalKGroupedTwoSegmentRangeDecoder {
    using FirstSegmentDecoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes, kPairAdjacentN>;

    static constexpr std::uint32_t kNumMBlocks =
        FirstSegmentDecoder::kNumMBlocks;
    static constexpr std::uint32_t kNumNBlocks =
        FirstSegmentDecoder::kNumNBlocks;
    static constexpr std::uint32_t kNumClusterTasksPerGroup =
        FirstSegmentDecoder::kNumClusterTasksPerGroup;

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedTwoSegmentRangeDecodedTask
    decode_logical_task(
            const std::uint32_t* reused_first_segment_state_words,
            const std::uint32_t* second_segment_state_words,
            const std::uint32_t logical_task,
            const std::uint32_t cluster_rank) {
        auto output_task = FirstSegmentDecoder::decode_logical_task(
            reused_first_segment_state_words, logical_task, cluster_rank);
        const auto first_segment_shape_k = output_task.shape_k;
        std::uint32_t first_segment_scale_rows;
        const auto group_idx = output_task.group_idx;
        std::uint32_t second_segment_shape_k;
        std::uint32_t second_segment_scale_rows;
        std::uint32_t second_segment_k_cumsum;
        std::uint32_t second_segment_sf_k_cumsum;
        if constexpr (kExplicitValueAndScalePrefixes) {
            first_segment_scale_rows =
                reused_first_segment_state_words[
                    kScalePrefixWord + group_idx + 1u] -
                reused_first_segment_state_words[
                    kScalePrefixWord + group_idx];
            const auto second_first_value = second_segment_state_words[
                kValuePrefixWord + group_idx];
            const auto second_last_value = second_segment_state_words[
                kValuePrefixWord + group_idx + 1u];
            second_segment_shape_k =
                second_last_value - second_first_value;
            second_segment_scale_rows =
                second_segment_state_words[
                    kScalePrefixWord + group_idx + 1u] -
                second_segment_state_words[
                    kScalePrefixWord + group_idx];
            second_segment_k_cumsum = second_first_value;
            second_segment_sf_k_cumsum = second_segment_state_words[
                kScalePrefixWord + group_idx];
        } else {
            first_segment_scale_rows =
                first_segment_shape_k / kSFKSpan;
            const auto second_first_pool_block =
                second_segment_state_words[
                    kPoolPrefixWord + group_idx];
            const auto second_last_pool_block =
                second_segment_state_words[
                    kPoolPrefixWord + group_idx + 1u];
            second_segment_shape_k =
                (second_last_pool_block - second_first_pool_block) *
                kPoolBlockRows;
            second_segment_scale_rows =
                second_segment_shape_k / kSFKSpan;
            second_segment_k_cumsum =
                second_first_pool_block * kPoolBlockRows;
            second_segment_sf_k_cumsum =
                second_first_pool_block *
                (kPoolBlockRows / kSFKSpan);
        }
        output_task.shape_k += second_segment_shape_k;

        return {
            output_task,
            first_segment_shape_k,
            second_segment_shape_k,
            first_segment_scale_rows,
            second_segment_scale_rows,
            second_segment_k_cumsum,
            second_segment_sf_k_cumsum,
        };
    }

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedTwoSegmentRangeDecodedTask
    decode_range_task(
            const std::uint32_t* reused_first_segment_state_words,
            const std::uint32_t* second_segment_state_words,
            const std::uint32_t first_logical_task,
            const std::uint32_t range_idx,
            const std::uint32_t cluster_rank) {
        return decode_logical_task(
            reused_first_segment_state_words,
            second_segment_state_words,
            first_logical_task + range_idx,
            cluster_rank);
    }
};

/** Decode one output tile over three disjoint, immutable K prefixes.
 *
 * The first state owns a copied segment-zero prefix plus the three-way expert
 * union.  Segment one and two keep their absolute physical prefixes.  The
 * returned logical K extent drives one FP32 accumulation and one epilogue,
 * while the provider maps individual TMA tiles back to the physical arenas.
 */
template <std::uint32_t BLOCK_M, std::uint32_t BLOCK_N,
          std::uint32_t kNumMulticast, bool kIsMulticastOnA,
          std::uint32_t SHAPE_M, std::uint32_t SHAPE_N,
          std::uint32_t kNum1DBlocksPerGroup,
          std::uint32_t kPoolBlockRows = 192u,
          std::uint32_t kSFKSpan = 64u,
          std::uint32_t kPoolPrefixWord = 31u,
          std::uint32_t kActiveExpertWord = 144u,
          std::uint32_t kValuePrefixWord = kPoolPrefixWord,
          std::uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          bool kPairAdjacentN = false>
struct ExternalKGroupedThreeSegmentRangeDecoder {
    using FirstSegmentDecoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes, kPairAdjacentN>;

    static constexpr std::uint32_t kNumMBlocks =
        FirstSegmentDecoder::kNumMBlocks;
    static constexpr std::uint32_t kNumNBlocks =
        FirstSegmentDecoder::kNumNBlocks;
    static constexpr std::uint32_t kNumClusterTasksPerGroup =
        FirstSegmentDecoder::kNumClusterTasksPerGroup;

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedThreeSegmentRangeDecodedTask
    decode_logical_task(
            const std::uint32_t* reused_first_segment_state_words,
            const std::uint32_t* second_segment_state_words,
            const std::uint32_t* third_segment_state_words,
            const std::uint32_t logical_task,
            const std::uint32_t cluster_rank) {
        auto output_task = FirstSegmentDecoder::decode_logical_task(
            reused_first_segment_state_words, logical_task, cluster_rank);
        const auto first_segment_shape_k = output_task.shape_k;
        const auto group_idx = output_task.group_idx;

        std::uint32_t first_segment_scale_rows;
        std::uint32_t second_segment_shape_k;
        std::uint32_t third_segment_shape_k;
        std::uint32_t second_segment_scale_rows;
        std::uint32_t third_segment_scale_rows;
        std::uint32_t second_segment_k_cumsum;
        std::uint32_t third_segment_k_cumsum;
        std::uint32_t second_segment_sf_k_cumsum;
        std::uint32_t third_segment_sf_k_cumsum;
        if constexpr (kExplicitValueAndScalePrefixes) {
            const auto decode_segment = [=](
                    const std::uint32_t* state_words,
                    std::uint32_t& shape_k,
                    std::uint32_t& scale_rows,
                    std::uint32_t& k_cumsum,
                    std::uint32_t& sf_k_cumsum) {
                const auto first_value =
                    state_words[kValuePrefixWord + group_idx];
                const auto last_value =
                    state_words[kValuePrefixWord + group_idx + 1u];
                shape_k = last_value - first_value;
                scale_rows =
                    state_words[kScalePrefixWord + group_idx + 1u] -
                    state_words[kScalePrefixWord + group_idx];
                k_cumsum = first_value;
                sf_k_cumsum =
                    state_words[kScalePrefixWord + group_idx];
            };
            first_segment_scale_rows =
                reused_first_segment_state_words[
                    kScalePrefixWord + group_idx + 1u] -
                reused_first_segment_state_words[
                    kScalePrefixWord + group_idx];
            decode_segment(
                second_segment_state_words,
                second_segment_shape_k, second_segment_scale_rows,
                second_segment_k_cumsum, second_segment_sf_k_cumsum);
            decode_segment(
                third_segment_state_words,
                third_segment_shape_k, third_segment_scale_rows,
                third_segment_k_cumsum, third_segment_sf_k_cumsum);
        } else {
            const auto decode_segment = [=](
                    const std::uint32_t* state_words,
                    std::uint32_t& shape_k,
                    std::uint32_t& scale_rows,
                    std::uint32_t& k_cumsum,
                    std::uint32_t& sf_k_cumsum) {
                const auto first_pool_block =
                    state_words[kPoolPrefixWord + group_idx];
                const auto last_pool_block =
                    state_words[kPoolPrefixWord + group_idx + 1u];
                shape_k =
                    (last_pool_block - first_pool_block) * kPoolBlockRows;
                scale_rows = shape_k / kSFKSpan;
                k_cumsum = first_pool_block * kPoolBlockRows;
                sf_k_cumsum = first_pool_block *
                    (kPoolBlockRows / kSFKSpan);
            };
            first_segment_scale_rows =
                first_segment_shape_k / kSFKSpan;
            decode_segment(
                second_segment_state_words,
                second_segment_shape_k, second_segment_scale_rows,
                second_segment_k_cumsum, second_segment_sf_k_cumsum);
            decode_segment(
                third_segment_state_words,
                third_segment_shape_k, third_segment_scale_rows,
                third_segment_k_cumsum, third_segment_sf_k_cumsum);
        }
        output_task.shape_k +=
            second_segment_shape_k + third_segment_shape_k;

        return {
            output_task,
            first_segment_shape_k,
            second_segment_shape_k,
            third_segment_shape_k,
            first_segment_scale_rows,
            second_segment_scale_rows,
            third_segment_scale_rows,
            second_segment_k_cumsum,
            third_segment_k_cumsum,
            second_segment_sf_k_cumsum,
            third_segment_sf_k_cumsum,
        };
    }

    DG_EXTERNAL_KGROUPED_HOST_DEVICE
    static constexpr ExternalKGroupedThreeSegmentRangeDecodedTask
    decode_range_task(
            const std::uint32_t* reused_first_segment_state_words,
            const std::uint32_t* second_segment_state_words,
            const std::uint32_t* third_segment_state_words,
            const std::uint32_t first_logical_task,
            const std::uint32_t range_idx,
            const std::uint32_t cluster_rank) {
        return decode_logical_task(
            reused_first_segment_state_words,
            second_segment_state_words,
            third_segment_state_words,
            first_logical_task + range_idx,
            cluster_rank);
    }
};

} // namespace deep_gemm::sched

#undef DG_EXTERNAL_KGROUPED_HOST_DEVICE
