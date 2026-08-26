#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/scheduler/external_k_grouped_range.hpp>

namespace deep_gemm::sched {

enum class IndexType {
    MN,
    K,
    SF_K,
};

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kEnsureZeroPadding = true,
          uint32_t kKAlignment = 128u,     // psum k-group start alignment
          uint32_t kSFKSpan = 512u,        // K covered by one k-grouped SF row
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    static constexpr GemmType kTaskGemmType = kGemmType;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kKAlignment;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = 0;
    static constexpr uint32_t kTaskShapeN = 0;
    static constexpr uint32_t kTaskPoolBlockRows = 0;

    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    // NOTES: only used by the non-psum path; the psum path never reads them.
    uint32_t next_group_idx, next_shape_k;
    // Only used for `KGroupedContiguousWithPsumLayout`
    uint32_t current_k_start = 0, current_k_end = 0;

    // Only used for k-grouped gemm
    CUTLASS_DEVICE void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = grouped_layout[group_idx];
            if (shape_k > 0)
                break;
        }
    }

    CUTLASS_DEVICE void get_next_psum_k_group(uint32_t &group_idx, uint32_t &shape_k,
                                               uint32_t &k_start, uint32_t &k_end) const {
        // NOTES: `grouped_layout[i]` is the psum end offset (K elements); each group starts at `align(prev_end, kKAlignment)`. Skip empty groups.
        for (; group_idx < kNumGroups; ++ group_idx) {
            const auto next_k_end = static_cast<uint32_t>(grouped_layout[group_idx]);
            k_start = math::align(k_end, kKAlignment);
            shape_k = next_k_end - k_start;
            k_end = next_k_end;
            if (shape_k > 0)
                break;
        }
    }

    // ReSharper disable once CppPossiblyUninitializedMember
    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout = nullptr) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
                get_next_psum_k_group(current_group_idx, current_shape_k, current_k_start, current_k_end);
            } else {
                get_next_k_group(current_group_idx, current_shape_k);
                next_group_idx = current_group_idx + 1;
                get_next_k_group(next_group_idx, next_shape_k);
            }
        }
    }

    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN) {
                    offset = current_group_idx * shape_dim;
                } else if constexpr (kIndexType == IndexType::K) {
                    if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout)
                        offset = current_k_start;
                    else
                        offset = current_k_cumsum;
                } else if constexpr (kIndexType == IndexType::SF_K) {
                    offset = current_sf_k_cumsum;
                }
            }
            return offset + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = math::align(current_psum_m, BLOCK_M);
                current_psum_m = grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if constexpr (is_k_grouped_contiguous(kGemmType)) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                if (next_block_idx < (current_num_valid_groups + 1) * num_blocks)
                    break;

                // Move to check the next group
                current_sf_k_cumsum += math::ceil_div(current_shape_k, kSFKSpan);
                current_num_valid_groups ++;
                if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
                    get_next_psum_k_group(++ current_group_idx, current_shape_k, current_k_start, current_k_end);
                } else {
                    current_k_cumsum += current_shape_k;
                    current_group_idx = next_group_idx ++;
                    current_shape_k = next_shape_k;
                    get_next_k_group(next_group_idx, next_shape_k);
                }
            }

            get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            // For SM90 only
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // Always aligned on N (constant bypass)
                                num_m_blocks % kNumMulticast == 0 or                  // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;                    // Peer CTA in bound
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            // Unreachable 
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

// One complete-cluster K-grouped GEMM task. ``m_block_idx`` and
// ``n_block_idx`` name CTA-rank 0; the provider derives the peer CTA's adjacent
// coordinate from its cluster rank.  ``k_cumsum`` is the exact element offset
// of this expert's K slice, while ``sf_k_cumsum`` is its scale-factor-row
// offset.  Keeping both offsets explicit avoids rescanning grouped_layout and
// preserves the caller's K accumulation order.
struct ExternalKGroupedTask {
    uint32_t group_idx;
    uint32_t m_block_idx;
    uint32_t n_block_idx;
    uint32_t shape_k;
    uint32_t k_cumsum;
    uint32_t sf_k_cumsum;
};

// Opaque launch descriptor passed through BF16 GEMM's existing grouped-layout
// slot when an external provider is selected.  This keeps the default kernel
// ABI and call path unchanged.
struct ExternalKGroupedTaskStream {
    const ExternalKGroupedTask* tasks;
    uint32_t num_tasks;
};

// A compact contiguous slice of logical K-grouped cluster tasks.  The state
// pointer addresses the 256-word K3 overlap-state model: pool-prefix words
// 31..143 and active-expert words 144..255.  Unlike
// ``ExternalKGroupedTaskStream``, this stream materializes no per-task
// descriptors.
struct ExternalKGroupedRangeStream {
    const uint32_t* overlap_state_words;
    uint32_t first_logical_task;
    uint32_t num_tasks;

    // Optional dynamic-provider fields.  Fixed-range providers ignore these;
    // keeping one stream ABI lets an embedded GEMM switch scheduling policy
    // without changing the body entry point.
    uint32_t* task_cursor = nullptr;
    uint32_t task_limit = 0;
    uint32_t* cluster_mailbox = nullptr;
    uint32_t batch_tasks = 0;
    uint32_t tasks_per_expert = 0;
    const uint32_t* expert_retired_counts = nullptr;
    uint32_t retirements_per_pool_block = 0;
    // Optional release-published generation for callers that reuse one
    // scheduler arena across sequential descriptor ranges.  Only the sole
    // publisher waits; its mailbox release makes the initialized state visible
    // to every other scheduler role without a second device-wide join.
    const uint32_t* state_ready_epoch = nullptr;
    uint32_t expected_state_epoch = 0;
    // Optional release-published task-count word. Persistent fixed-state
    // providers acquire the generation first, then load this word themselves;
    // a caller-side value loaded before the acquire would race arena reuse.
    const uint32_t* published_num_tasks = nullptr;
};

// Two-segment providers use a distinct wrapper so the hot single-range stream
// ABI and its per-thread local footprint remain unchanged. ``first_segment``
// points at the first range's prefix plus the rebuilt union active-expert list;
// the second pointer addresses the other range's disjoint prefix.
//
// Publication contract:
//   1. the sole publisher acquires completion of both prefix/operand writers;
//   2. it builds the union, resets cursor and every mailbox word, and publishes
//      the union task count;
//   3. after any required async/global proxy fence, it release-stores a fresh
//      generation to ``first_segment.state_ready_epoch``;
//   4. neither prefix nor the union list may be reused until all scheduler roles
//      consume their terminal mailbox token.
//
// The leader TMA role acquires that generation before the task count or union,
// and its release-published mailbox sequence carries visibility transitively to
// the MMA and epilogue roles. No second generation word is required.
struct ExternalKGroupedTwoSegmentRangeStream {
    ExternalKGroupedRangeStream first_segment;
    const uint32_t* second_segment_state_words;
    // Optional progressive one-way phase handoff. The sole mailbox publisher
    // checks this monotonic word and the unclaimed-task tail only between fully
    // acknowledged batches. Suffix clusters peel off as the tail shrinks while
    // the prefix below ``first_transition_cluster`` always drains the shared
    // cursor. Publishing a zero-sized batch retires the complete two-CTA
    // cluster without dropping a claimed task.
    const uint32_t* transition_ready = nullptr;
    uint32_t first_transition_cluster = 0u;
};

/** Immutable two-segment stream for a terminal BF16 K-grouped reduction.
 *
 * Both prefix states and their union active-expert list are fully published
 * before the embedded GEMM starts.  Unlike the general two-segment stream,
 * this descriptor carries no generation, retirement, transition, or
 * per-expert readiness state.  Keeping it distinct also leaves the hot
 * one-range stream ABI and specialization unchanged.
 */
struct ExternalKGroupedTerminalTwoSegmentRangeStream {
    const uint32_t* first_segment_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;
    uint32_t batch_tasks;
    uint32_t tasks_per_expert;
};

/** Immutable three-segment stream for terminal BF16 K reductions.
 *
 * The first state owns a copied physical prefix and the three-way expert
 * union. The second state is the later of two contiguous physical range
 * arenas; the provider derives the third state with its compile-time stride.
 * One shared cursor/mailbox schedules complete output tiles, so the GEMM
 * keeps a single FP32 accumulator across all three TMA-backed segments and
 * stores exactly once without adding a workspace allocation.
 */
struct ExternalKGroupedTerminalThreeSegmentRangeStream {
    const uint32_t* first_segment_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;
    uint32_t batch_tasks;
    uint32_t tasks_per_expert;
};

template <bool kTwoSegmentK>
struct ExternalKGroupedRangeStreamAccessor;

template <>
struct ExternalKGroupedRangeStreamAccessor<false> {
    CUTLASS_DEVICE static const ExternalKGroupedRangeStream& first(
            int* opaque_task_stream) {
        return *reinterpret_cast<const ExternalKGroupedRangeStream*>(
            opaque_task_stream);
    }

    CUTLASS_DEVICE static const uint32_t* second(int*) {
        return nullptr;
    }

    CUTLASS_DEVICE static const uint32_t* transition_ready(int*) {
        return nullptr;
    }

    CUTLASS_DEVICE static uint32_t first_transition_cluster(int*) {
        return 0u;
    }
};

template <>
struct ExternalKGroupedRangeStreamAccessor<true> {
    CUTLASS_DEVICE static const ExternalKGroupedRangeStream& first(
            int* opaque_task_stream) {
        return reinterpret_cast<
            const ExternalKGroupedTwoSegmentRangeStream*>(
                opaque_task_stream)->first_segment;
    }

    CUTLASS_DEVICE static const uint32_t* second(
            int* opaque_task_stream) {
        return reinterpret_cast<
            const ExternalKGroupedTwoSegmentRangeStream*>(
                opaque_task_stream)->second_segment_state_words;
    }

    CUTLASS_DEVICE static const uint32_t* transition_ready(
            int* opaque_task_stream) {
        return reinterpret_cast<
            const ExternalKGroupedTwoSegmentRangeStream*>(
                opaque_task_stream)->transition_ready;
    }

    CUTLASS_DEVICE static uint32_t first_transition_cluster(
            int* opaque_task_stream) {
        return reinterpret_cast<
            const ExternalKGroupedTwoSegmentRangeStream*>(
                opaque_task_stream)->first_transition_cluster;
    }
};

// Adapter for a caller-published list of complete-cluster K-grouped tasks.
//
// Publication contract:
//   * every CTA in a multicast cluster receives the same ``tasks`` pointer and
//     ``num_tasks``;
//   * task storage remains immutable until every role has exhausted the list;
//   * the publisher establishes shared/global visibility before entry; and
//   * rank-0 coordinates have room for all ``kNumMulticast`` adjacent CTAs.
//
// The adapter intentionally mirrors the subset of ``Scheduler`` consumed by
// SM100 BF16 GEMM.  It never derives a logical output tile from physical
// ``blockIdx.x``.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kKAlignment = 1u>
struct ExternalKGroupedTaskProvider {
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kKAlignment;
    static constexpr uint32_t kTaskNumSMs = 0;
    static constexpr uint32_t kTaskShapeM = 0;
    static constexpr uint32_t kTaskShapeN = 0;
    static constexpr uint32_t kTaskPoolBlockRows = 0;

    static_assert(
        kNumMulticast == 1 or kNumMulticast == 2,
        "External K-grouped tasks support one- or two-CTA clusters");

    int current_iter = -1;

    uint32_t num_blocks_in_group = kNumMulticast;
    bool is_peer_cta_alive = true;

    uint32_t current_group_idx = 0;
    uint32_t current_shape_k = 0;
    uint32_t current_k_cumsum = 0;
    uint32_t current_sf_k_cumsum = 0;

    const ExternalKGroupedTask* tasks;
    uint32_t num_tasks;

    CUTLASS_DEVICE explicit ExternalKGroupedTaskProvider(
            const ExternalKGroupedTask* tasks = nullptr,
            const uint32_t num_tasks = 0):
        tasks(tasks), num_tasks(num_tasks) {}

    CUTLASS_DEVICE explicit ExternalKGroupedTaskProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream) {
        const auto* task_stream =
            reinterpret_cast<const ExternalKGroupedTaskStream*>(
                opaque_task_stream);
        tasks = task_stream->tasks;
        num_tasks = task_stream->num_tasks;
    }

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto task_idx = static_cast<uint32_t>(++ current_iter);
        if (task_idx >= num_tasks)
            return false;

        const auto task = tasks[task_idx];
        current_group_idx = task.group_idx;
        current_shape_k = task.shape_k;
        current_k_cumsum = task.k_cumsum;
        current_sf_k_cumsum = task.sf_k_cumsum;

        const auto cluster_rank = static_cast<uint32_t>(
            cute::block_rank_in_cluster());
        m_block_idx = task.m_block_idx +
            (kIsMulticastOnA ? 0u : cluster_rank);
        n_block_idx = task.n_block_idx +
            (kIsMulticastOnA ? cluster_rank : 0u);
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0) const {
        uint32_t offset = 0;
        if constexpr (kWithGroupOffset) {
            if constexpr (kIndexType == IndexType::MN) {
                offset = current_group_idx * shape_dim;
            } else if constexpr (kIndexType == IndexType::K) {
                offset = current_k_cumsum;
            } else if constexpr (kIndexType == IndexType::SF_K) {
                offset = current_sf_k_cumsum;
            }
        }
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    // These methods are not used by the SM100 path, but retaining Scheduler's
    // surface keeps the provider safe for shared generic helpers.
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return kNumMulticast > 1;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

// Decode a contiguous range of complete-cluster K-grouped tasks directly from
// the compact K3 overlap state.  Logical tasks use the exact CTA-linear order
// of ``Scheduler::get_swizzled_block_idx`` compressed by ``kNumMulticast``;
// the provider therefore preserves the standalone BF16 scheduler's M/N
// swizzle.  The default mode decodes a caller-owned contiguous range;
// persistent mode deterministically stripes four-task quanta across physical
// two-CTA clusters using ``blockIdx.x``.
//
// ``SHAPE_M`` and ``SHAPE_N`` are compile-time output dimensions so divisions
// and the 1-D swizzle geometry fold to constants for the two K3 wgrads.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>(),
          bool kPersistentStrided = false,
          uint32_t kTasksPerQuantum = 4u,
          bool kTwoSegmentK = false,
          uint32_t kValuePrefixWord = kPoolPrefixWord,
          uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          uint32_t kValueAlignment = kSFKSpan,
          uint32_t kTmaTileK = kSFKSpan>
struct ExternalKGroupedRangeProvider {
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kValueAlignment;
    static constexpr uint32_t kTaskScaleKSpan = kSFKSpan;
    static constexpr uint32_t kTaskTmaTileK = kTmaTileK;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = SHAPE_M;
    static constexpr uint32_t kTaskShapeN = SHAPE_N;
    static constexpr uint32_t kTaskPoolBlockRows = kPoolBlockRows;
    static constexpr bool kTaskHasTwoSegmentK = kTwoSegmentK;

    using Decoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes>;
    using TwoSegmentDecoder = ExternalKGroupedTwoSegmentRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes>;

    static constexpr uint32_t kNumMBlocks = Decoder::kNumMBlocks;
    static constexpr uint32_t kNumNBlocks = Decoder::kNumNBlocks;
    static constexpr uint32_t kNumClusterTasksPerGroup =
        Decoder::kNumClusterTasksPerGroup;
    static constexpr uint32_t kNumClusters =
        kNumSMs / kNumMulticast;

    static_assert(!kPersistentStrided ||
                      kNumSMs % kNumMulticast == 0u,
                  "Persistent K-grouped ranges require complete clusters");
    static_assert(!kPersistentStrided || kTasksPerQuantum != 0u,
                  "Persistent K-grouped ranges require nonempty quanta");
    static_assert(!kPersistentStrided || kTasksPerQuantum % 4u == 0u,
                  "Persistent K-grouped ranges must reset epilogue phases");

    int current_iter = -1;

    uint32_t num_blocks_in_group = kNumMulticast;
    bool is_peer_cta_alive = true;

    uint32_t current_group_idx = 0;
    uint32_t current_shape_k = 0;
    uint32_t current_k_cumsum = 0;
    uint32_t current_sf_k_cumsum = 0;
    uint32_t current_first_segment_shape_k = 0;
    uint32_t current_first_segment_scale_rows = 0;
    uint32_t current_second_segment_k_cumsum = 0;
    uint32_t current_second_segment_sf_k_cumsum = 0;

    const uint32_t* overlap_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t first_logical_task;
    uint32_t num_tasks;
    const uint32_t* state_ready_epoch;
    uint32_t expected_state_epoch;
    const uint32_t* published_num_tasks;

    CUTLASS_DEVICE explicit ExternalKGroupedRangeProvider(
            const ExternalKGroupedRangeStream& stream):
        ExternalKGroupedRangeProvider(stream, nullptr) {
        static_assert(
            !kTwoSegmentK,
            "Two-segment provider requires its two-segment stream wrapper");
    }

    CUTLASS_DEVICE explicit ExternalKGroupedRangeProvider(
            const ExternalKGroupedTwoSegmentRangeStream& stream):
        ExternalKGroupedRangeProvider(
            stream.first_segment, stream.second_segment_state_words) {
        static_assert(
            kTwoSegmentK,
            "Ordinary provider requires its single-range stream");
    }

    CUTLASS_DEVICE explicit ExternalKGroupedRangeProvider(
            const ExternalKGroupedRangeStream& stream,
            const uint32_t* second_segment_state_words):
        overlap_state_words(stream.overlap_state_words),
        second_segment_state_words(second_segment_state_words),
        first_logical_task(stream.first_logical_task),
        num_tasks(stream.num_tasks),
        state_ready_epoch(stream.state_ready_epoch),
        expected_state_epoch(stream.expected_state_epoch),
        published_num_tasks(stream.published_num_tasks) {
        if constexpr (kTwoSegmentK)
            DG_DEVICE_ASSERT(second_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(
            published_num_tasks == nullptr || state_ready_epoch != nullptr);
    }

    CUTLASS_DEVICE explicit ExternalKGroupedRangeProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream):
        ExternalKGroupedRangeProvider(
            ExternalKGroupedRangeStreamAccessor<kTwoSegmentK>::first(
                opaque_task_stream),
            ExternalKGroupedRangeStreamAccessor<kTwoSegmentK>::second(
                opaque_task_stream)) {}

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto iteration = static_cast<uint32_t>(++ current_iter);
        uint32_t range_idx = iteration;
        if constexpr (kPersistentStrided) {
            if (iteration == 0u && state_ready_epoch != nullptr) {
                while (ptx::ld_acq(state_ready_epoch) !=
                       expected_state_epoch) {
                    __nanosleep(64);
                }
                if (published_num_tasks != nullptr)
                    num_tasks = ptx::ld_acq(published_num_tasks);
            }
            const auto cluster_idx =
                static_cast<uint32_t>(blockIdx.x) / kNumMulticast;
            range_idx = external_k_grouped_strided_task_index(
                cluster_idx, iteration, kNumClusters,
                kTasksPerQuantum);
        }
        if (range_idx >= num_tasks)
            return false;

        const auto cluster_rank = static_cast<uint32_t>(
            cute::block_rank_in_cluster());
        if constexpr (kTwoSegmentK) {
            const auto decoded = TwoSegmentDecoder::decode_range_task(
                overlap_state_words, second_segment_state_words,
                first_logical_task, range_idx, cluster_rank);
            const auto& task = decoded.output_task;
            current_group_idx = task.group_idx;
            current_shape_k = task.shape_k;
            current_k_cumsum = task.k_cumsum;
            current_sf_k_cumsum = task.sf_k_cumsum;
            current_first_segment_shape_k =
                decoded.first_segment_shape_k;
            current_first_segment_scale_rows =
                decoded.first_segment_scale_rows;
            current_second_segment_k_cumsum =
                decoded.second_segment_k_cumsum;
            current_second_segment_sf_k_cumsum =
                decoded.second_segment_sf_k_cumsum;
            DG_DEVICE_ASSERT(
                external_k_grouped_two_segment_is_tile_aligned(
                    decoded.first_segment_shape_k,
                    decoded.second_segment_shape_k,
                    kTmaTileK));
            num_blocks_in_group = task.num_blocks_in_swizzle_group;
            m_block_idx = task.m_block_idx;
            n_block_idx = task.n_block_idx;
        } else {
            const auto task = Decoder::decode_range_task(
                overlap_state_words, first_logical_task,
                range_idx, cluster_rank);
            current_group_idx = task.group_idx;
            current_shape_k = task.shape_k;
            current_k_cumsum = task.k_cumsum;
            current_sf_k_cumsum = task.sf_k_cumsum;
            num_blocks_in_group = task.num_blocks_in_swizzle_group;
            m_block_idx = task.m_block_idx;
            n_block_idx = task.n_block_idx;
        }
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0) const {
        if constexpr (
                kTwoSegmentK && kIndexType == IndexType::K) {
            static_assert(
                kWithGroupOffset,
                "Two-segment K operands require grouped physical offsets");
            return external_k_grouped_two_segment_physical_index(
                block_idx * block_size,
                current_first_segment_shape_k,
                current_k_cumsum,
                current_second_segment_k_cumsum);
        } else if constexpr (
                kTwoSegmentK && kIndexType == IndexType::SF_K) {
            static_assert(
                kWithGroupOffset,
                "Two-segment SF_K operands require grouped physical offsets");
            return external_k_grouped_two_segment_physical_index(
                block_idx * block_size,
                current_first_segment_scale_rows,
                current_sf_k_cumsum,
                current_second_segment_sf_k_cumsum);
        }
        uint32_t offset = 0;
        if constexpr (kWithGroupOffset) {
            if constexpr (kIndexType == IndexType::MN) {
                offset = current_group_idx * shape_dim;
            } else if constexpr (kIndexType == IndexType::K) {
                offset = current_k_cumsum;
            } else if constexpr (kIndexType == IndexType::SF_K) {
                offset = current_sf_k_cumsum;
            }
        }
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return kNumMulticast > 1;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

// Convenience spelling for a fixed/strided stream whose first state contains
// the union active-expert list and whose stream supplies a second prefix.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>(),
          bool kPersistentStrided = false,
          uint32_t kTasksPerQuantum = 4u,
          uint32_t kValuePrefixWord = kPoolPrefixWord,
          uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          uint32_t kValueAlignment = kSFKSpan,
          uint32_t kTmaTileK = kSFKSpan>
using ExternalKGroupedTwoSegmentRangeProvider =
    ExternalKGroupedRangeProvider<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        kNumSMs, SHAPE_M, SHAPE_N,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kNum1DBlocksPerGroup,
        kPersistentStrided, kTasksPerQuantum, true,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes,
        kValueAlignment, kTmaTileK>;

template <bool kEnabled>
struct ExternalKGroupedOneWayClusterTransition {
    CUTLASS_DEVICE explicit ExternalKGroupedOneWayClusterTransition(
            const uint32_t*, const uint32_t) {}

    CUTLASS_DEVICE bool requested(
            const uint32_t, const uint32_t*, const uint32_t,
            const uint32_t, const uint32_t) const {
        return false;
    }
};

template <>
struct ExternalKGroupedOneWayClusterTransition<true> {
    const uint32_t* ready;
    uint32_t first_cluster;

    CUTLASS_DEVICE explicit ExternalKGroupedOneWayClusterTransition(
            const uint32_t* ready_, const uint32_t first_cluster_):
        ready(ready_), first_cluster(first_cluster_) {}

    /** Return true once this suffix cluster can leave the final dW2 tail.
     *
     * One unclaimed batch of capacity is retained per dW2 cluster, capped by
     * the physical grid and floored by ``first_cluster``. Since ``task_cursor``
     * only advances, the retained prefix can only shrink. A stale acquire load
     * is conservative: it retains a cluster for one more batch boundary.
     */
    CUTLASS_DEVICE bool requested(
            const uint32_t cluster_idx, const uint32_t* task_cursor,
            const uint32_t task_limit, const uint32_t batch_tasks,
            const uint32_t num_clusters) const {
        DG_DEVICE_ASSERT(ready != nullptr);
        DG_DEVICE_ASSERT(first_cluster != 0u);
        DG_DEVICE_ASSERT(first_cluster < num_clusters);
        DG_DEVICE_ASSERT(cluster_idx < num_clusters);
        DG_DEVICE_ASSERT(task_cursor != nullptr);
        DG_DEVICE_ASSERT(batch_tasks != 0u);
        if (cluster_idx < first_cluster || ptx::ld_acq(ready) == 0u)
            return false;

        const uint32_t claimed = cute::min(
            ptx::ld_acq(task_cursor), task_limit);
        const uint32_t remaining = task_limit - claimed;
        const uint32_t tail_clusters =
            remaining / batch_tasks + (remaining % batch_tasks != 0u);
        const uint32_t retained_clusters = tail_clusters < first_cluster
            ? first_cluster : cute::min(tail_clusters, num_clusters);
        return cluster_idx >= retained_clusters;
    }
};

// Dynamically distribute compact K-grouped ranges among persistent two-CTA
// clusters while presenting one deterministic task stream to every scheduler
// role inside a cluster.  ``sm100_bf16_gemm_body`` owns independent provider
// instances in its TMA, MMA, and epilogue warps, so a global fetch-add from
// every instance would duplicate work.  Instead, the leader-CTA TMA role is
// the sole claimant and publishes one batch through a four-word per-cluster
// mailbox:
//
//   [0] acquire mask, [1] first logical task, [2] task count, [3] sequence.
//
// Each of the eleven consuming roles copies the payload into registers and
// release-ORs one unique bit.  The publisher may reuse the mailbox only after
// observing the complete mask with an acquire load.  A zero-sized published
// batch is the per-cluster terminal token.  This protocol permits clusters to
// join late, partitions [0, task_limit) exactly through one atomic claim per
// batch, and allocates no task descriptors.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          bool kTwoSegmentK = false,
          uint32_t kNumAuxiliarySchedulerWarps = 0u,
          uint32_t kValuePrefixWord = kPoolPrefixWord,
          uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          uint32_t kRetirementBias = 0u,
          uint32_t kReadinessPrefixWord = kPoolPrefixWord,
          uint32_t kValueAlignment = kSFKSpan,
          uint32_t kTmaTileK = kSFKSpan,
          bool kReadyFirstTaskClaim = false,
          bool kFeatureReadyFirstTaskClaim = false,
          uint32_t kFeatureReadyWord = 0u,
          bool kK3WgradPhaseTagged = false,
          uint32_t kK3DW13CursorWord = 0u,
          uint32_t kK3DW13FeatureDoneWord = 0u,
          uint32_t kK3DW13QuantDoneWord = 0u,
          bool kOneWayClusterTransition = false,
          bool kK3BF16WgradPhaseTagged = false>
struct ExternalKGroupedDynamicRangeProvider:
    private ExternalKGroupedOneWayClusterTransition<
        kOneWayClusterTransition> {
    using Transition = ExternalKGroupedOneWayClusterTransition<
        kOneWayClusterTransition>;
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kValueAlignment;
    static constexpr uint32_t kTaskScaleKSpan = kSFKSpan;
    static constexpr uint32_t kTaskTmaTileK = kTmaTileK;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = SHAPE_M;
    static constexpr uint32_t kTaskShapeN = SHAPE_N;
    static constexpr uint32_t kTaskPoolBlockRows = kPoolBlockRows;
    static constexpr bool kTaskHasTwoSegmentK = kTwoSegmentK;
    static constexpr uint32_t kTaskRetirementBias = kRetirementBias;
    static constexpr uint32_t kTaskReadinessPrefixWord =
        kReadinessPrefixWord;
    static constexpr uint32_t kTaskNumAuxiliarySchedulerWarps =
        kNumAuxiliarySchedulerWarps;
    static constexpr bool kTaskReadyFirstTaskClaim =
        kReadyFirstTaskClaim;
    static constexpr bool kTaskFeatureReadyFirstTaskClaim =
        kFeatureReadyFirstTaskClaim;
    static constexpr uint32_t kTaskFeatureReadyWord =
        kFeatureReadyWord;
    static constexpr bool kTaskPhaseTagged =
        kK3WgradPhaseTagged || kK3BF16WgradPhaseTagged;
    static constexpr bool kTaskBF16PhaseTagged =
        kK3BF16WgradPhaseTagged;
    static constexpr bool kTaskOneWayClusterTransition =
        kOneWayClusterTransition;
    static constexpr uint32_t kTaskPhaseBit = 0x80000000u;

    using Decoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes>;
    using TwoSegmentDecoder = ExternalKGroupedTwoSegmentRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes>;
    using K3DW13Decoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        6144u, 3584u, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes>;
    using K3DW13TwoSegmentDecoder =
        ExternalKGroupedTwoSegmentRangeDecoder<
            BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
            6144u, 3584u, kNum1DBlocksPerGroup,
            kPoolBlockRows, kSFKSpan,
            kPoolPrefixWord, kActiveExpertWord,
            kValuePrefixWord, kScalePrefixWord,
            kExplicitValueAndScalePrefixes>;

    static constexpr uint32_t kNumMBlocks = Decoder::kNumMBlocks;
    static constexpr uint32_t kNumNBlocks = Decoder::kNumNBlocks;
    static constexpr uint32_t kNumClusterTasksPerGroup =
        Decoder::kNumClusterTasksPerGroup;
    static constexpr uint32_t kK3DW13ClusterTasksPerGroup =
        K3DW13Decoder::kNumClusterTasksPerGroup;
    // Two TMA roles, one leader-CTA MMA role, optional auxiliary scheduler
    // warps in both CTAs, and one epilogue role per epilogue warp in each CTA.
    static constexpr uint32_t kNumSchedulerRoles =
        3u + 2u * kNumAuxiliarySchedulerWarps +
        2u * kNumEpilogueWarps;
    static constexpr uint32_t kCompleteAcquireMask =
        (1u << kNumSchedulerRoles) - 1u;

    static_assert(kNumMulticast == 2u,
                  "Dynamic K-grouped provider requires two-CTA clusters");
    static_assert(kNumSchedulerRoles < 32u,
                  "Dynamic K-grouped provider role mask exceeds u32");
    static_assert(
        2u + kNumAuxiliarySchedulerWarps <= kFirstEpilogueWarp,
        "Auxiliary scheduler warps overlap epilogue scheduler roles");
    static_assert(
        !kFeatureReadyFirstTaskClaim || kReadyFirstTaskClaim,
        "Feature readiness is a specialization of ready-first claims");
    static_assert(
        !kK3WgradPhaseTagged ||
            (kFeatureReadyFirstTaskClaim && !kTwoSegmentK &&
             SHAPE_M == 3584u && SHAPE_N == 3072u &&
             kK3DW13ClusterTasksPerGroup == 336u &&
             kK3DW13CursorWord != kK3DW13FeatureDoneWord &&
             kK3DW13FeatureDoneWord != kK3DW13QuantDoneWord),
        "Phase-tagged K3 wgrad requires exact disjoint dW2/dW13 state");
    static_assert(
        !kK3BF16WgradPhaseTagged ||
            (kTwoSegmentK && !kReadyFirstTaskClaim &&
             !kFeatureReadyFirstTaskClaim &&
             SHAPE_M == 3584u && SHAPE_N == 3072u &&
             Decoder::kNumClusterTasksPerGroup == 168u &&
             K3DW13TwoSegmentDecoder::kNumClusterTasksPerGroup == 336u),
        "Phase-tagged BF16 wgrad requires exact K3 two-segment geometry");
    static_assert(
        !(kK3WgradPhaseTagged && kK3BF16WgradPhaseTagged),
        "Exact and BF16 phase-tagged providers are mutually exclusive");
    static_assert(
        !kOneWayClusterTransition ||
            (kTwoSegmentK && !kReadyFirstTaskClaim &&
             !kTaskPhaseTagged),
        "One-way cluster transition is restricted to ordinary two-segment queues");

    int current_iter = -1;

    uint32_t num_blocks_in_group = kNumMulticast;
    bool is_peer_cta_alive = true;

    uint32_t current_group_idx = 0;
    uint32_t current_shape_k = 0;
    uint32_t current_k_cumsum = 0;
    uint32_t current_sf_k_cumsum = 0;
    uint32_t current_first_segment_shape_k = 0;
    uint32_t current_first_segment_scale_rows = 0;
    uint32_t current_second_segment_k_cumsum = 0;
    uint32_t current_second_segment_sf_k_cumsum = 0;
    uint32_t current_wgrad_phase = 0;

    const uint32_t* overlap_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;
    uint32_t batch_tasks;
    uint32_t tasks_per_expert;
    const uint32_t* expert_retired_counts;
    uint32_t retirements_per_pool_block;
    const uint32_t* state_ready_epoch;
    uint32_t expected_state_epoch;
    const uint32_t* published_num_tasks;

    uint32_t batch_sequence = 0;
    uint32_t batch_first = 0;
    uint32_t batch_count = 0;
    uint32_t batch_offset = 0;
    bool terminal = false;

    CUTLASS_DEVICE explicit ExternalKGroupedDynamicRangeProvider(
            const ExternalKGroupedRangeStream& stream):
        ExternalKGroupedDynamicRangeProvider(stream, nullptr, nullptr, 0u) {
        static_assert(
            !kTwoSegmentK,
            "Two-segment provider requires its two-segment stream wrapper");
    }

    CUTLASS_DEVICE explicit ExternalKGroupedDynamicRangeProvider(
            const ExternalKGroupedTwoSegmentRangeStream& stream):
        ExternalKGroupedDynamicRangeProvider(
            stream.first_segment, stream.second_segment_state_words,
            stream.transition_ready, stream.first_transition_cluster) {
        static_assert(
            kTwoSegmentK,
            "Ordinary provider requires its single-range stream");
    }

    CUTLASS_DEVICE explicit ExternalKGroupedDynamicRangeProvider(
            const ExternalKGroupedRangeStream& stream,
            const uint32_t* second_segment_state_words,
            const uint32_t* transition_ready,
            const uint32_t first_transition_cluster):
        Transition(transition_ready, first_transition_cluster),
        overlap_state_words(stream.overlap_state_words),
        second_segment_state_words(second_segment_state_words),
        task_cursor(stream.task_cursor),
        task_limit(stream.task_limit),
        cluster_mailbox(stream.cluster_mailbox),
        batch_tasks(stream.batch_tasks),
        tasks_per_expert(stream.tasks_per_expert),
        expert_retired_counts(stream.expert_retired_counts),
        retirements_per_pool_block(
            stream.retirements_per_pool_block),
        state_ready_epoch(stream.state_ready_epoch),
        expected_state_epoch(stream.expected_state_epoch),
        published_num_tasks(stream.published_num_tasks) {
        DG_DEVICE_ASSERT(task_cursor != nullptr);
        DG_DEVICE_ASSERT(cluster_mailbox != nullptr);
        DG_DEVICE_ASSERT(batch_tasks != 0u);
        DG_DEVICE_ASSERT(tasks_per_expert != 0u);
        if constexpr (kTwoSegmentK)
            DG_DEVICE_ASSERT(second_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(
            published_num_tasks == nullptr || state_ready_epoch != nullptr);
    }

    CUTLASS_DEVICE explicit ExternalKGroupedDynamicRangeProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream):
        ExternalKGroupedDynamicRangeProvider(
            ExternalKGroupedRangeStreamAccessor<kTwoSegmentK>::first(
                opaque_task_stream),
            ExternalKGroupedRangeStreamAccessor<kTwoSegmentK>::second(
                opaque_task_stream),
            ExternalKGroupedRangeStreamAccessor<kTwoSegmentK>::transition_ready(
                opaque_task_stream),
            ExternalKGroupedRangeStreamAccessor<
                kTwoSegmentK>::first_transition_cluster(
                    opaque_task_stream)) {}

    CUTLASS_DEVICE uint32_t get_scheduler_role_bit(
            const uint32_t warp_idx,
            const uint32_t cluster_rank) const {
        if (warp_idx == 0u)
            return 1u << cluster_rank;
        if (warp_idx == 1u) {
            DG_DEVICE_ASSERT(cluster_rank == 0u);
            return 1u << 2u;
        }
        constexpr uint32_t kFirstAuxiliaryWarp = 2u;
        if (warp_idx >= kFirstAuxiliaryWarp &&
            warp_idx <
                kFirstAuxiliaryWarp + kNumAuxiliarySchedulerWarps) {
            const uint32_t auxiliary_warp =
                warp_idx - kFirstAuxiliaryWarp;
            return 1u <<
                (3u + cluster_rank * kNumAuxiliarySchedulerWarps +
                 auxiliary_warp);
        }
        DG_DEVICE_ASSERT(
            warp_idx >= kFirstEpilogueWarp &&
            warp_idx < kFirstEpilogueWarp + kNumEpilogueWarps);
        const uint32_t epilogue_warp =
            warp_idx - kFirstEpilogueWarp;
        return 1u <<
            (3u + 2u * kNumAuxiliarySchedulerWarps +
             cluster_rank * kNumEpilogueWarps + epilogue_warp);
    }

    CUTLASS_DEVICE bool acquire_next_batch() {
        // TMA calls the provider from one elected lane, so the convergent
        // ``canonical_warp_idx_sync`` helper is invalid here.  The BF16 body
        // uses the canonical CUDA thread layout without warp remapping; the
        // non-synchronizing index is therefore the exact role warp ID.
        const uint32_t warp_idx = cutlass::canonical_warp_idx();
        const uint32_t lane_idx = threadIdx.x & 31u;
        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const uint32_t active_mask = __activemask();
        const int elected_lane = __ffs(static_cast<int>(active_mask)) - 1;
        const bool elected = lane_idx == static_cast<uint32_t>(elected_lane);
        const bool publisher =
            elected && cluster_rank == 0u && warp_idx == 0u;
        const uint32_t next_sequence = batch_sequence + 1u;

        uint32_t first = 0u;
        uint32_t count = 0u;
        if (publisher) {
            if (state_ready_epoch != nullptr) {
                while (ptx::ld_acq(state_ready_epoch) !=
                       expected_state_epoch) {
                    __nanosleep(64);
                }
                // Range-state arenas are reused.  Loading the count before
                // acquiring this generation can pair a fresh cursor/prefix
                // with the preceding range's task limit.
                if (published_num_tasks != nullptr)
                    task_limit = ptx::ld_acq(published_num_tasks);
            }
            while (ptx::ld_acq(cluster_mailbox) !=
                   kCompleteAcquireMask) {
                __nanosleep(64);
            }

            // No role can reread the previous payload after its release ack.
            // Reset the mask before publishing the next sequence token.
            cluster_mailbox[0] = 0u;
            const auto readiness_target = [&](const uint32_t expert) {
                const uint32_t first_readiness_units =
                    overlap_state_words[
                        kReadinessPrefixWord + expert + 1u] -
                    overlap_state_words[
                        kReadinessPrefixWord + expert];
                // A two-segment stream treats this optional counter as one
                // aggregate retirement count spanning both K slices. Set it
                // null when the union generation itself proves both operands
                // ready.
                const uint32_t second_readiness_units =
                    kTwoSegmentK
                    ? second_segment_state_words[
                          kReadinessPrefixWord + expert + 1u] -
                          second_segment_state_words[
                              kReadinessPrefixWord + expert]
                    : 0u;
                return external_k_grouped_retirement_target(
                    first_readiness_units, second_readiness_units,
                    retirements_per_pool_block, kRetirementBias);
            };

            const uint32_t cluster_idx =
                static_cast<uint32_t>(blockIdx.x) / kNumMulticast;
            if (Transition::requested(
                    cluster_idx, task_cursor, task_limit, batch_tasks,
                    kNumSMs / kNumMulticast)) {
                // Sticky local terminal: every scheduler role in both CTAs
                // consumes this same zero-count mailbox payload. Other
                // clusters retain ownership of the unchanged global cursor.
                count = 0u;
            } else if constexpr (kK3WgradPhaseTagged) {
                // One mailbox arbitrates both exact wgrad phases.  The high
                // bit of word one tags dW13; the remaining bits retain the
                // ordinary phase-local logical task.  Quantization producers
                // never write this mailbox after their sticky dW2 handoff.
                DG_DEVICE_ASSERT(expert_retired_counts != nullptr);
                DG_DEVICE_ASSERT(task_limit % tasks_per_expert == 0u);
                DG_DEVICE_ASSERT(tasks_per_expert == 168u);
                DG_DEVICE_ASSERT(batch_tasks % 4u == 0u);
                constexpr uint32_t kDW13TasksPerExpert = 336u;
                constexpr uint32_t kDW13FeaturePanels = 76u;
                const uint32_t active_count =
                    task_limit / tasks_per_expert;
                DG_DEVICE_ASSERT(
                    active_count <=
                    kTaskPhaseBit / kDW13TasksPerExpert);
                const uint32_t cluster_idx =
                    static_cast<uint32_t>(blockIdx.x) / kNumMulticast;
                const uint32_t scan_start = active_count == 0u
                    ? 0u
                    : (cluster_idx + batch_sequence) % active_count;
                auto* const dw2_cursors = const_cast<uint32_t*>(
                    expert_retired_counts);
                auto* const dw13_cursors = const_cast<uint32_t*>(
                    overlap_state_words + kK3DW13CursorWord);

                while (active_count != 0u) {
                    bool all_dw13_claimed = true;
                    bool claimed = false;

                    // Drain every newly ready dW13 quantum before claiming
                    // more dW2.  This is the bounded feedback edge that turns
                    // dW2 retirement into useful dW13 UMMA instead of a
                    // producer-only prefix followed by a terminal suffix.
                    for (uint32_t scan = 0u;
                         scan < active_count; ++scan) {
                        const uint32_t active_expert =
                            (scan_start + scan) % active_count;
                        const uint32_t expert = overlap_state_words[
                            kActiveExpertWord + active_expert];
                        const uint32_t expected = readiness_target(expert);
                        DG_DEVICE_ASSERT(expected != 0u);
                        const uint32_t terminal_value =
                            expected + kDW13TasksPerExpert;
                        uint32_t observed = ptx::ld_acq(
                            dw13_cursors + expert);
                        DG_DEVICE_ASSERT(observed <= terminal_value);
                        if (observed < terminal_value)
                            all_dw13_claimed = false;
                        while (observed >= expected &&
                               observed < terminal_value) {
                            const uint32_t local_first = observed - expected;
                            count = cute::min(
                                batch_tasks,
                                kDW13TasksPerExpert - local_first);
                            const uint32_t completed_panels = ptx::ld_acq(
                                overlap_state_words +
                                kK3DW13QuantDoneWord + expert);
                            DG_DEVICE_ASSERT(
                                completed_panels <= kDW13FeaturePanels);
                            if (completed_panels < kDW13FeaturePanels) {
                                const uint32_t scale_rows =
                                    overlap_state_words[
                                        kScalePrefixWord + expert + 1u] -
                                    overlap_state_words[
                                        kScalePrefixWord + expert];
                                const auto* const feature_done =
                                    overlap_state_words +
                                    kK3DW13FeatureDoneWord +
                                    expert * kDW13FeaturePanels;
                                uint32_t ready_count = 0u;
                                while (ready_count < count) {
                                    const auto panels =
                                        external_k_grouped_k3_dw13_feature_panels(
                                            local_first + ready_count);
                                    if (ptx::ld_acq(
                                            feature_done + panels.a_first) <
                                            scale_rows ||
                                        ptx::ld_acq(
                                            feature_done + panels.a_second) <
                                            scale_rows ||
                                        ptx::ld_acq(
                                            feature_done + panels.b_first) <
                                            scale_rows ||
                                        ptx::ld_acq(
                                            feature_done + panels.b_second) <
                                            scale_rows) {
                                        break;
                                    }
                                    ++ready_count;
                                }
                                count = ready_count & ~3u;
                                if (count == 0u)
                                    break;
                            }
                            const uint32_t previous = atomicCAS(
                                dw13_cursors + expert,
                                observed, observed + count);
                            if (previous == observed) {
                                first = kTaskPhaseBit |
                                    (active_expert * kDW13TasksPerExpert +
                                     local_first);
                                claimed = true;
                                break;
                            }
                            observed = previous;
                            DG_DEVICE_ASSERT(observed <= terminal_value);
                        }
                        if (claimed)
                            break;
                    }
                    if (claimed)
                        break;

                    bool all_dw2_claimed = true;
                    for (uint32_t scan = 0u;
                         scan < active_count; ++scan) {
                        const uint32_t active_expert =
                            (scan_start + scan) % active_count;
                        const uint32_t expert = overlap_state_words[
                            kActiveExpertWord + active_expert];
                        uint32_t observed = ptx::ld_acq(
                            dw2_cursors + expert);
                        DG_DEVICE_ASSERT(observed <= tasks_per_expert);
                        if (observed < tasks_per_expert)
                            all_dw2_claimed = false;
                        while (observed < tasks_per_expert) {
                            const uint32_t local_first = observed;
                            count = cute::min(
                                batch_tasks,
                                tasks_per_expert - local_first);
                            const auto* const ready_masks =
                                overlap_state_words +
                                kFeatureReadyWord + expert * 2u;
                            const uint32_t ready_lo =
                                ptx::ld_acq(ready_masks);
                            const uint32_t ready_hi =
                                ptx::ld_acq(ready_masks + 1u);
                            uint32_t ready_count = 0u;
                            while (ready_count < count &&
                                   external_k_grouped_k3_dw2_task_is_feature_ready(
                                       local_first + ready_count,
                                       ready_lo, ready_hi)) {
                                ++ready_count;
                            }
                            count = ready_count & ~3u;
                            if (count == 0u)
                                break;
                            const uint32_t previous = atomicCAS(
                                dw2_cursors + expert,
                                observed, observed + count);
                            if (previous == observed) {
                                first = active_expert * tasks_per_expert +
                                    local_first;
                                claimed = true;
                                break;
                            }
                            observed = previous;
                            DG_DEVICE_ASSERT(observed <= tasks_per_expert);
                        }
                        if (claimed)
                            break;
                    }
                    if (claimed)
                        break;
                    if (all_dw2_claimed && all_dw13_claimed) {
                        count = 0u;
                        break;
                    }
                    __nanosleep(64);
                }
            } else if constexpr (kK3BF16WgradPhaseTagged) {
                // One work-conserving mailbox owns both two-segment BF16
                // weight-gradient phases. W13 tasks are claimed only after
                // the final packed-weight reader release-publishes the exact
                // per-expert target. If no W13 quantum is ready, the same
                // cluster claims ordinary dW2 work from the existing global
                // cursor. A phase switch can therefore occur only at a
                // complete four-task mailbox boundary.
                DG_DEVICE_ASSERT(expert_retired_counts != nullptr);
                DG_DEVICE_ASSERT(task_limit % tasks_per_expert == 0u);
                DG_DEVICE_ASSERT(tasks_per_expert == 168u);
                DG_DEVICE_ASSERT(batch_tasks % 4u == 0u);
                constexpr uint32_t kDW13TasksPerExpert = 336u;
                const uint32_t active_count =
                    task_limit / tasks_per_expert;
                DG_DEVICE_ASSERT(
                    active_count <=
                    kTaskPhaseBit / kDW13TasksPerExpert);
                const uint32_t scan_start = active_count == 0u
                    ? 0u
                    : (cluster_idx + batch_sequence) % active_count;
                auto* const dw13_cursors = const_cast<uint32_t*>(
                    expert_retired_counts);

                while (true) {
                    bool all_dw13_claimed = true;
                    bool claimed = false;

                    // Prioritize newly ready dW13 quanta. This shortens the
                    // terminal dW13 tail without reserving a cluster that
                    // could otherwise make forward progress on dW2.
                    for (uint32_t scan = 0u;
                         scan < active_count; ++scan) {
                        const uint32_t active_expert =
                            (scan_start + scan) % active_count;
                        const uint32_t expert = overlap_state_words[
                            kActiveExpertWord + active_expert];
                        const uint32_t expected =
                            readiness_target(expert);
                        DG_DEVICE_ASSERT(expected != 0u);
                        const uint32_t terminal_value =
                            expected + kDW13TasksPerExpert;
                        uint32_t observed = ptx::ld_acq(
                            expert_retired_counts + expert);
                        DG_DEVICE_ASSERT(observed <= terminal_value);
                        if (observed < terminal_value)
                            all_dw13_claimed = false;
                        while (observed >= expected &&
                               observed < terminal_value) {
                            const uint32_t local_first =
                                observed - expected;
                            count = cute::min(
                                batch_tasks,
                                kDW13TasksPerExpert - local_first);
                            const uint32_t previous = atomicCAS(
                                dw13_cursors + expert,
                                observed, observed + count);
                            if (previous == observed) {
                                first = kTaskPhaseBit |
                                    (active_expert *
                                         kDW13TasksPerExpert +
                                     local_first);
                                claimed = true;
                                break;
                            }
                            observed = previous;
                            DG_DEVICE_ASSERT(
                                observed <= terminal_value);
                        }
                        if (claimed)
                            break;
                    }
                    if (claimed)
                        break;

                    // BF16 dW2 operands are immutable and fully prepared at
                    // suffix entry, so its existing monotonically increasing
                    // cursor is sufficient. Keep a claim inside one expert
                    // and one four-task phase-restoring quantum.
                    bool all_dw2_claimed = false;
                    while (true) {
                        first = ptx::ld_acq(task_cursor);
                        if (first >= task_limit) {
                            all_dw2_claimed = true;
                            count = 0u;
                            break;
                        }
                        const uint32_t expert_end =
                            (first / tasks_per_expert + 1u) *
                            tasks_per_expert;
                        count = cute::min(
                            batch_tasks,
                            cute::min(task_limit - first,
                                      expert_end - first));
                        if (atomicCAS(
                                task_cursor, first, first + count) ==
                            first) {
                            claimed = true;
                            break;
                        }
                    }
                    if (claimed)
                        break;
                    if (all_dw2_claimed && all_dw13_claimed) {
                        count = 0u;
                        break;
                    }
                    // dW2 may be fully claimed while W13 dgrad is still
                    // publishing expert readiness. Do not emit a sticky
                    // terminal token until both phase-local queues retire.
                    __nanosleep(64);
                }
            } else if constexpr (kReadyFirstTaskClaim) {
                // A global cursor lets clusters reserve future experts and
                // then park on their readiness counters.  For a streaming
                // producer that creates severe head-of-line blocking.  Once
                // an expert reaches its exact readiness target, no producer
                // may modify that word again, so reuse the same word as the
                // expert-local task cursor and claim only ready work.
                DG_DEVICE_ASSERT(expert_retired_counts != nullptr);
                DG_DEVICE_ASSERT(task_limit % tasks_per_expert == 0u);
                DG_DEVICE_ASSERT(batch_tasks % 4u == 0u);
                DG_DEVICE_ASSERT(tasks_per_expert % 4u == 0u);
                static_assert(
                    !kFeatureReadyFirstTaskClaim ||
                        (!kTwoSegmentK && kNumMulticast == 2u &&
                         kIsMulticastOnA && BLOCK_M == 256u &&
                         BLOCK_N == 128u && SHAPE_M == 3584u &&
                         SHAPE_N == 3072u &&
                         kNum1DBlocksPerGroup == 16u),
                    "Feature-ready claims require exact K3 dW2 geometry");
                const uint32_t active_count =
                    task_limit / tasks_per_expert;
                const uint32_t cluster_idx =
                    static_cast<uint32_t>(blockIdx.x) / kNumMulticast;
                const uint32_t scan_start = active_count == 0u
                    ? 0u
                    : (cluster_idx + batch_sequence) % active_count;
                auto* mutable_retired_counts = const_cast<uint32_t*>(
                    expert_retired_counts);

                while (active_count != 0u) {
                    bool all_claimed = true;
                    bool claimed = false;
                    for (uint32_t scan = 0u; scan < active_count; ++scan) {
                        const uint32_t active_expert =
                            (scan_start + scan) % active_count;
                        const uint32_t expert = overlap_state_words[
                            kActiveExpertWord + active_expert];
                        const uint32_t expected =
                            kFeatureReadyFirstTaskClaim
                            ? 0u : readiness_target(expert);
                        if constexpr (!kFeatureReadyFirstTaskClaim)
                            DG_DEVICE_ASSERT(expected != 0u);
                        const uint32_t terminal_value =
                            expected + tasks_per_expert;
                        uint32_t observed = ptx::ld_acq(
                            expert_retired_counts + expert);
                        DG_DEVICE_ASSERT(observed <= terminal_value);
                        if (observed < terminal_value)
                            all_claimed = false;
                        while (observed < terminal_value) {
                            if constexpr (!kFeatureReadyFirstTaskClaim) {
                                if (observed < expected)
                                    break;
                            }
                            const uint32_t local_first =
                                observed - expected;
                            count = cute::min(
                                batch_tasks,
                                tasks_per_expert - local_first);
                            if constexpr (kFeatureReadyFirstTaskClaim) {
                                const auto* const ready_masks =
                                    overlap_state_words +
                                    kFeatureReadyWord + expert * 2u;
                                const uint32_t ready_lo =
                                    ptx::ld_acq(ready_masks);
                                const uint32_t ready_hi =
                                    ptx::ld_acq(ready_masks + 1u);
                                uint32_t ready_count = 0u;
                                while (ready_count < count &&
                                       external_k_grouped_k3_dw2_task_is_feature_ready(
                                           local_first + ready_count,
                                           ready_lo, ready_hi)) {
                                    ++ready_count;
                                }
                                // Every embedded batch must restore both the
                                // mainloop and epilogue barrier phases.
                                count = ready_count & ~3u;
                                if (count == 0u)
                                    break;
                            }
                            const uint32_t previous = atomicCAS(
                                mutable_retired_counts + expert,
                                observed, observed + count);
                            if (previous == observed) {
                                first =
                                    active_expert * tasks_per_expert +
                                    local_first;
                                claimed = true;
                                break;
                            }
                            observed = previous;
                            DG_DEVICE_ASSERT(observed <= terminal_value);
                        }
                        if (claimed)
                            break;
                    }
                    if (claimed)
                        break;
                    if (all_claimed) {
                        count = 0u;
                        break;
                    }
                    __nanosleep(64);
                }
            } else {
                // Keep every claim inside one expert. Besides avoiding a
                // cross-expert batch, K3's task counts keep each resulting
                // range aligned to the four-task epilogue phase quantum.
                while (true) {
                    first = ptx::ld_acq(task_cursor);
                    if (first >= task_limit) {
                        count = 0u;
                        break;
                    }
                    const uint32_t expert_end =
                        (first / tasks_per_expert + 1u) *
                        tasks_per_expert;
                    count = cute::min(
                        batch_tasks,
                        cute::min(task_limit - first,
                                  expert_end - first));
                    if (atomicCAS(
                            task_cursor, first, first + count) == first)
                        break;
                }

                // Optional descriptor-readiness policy. dW13 supplies its
                // per-expert retirement counters; dW2 leaves the pointer null.
                // Only the publisher waits before exposing the batch sequence,
                // so every scheduler role observes the ready operands through
                // the same acquire edge.
                if (count != 0u && expert_retired_counts != nullptr) {
                    const uint32_t first_active_expert =
                        first / tasks_per_expert;
                    const uint32_t last_active_expert =
                        (first + count - 1u) / tasks_per_expert;
                    for (uint32_t active_expert = first_active_expert;
                         active_expert <= last_active_expert;
                         ++active_expert) {
                        const uint32_t expert = overlap_state_words[
                            kActiveExpertWord + active_expert];
                        const uint32_t expected = readiness_target(expert);
                        while (ptx::ld_acq(
                                   expert_retired_counts + expert) <
                               expected) {
                            __nanosleep(64);
                        }
                    }
                }
            }
            if constexpr (kK3BF16WgradPhaseTagged) {
                if (count != 0u &&
                    (first & kTaskPhaseBit) != 0u) {
                    // A tagged batch exists only after this publisher's
                    // acquire load/CAS has observed the exact per-expert W13
                    // retirement target.  Its phase-one D epilogue overwrites
                    // that retired W13-weight arena through TMA stores. Carry
                    // the generic acquire into the async proxy before the
                    // mailbox release publishes work to the remaining roles;
                    // a producer-side fence cannot order these later consumer
                    // async operations.
                    asm volatile(
                        "fence.proxy.async.global;" ::: "memory");
                }
            }
            cluster_mailbox[1] = first;
            cluster_mailbox[2] = count;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(cluster_mailbox + 3), "r"(next_sequence)
                : "memory");
        }

        if (elected) {
            while (ptx::ld_acq(cluster_mailbox + 3) !=
                   next_sequence) {
                __nanosleep(64);
            }
            first = cluster_mailbox[1];
            count = cluster_mailbox[2];

            // Payload reads happen-before this release acknowledgement.  The
            // publisher's acquire poll therefore protects mailbox reuse.
            ptx::red_or_rel_gpu(
                cluster_mailbox,
                get_scheduler_role_bit(warp_idx, cluster_rank));
        }

        first = __shfl_sync(active_mask, first, elected_lane);
        count = __shfl_sync(active_mask, count, elected_lane);
        batch_sequence = next_sequence;
        batch_first = first;
        batch_count = count;
        batch_offset = 0u;
        terminal = count == 0u;
        return !terminal;
    }

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        ++current_iter;
        if (terminal)
            return false;
        if (batch_offset == batch_count && !acquire_next_batch())
            return false;

        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const auto range_idx = batch_offset++;
        if constexpr (kTaskPhaseTagged) {
            current_wgrad_phase =
                (batch_first & kTaskPhaseBit) != 0u ? 1u : 0u;
            const uint32_t phase_first =
                batch_first & ~kTaskPhaseBit;
            if constexpr (kK3BF16WgradPhaseTagged) {
                ExternalKGroupedTwoSegmentRangeDecodedTask decoded{};
                if (current_wgrad_phase == 0u) {
                    decoded = TwoSegmentDecoder::decode_range_task(
                        overlap_state_words,
                        second_segment_state_words,
                        phase_first, range_idx, cluster_rank);
                } else {
                    decoded = K3DW13TwoSegmentDecoder::decode_range_task(
                        overlap_state_words,
                        second_segment_state_words,
                        phase_first, range_idx, cluster_rank);
                }
                const auto& task = decoded.output_task;
                current_group_idx = task.group_idx;
                current_shape_k = task.shape_k;
                current_k_cumsum = task.k_cumsum;
                current_sf_k_cumsum = task.sf_k_cumsum;
                current_first_segment_shape_k =
                    decoded.first_segment_shape_k;
                current_first_segment_scale_rows =
                    decoded.first_segment_scale_rows;
                current_second_segment_k_cumsum =
                    decoded.second_segment_k_cumsum;
                current_second_segment_sf_k_cumsum =
                    decoded.second_segment_sf_k_cumsum;
                DG_DEVICE_ASSERT(
                    external_k_grouped_two_segment_is_tile_aligned(
                        decoded.first_segment_shape_k,
                        decoded.second_segment_shape_k,
                        kTmaTileK));
                num_blocks_in_group =
                    task.num_blocks_in_swizzle_group;
                m_block_idx = task.m_block_idx;
                n_block_idx = task.n_block_idx;
            } else {
                ExternalKGroupedRangeDecodedTask task{};
                if (current_wgrad_phase == 0u) {
                    task = Decoder::decode_range_task(
                        overlap_state_words, phase_first,
                        range_idx, cluster_rank);
                } else {
                    task = K3DW13Decoder::decode_range_task(
                        overlap_state_words, phase_first,
                        range_idx, cluster_rank);
                }
                current_group_idx = task.group_idx;
                current_shape_k = task.shape_k;
                current_k_cumsum = task.k_cumsum;
                current_sf_k_cumsum = task.sf_k_cumsum;
                num_blocks_in_group =
                    task.num_blocks_in_swizzle_group;
                m_block_idx = task.m_block_idx;
                n_block_idx = task.n_block_idx;
            }
        } else if constexpr (kTwoSegmentK) {
            const auto decoded = TwoSegmentDecoder::decode_range_task(
                overlap_state_words, second_segment_state_words,
                batch_first, range_idx, cluster_rank);
            const auto& task = decoded.output_task;
            current_group_idx = task.group_idx;
            current_shape_k = task.shape_k;
            current_k_cumsum = task.k_cumsum;
            current_sf_k_cumsum = task.sf_k_cumsum;
            current_first_segment_shape_k =
                decoded.first_segment_shape_k;
            current_first_segment_scale_rows =
                decoded.first_segment_scale_rows;
            current_second_segment_k_cumsum =
                decoded.second_segment_k_cumsum;
            current_second_segment_sf_k_cumsum =
                decoded.second_segment_sf_k_cumsum;
            DG_DEVICE_ASSERT(
                external_k_grouped_two_segment_is_tile_aligned(
                    decoded.first_segment_shape_k,
                    decoded.second_segment_shape_k,
                    kTmaTileK));
            num_blocks_in_group = task.num_blocks_in_swizzle_group;
            m_block_idx = task.m_block_idx;
            n_block_idx = task.n_block_idx;
        } else {
            const auto task = Decoder::decode_range_task(
                overlap_state_words, batch_first,
                range_idx, cluster_rank);
            current_group_idx = task.group_idx;
            current_shape_k = task.shape_k;
            current_k_cumsum = task.k_cumsum;
            current_sf_k_cumsum = task.sf_k_cumsum;
            num_blocks_in_group = task.num_blocks_in_swizzle_group;
            m_block_idx = task.m_block_idx;
            n_block_idx = task.n_block_idx;
        }
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0) const {
        if constexpr (
                kTwoSegmentK && kIndexType == IndexType::K) {
            static_assert(
                kWithGroupOffset,
                "Two-segment K operands require grouped physical offsets");
            return external_k_grouped_two_segment_physical_index(
                block_idx * block_size,
                current_first_segment_shape_k,
                current_k_cumsum,
                current_second_segment_k_cumsum);
        } else if constexpr (
                kTwoSegmentK && kIndexType == IndexType::SF_K) {
            static_assert(
                kWithGroupOffset,
                "Two-segment SF_K operands require grouped physical offsets");
            return external_k_grouped_two_segment_physical_index(
                block_idx * block_size,
                current_first_segment_scale_rows,
                current_sf_k_cumsum,
                current_second_segment_sf_k_cumsum);
        }
        uint32_t offset = 0;
        if constexpr (kWithGroupOffset) {
            if constexpr (kIndexType == IndexType::MN) {
                // The grouped B operand advances by N while the grouped D
                // output advances by M.  Both enter through the historical
                // MN index kind, so recover the axis from the exact static
                // phase-zero dimensions instead of applying one stride to
                // both descriptors.  The phase-tagged K3 specializations
                // require distinct 3584x3072 phase-zero dimensions above.
                if constexpr (kTaskPhaseTagged) {
                    DG_DEVICE_ASSERT(
                        shape_dim == SHAPE_M || shape_dim == SHAPE_N);
                }
                const uint32_t phase_shape_dim =
                    kTaskPhaseTagged && current_wgrad_phase != 0u
                    ? (shape_dim == SHAPE_M ? 6144u : 3584u)
                    : shape_dim;
                offset = current_group_idx * phase_shape_dim;
            } else if constexpr (kIndexType == IndexType::K) {
                offset = current_k_cumsum;
            } else if constexpr (kIndexType == IndexType::SF_K) {
                offset = current_sf_k_cumsum;
            }
        }
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return true;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

// Lightweight dynamic provider for a terminal K-grouped range whose operands
// are already immutable.  The general dynamic provider above also supports
// generation handoffs, readiness counters, two-segment K, feature readiness,
// and phase-tagged streams.  Carrying that state through every BF16 scheduler
// role needlessly raises the enclosing megakernel's register/stack pressure
// when none of those policies can occur.  This specialization retains only a
// fixed-size atomic claim and the same eleven-role cluster mailbox protocol.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kBatchTasks, uint32_t kTasksPerExpert,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          bool kPairAdjacentN = false>
struct ExternalKGroupedTerminalDynamicRangeProvider {
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kSFKSpan;
    static constexpr uint32_t kTaskScaleKSpan = kSFKSpan;
    static constexpr uint32_t kTaskTmaTileK = kSFKSpan;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = SHAPE_M;
    static constexpr uint32_t kTaskShapeN = SHAPE_N;
    static constexpr uint32_t kTaskPoolBlockRows = kPoolBlockRows;
    static constexpr bool kTaskPairedN = kPairAdjacentN;

    using Decoder = ExternalKGroupedRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kPoolPrefixWord, kPoolPrefixWord, false,
        kPairAdjacentN>;

    // Two TMA roles, the leader-CTA MMA role, and four epilogue roles in
    // each CTA consume each published batch.
    static constexpr uint32_t kNumSchedulerRoles =
        3u + 2u * kNumEpilogueWarps;
    static constexpr uint32_t kCompleteAcquireMask =
        (1u << kNumSchedulerRoles) - 1u;

    static_assert(kNumMulticast == 2u,
                  "terminal dynamic provider requires two-CTA clusters");
    static_assert(kNumSMs % kNumMulticast == 0u,
                  "terminal dynamic provider requires complete clusters");
    static_assert(kBatchTasks != 0u && kBatchTasks % 4u == 0u,
                  "terminal batches must reset BF16 epilogue phases");
    static_assert(kTasksPerExpert % kBatchTasks == 0u,
                  "terminal batches must not cross experts");
    static_assert(kTasksPerExpert == Decoder::kNumClusterTasksPerGroup,
                  "terminal task count must match paired-N decode geometry");

    int current_iter = -1;
    uint32_t current_group_idx = 0u;
    uint32_t current_shape_k = 0u;
    uint32_t current_k_cumsum = 0u;
    uint32_t current_sf_k_cumsum = 0u;

    const uint32_t* state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;

    uint32_t batch_sequence = 0u;
    uint32_t batch_first = 0u;
    uint32_t batch_count = 0u;
    uint32_t batch_offset = 0u;
    bool terminal = false;

    CUTLASS_DEVICE explicit ExternalKGroupedTerminalDynamicRangeProvider(
            const ExternalKGroupedRangeStream& stream):
        state_words(stream.overlap_state_words),
        task_cursor(stream.task_cursor),
        task_limit(stream.task_limit),
        cluster_mailbox(stream.cluster_mailbox) {
        DG_DEVICE_ASSERT(state_words != nullptr);
        DG_DEVICE_ASSERT(task_cursor != nullptr);
        DG_DEVICE_ASSERT(cluster_mailbox != nullptr);
        DG_DEVICE_ASSERT(stream.batch_tasks == kBatchTasks);
        DG_DEVICE_ASSERT(stream.tasks_per_expert == kTasksPerExpert);
        DG_DEVICE_ASSERT(task_limit % kBatchTasks == 0u);
        DG_DEVICE_ASSERT(stream.expert_retired_counts == nullptr);
        DG_DEVICE_ASSERT(stream.state_ready_epoch == nullptr);
        DG_DEVICE_ASSERT(stream.published_num_tasks == nullptr);
    }

    CUTLASS_DEVICE explicit ExternalKGroupedTerminalDynamicRangeProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream):
        ExternalKGroupedTerminalDynamicRangeProvider(
            ExternalKGroupedRangeStreamAccessor<false>::first(
                opaque_task_stream)) {}

    CUTLASS_DEVICE uint32_t get_scheduler_role_bit(
            const uint32_t warp_idx,
            const uint32_t cluster_rank) const {
        if (warp_idx == 0u)
            return 1u << cluster_rank;
        if (warp_idx == 1u) {
            DG_DEVICE_ASSERT(cluster_rank == 0u);
            return 1u << 2u;
        }
        DG_DEVICE_ASSERT(
            warp_idx >= kFirstEpilogueWarp &&
            warp_idx < kFirstEpilogueWarp + kNumEpilogueWarps);
        return 1u <<
            (3u + cluster_rank * kNumEpilogueWarps +
             warp_idx - kFirstEpilogueWarp);
    }

    CUTLASS_DEVICE bool acquire_next_batch() {
        const uint32_t warp_idx = cutlass::canonical_warp_idx();
        const uint32_t lane_idx = threadIdx.x & 31u;
        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const uint32_t active_mask = __activemask();
        const int elected_lane = __ffs(static_cast<int>(active_mask)) - 1;
        const bool elected =
            lane_idx == static_cast<uint32_t>(elected_lane);
        const bool publisher =
            elected && cluster_rank == 0u && warp_idx == 0u;
        const uint32_t next_sequence = batch_sequence + 1u;

        uint32_t first = 0u;
        uint32_t count = 0u;
        if (publisher) {
            while (ptx::ld_acq(cluster_mailbox) !=
                   kCompleteAcquireMask) {
                __nanosleep(64);
            }
            cluster_mailbox[0] = 0u;

            // The task limit and every expert span are exact multiples of the
            // fixed quantum, so one atomic add cannot split an expert or
            // publish a partial nonterminal batch.
            first = atomicAdd(task_cursor, kBatchTasks);
            count = first < task_limit ? kBatchTasks : 0u;
            cluster_mailbox[1] = first;
            cluster_mailbox[2] = count;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(cluster_mailbox + 3), "r"(next_sequence)
                : "memory");
        }

        if (elected) {
            while (ptx::ld_acq(cluster_mailbox + 3) !=
                   next_sequence) {
                __nanosleep(64);
            }
            first = cluster_mailbox[1];
            count = cluster_mailbox[2];
            ptx::red_or_rel_gpu(
                cluster_mailbox,
                get_scheduler_role_bit(warp_idx, cluster_rank));
        }

        first = __shfl_sync(active_mask, first, elected_lane);
        count = __shfl_sync(active_mask, count, elected_lane);
        batch_sequence = next_sequence;
        batch_first = first;
        batch_count = count;
        batch_offset = 0u;
        terminal = count == 0u;
        return !terminal;
    }

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        ++current_iter;
        if (terminal)
            return false;
        if (batch_offset == batch_count && !acquire_next_batch())
            return false;

        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const auto task = Decoder::decode_range_task(
            state_words, batch_first, batch_offset++, cluster_rank);
        current_group_idx = task.group_idx;
        current_shape_k = task.shape_k;
        current_k_cumsum = task.k_cumsum;
        current_sf_k_cumsum = task.sf_k_cumsum;
        m_block_idx = task.m_block_idx;
        n_block_idx = task.n_block_idx;
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0u) const {
        uint32_t offset = 0u;
        if constexpr (kWithGroupOffset) {
            if constexpr (kIndexType == IndexType::MN)
                offset = current_group_idx * shape_dim;
            else if constexpr (kIndexType == IndexType::K)
                offset = current_k_cumsum;
            else if constexpr (kIndexType == IndexType::SF_K)
                offset = current_sf_k_cumsum;
        }
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return true;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

// Dynamic counterpart of ``ExternalKGroupedTwoSegmentRangeProvider``. The
// leader TMA role acquires the union generation before reading either prefix,
// then release-publishes each decoded batch through the existing cluster
// mailbox protocol. No task descriptors or auxiliary allocations are needed.
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          bool kOneWayClusterTransition = false,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          uint32_t kNumAuxiliarySchedulerWarps = 0u,
          uint32_t kValuePrefixWord = kPoolPrefixWord,
          uint32_t kScalePrefixWord = kPoolPrefixWord,
          bool kExplicitValueAndScalePrefixes = false,
          uint32_t kRetirementBias = 0u,
          uint32_t kReadinessPrefixWord = kPoolPrefixWord,
          uint32_t kValueAlignment = kSFKSpan,
          uint32_t kTmaTileK = kSFKSpan>
using ExternalKGroupedTwoSegmentDynamicRangeProvider =
    ExternalKGroupedDynamicRangeProvider<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        kNumSMs, SHAPE_M, SHAPE_N,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kFirstEpilogueWarp, kNumEpilogueWarps,
        kNum1DBlocksPerGroup, true,
        kNumAuxiliarySchedulerWarps,
        kValuePrefixWord, kScalePrefixWord,
        kExplicitValueAndScalePrefixes,
        kRetirementBias, kReadinessPrefixWord,
        kValueAlignment, kTmaTileK,
        false, false, 0u, false, 0u, 0u, 0u,
        kOneWayClusterTransition>;

/** Lightweight terminal provider for two immutable physical K segments.
 *
 * The logical task stream is identical to the one-range terminal provider:
 * one fixed-size atomic claim and one eleven-role cluster mailbox.  Decoding
 * additionally maps each logical K tile onto either absolute physical prefix,
 * so empty or skewed true-varlen segments need no descriptors or temporary
 * concatenation.  This provider is deliberately BF16-only and therefore
 * carries no scale-factor coordinates.
 */
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kBatchTasks, uint32_t kTasksPerExpert,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          bool kPairAdjacentN = false>
struct ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider {
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kSFKSpan;
    static constexpr uint32_t kTaskScaleKSpan = kSFKSpan;
    static constexpr uint32_t kTaskTmaTileK = kSFKSpan;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = SHAPE_M;
    static constexpr uint32_t kTaskShapeN = SHAPE_N;
    static constexpr uint32_t kTaskPoolBlockRows = kPoolBlockRows;
    static constexpr bool kTaskHasTwoSegmentK = true;
    static constexpr bool kTaskPairedN = kPairAdjacentN;

    using Decoder = ExternalKGroupedTwoSegmentRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kPoolPrefixWord, kPoolPrefixWord, false,
        kPairAdjacentN>;

    // Two TMA roles, the leader-CTA MMA role, and four epilogue roles in
    // each CTA consume each published batch.
    static constexpr uint32_t kNumSchedulerRoles =
        3u + 2u * kNumEpilogueWarps;
    static constexpr uint32_t kCompleteAcquireMask =
        (1u << kNumSchedulerRoles) - 1u;

    static_assert(kNumMulticast == 2u,
                  "terminal two-segment provider requires two-CTA clusters");
    static_assert(kNumSMs % kNumMulticast == 0u,
                  "terminal two-segment provider requires complete clusters");
    static_assert(kBatchTasks != 0u && kBatchTasks % 4u == 0u,
                  "terminal two-segment batches must reset BF16 epilogue phases");
    static_assert(kTasksPerExpert % kBatchTasks == 0u,
                  "terminal two-segment batches must not cross experts");
    static_assert(kTasksPerExpert == Decoder::kNumClusterTasksPerGroup,
                  "terminal two-segment task count must match paired-N decode geometry");
    static_assert(kPoolBlockRows % kSFKSpan == 0u,
                  "terminal two-segment prefixes must be TMA-tile aligned");

    int current_iter = -1;
    uint32_t current_group_idx = 0u;
    uint32_t current_shape_k = 0u;
    uint32_t current_k_cumsum = 0u;
    uint32_t current_first_segment_shape_k = 0u;
    uint32_t current_second_segment_k_cumsum = 0u;

    const uint32_t* first_segment_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;

    uint32_t batch_sequence = 0u;
    uint32_t batch_first = 0u;
    uint32_t batch_count = 0u;
    uint32_t batch_offset = 0u;
    bool terminal = false;

    CUTLASS_DEVICE explicit
    ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider(
            const ExternalKGroupedTerminalTwoSegmentRangeStream& stream):
        first_segment_state_words(stream.first_segment_state_words),
        second_segment_state_words(stream.second_segment_state_words),
        task_cursor(stream.task_cursor),
        task_limit(stream.task_limit),
        cluster_mailbox(stream.cluster_mailbox) {
        DG_DEVICE_ASSERT(first_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(second_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(task_cursor != nullptr);
        DG_DEVICE_ASSERT(cluster_mailbox != nullptr);
        DG_DEVICE_ASSERT(stream.batch_tasks == kBatchTasks);
        DG_DEVICE_ASSERT(stream.tasks_per_expert == kTasksPerExpert);
        DG_DEVICE_ASSERT(task_limit % kBatchTasks == 0u);
    }

    CUTLASS_DEVICE explicit
    ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream):
        ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider(
            *reinterpret_cast<const
                ExternalKGroupedTerminalTwoSegmentRangeStream*>(
                    opaque_task_stream)) {}

    CUTLASS_DEVICE uint32_t get_scheduler_role_bit(
            const uint32_t warp_idx,
            const uint32_t cluster_rank) const {
        if (warp_idx == 0u)
            return 1u << cluster_rank;
        if (warp_idx == 1u) {
            DG_DEVICE_ASSERT(cluster_rank == 0u);
            return 1u << 2u;
        }
        DG_DEVICE_ASSERT(
            warp_idx >= kFirstEpilogueWarp &&
            warp_idx < kFirstEpilogueWarp + kNumEpilogueWarps);
        return 1u <<
            (3u + cluster_rank * kNumEpilogueWarps +
             warp_idx - kFirstEpilogueWarp);
    }

    CUTLASS_DEVICE bool acquire_next_batch() {
        const uint32_t warp_idx = cutlass::canonical_warp_idx();
        const uint32_t lane_idx = threadIdx.x & 31u;
        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const uint32_t active_mask = __activemask();
        const int elected_lane = __ffs(static_cast<int>(active_mask)) - 1;
        const bool elected =
            lane_idx == static_cast<uint32_t>(elected_lane);
        const bool publisher =
            elected && cluster_rank == 0u && warp_idx == 0u;
        const uint32_t next_sequence = batch_sequence + 1u;

        uint32_t first = 0u;
        uint32_t count = 0u;
        if (publisher) {
            while (ptx::ld_acq(cluster_mailbox) !=
                   kCompleteAcquireMask) {
                __nanosleep(64);
            }
            cluster_mailbox[0] = 0u;

            // Exact task/expert divisibility makes every nonterminal claim a
            // complete four-task quantum wholly owned by one expert.
            first = atomicAdd(task_cursor, kBatchTasks);
            count = first < task_limit ? kBatchTasks : 0u;
            cluster_mailbox[1] = first;
            cluster_mailbox[2] = count;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(cluster_mailbox + 3), "r"(next_sequence)
                : "memory");
        }

        if (elected) {
            while (ptx::ld_acq(cluster_mailbox + 3) !=
                   next_sequence) {
                __nanosleep(64);
            }
            first = cluster_mailbox[1];
            count = cluster_mailbox[2];
            ptx::red_or_rel_gpu(
                cluster_mailbox,
                get_scheduler_role_bit(warp_idx, cluster_rank));
        }

        first = __shfl_sync(active_mask, first, elected_lane);
        count = __shfl_sync(active_mask, count, elected_lane);
        batch_sequence = next_sequence;
        batch_first = first;
        batch_count = count;
        batch_offset = 0u;
        terminal = count == 0u;
        return !terminal;
    }

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        ++current_iter;
        if (terminal)
            return false;
        if (batch_offset == batch_count && !acquire_next_batch())
            return false;

        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const auto decoded = Decoder::decode_range_task(
            first_segment_state_words, second_segment_state_words,
            batch_first, batch_offset++, cluster_rank);
        const auto& task = decoded.output_task;
        current_group_idx = task.group_idx;
        current_shape_k = task.shape_k;
        current_k_cumsum = task.k_cumsum;
        current_first_segment_shape_k =
            decoded.first_segment_shape_k;
        current_second_segment_k_cumsum =
            decoded.second_segment_k_cumsum;
        DG_DEVICE_ASSERT(
            external_k_grouped_two_segment_is_tile_aligned(
                decoded.first_segment_shape_k,
                decoded.second_segment_shape_k,
                kSFKSpan));
        m_block_idx = task.m_block_idx;
        n_block_idx = task.n_block_idx;
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0u) const {
        static_assert(
            kIndexType != IndexType::SF_K,
            "terminal two-segment BF16 provider has no scale-factor operand");
        if constexpr (kIndexType == IndexType::K) {
            static_assert(
                kWithGroupOffset,
                "two-segment BF16 K operands require physical offsets");
            return external_k_grouped_two_segment_physical_index(
                block_idx * block_size,
                current_first_segment_shape_k,
                current_k_cumsum,
                current_second_segment_k_cumsum);
        }
        const uint32_t offset = kWithGroupOffset
            ? current_group_idx * shape_dim : 0u;
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return true;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

/** Lightweight terminal provider for three immutable physical K segments.
 *
 * One logical output tile traverses segment zero, one, then two while keeping
 * the same FP32 TMEM accumulator.  The ordinary BF16 TMA producer asks this
 * provider for each physical K coordinate, so no operand concatenation,
 * intermediate BF16 epilogue, range barrier, or extra allocation is needed.
 */
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kBatchTasks, uint32_t kTasksPerExpert,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kPhysicalRangeStateStride = 0u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          bool kPairAdjacentN = false>
struct ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider {
    static constexpr GemmType kTaskGemmType =
        GemmType::KGroupedContiguous;
    static constexpr uint32_t kTaskBlockM = BLOCK_M;
    static constexpr uint32_t kTaskBlockN = BLOCK_N;
    static constexpr uint32_t kTaskNumMulticast = kNumMulticast;
    static constexpr bool kTaskIsMulticastOnA = kIsMulticastOnA;
    static constexpr uint32_t kTaskKAlignment = kSFKSpan;
    static constexpr uint32_t kTaskScaleKSpan = kSFKSpan;
    static constexpr uint32_t kTaskTmaTileK = kSFKSpan;
    static constexpr uint32_t kTaskNumSMs = kNumSMs;
    static constexpr uint32_t kTaskShapeM = SHAPE_M;
    static constexpr uint32_t kTaskShapeN = SHAPE_N;
    static constexpr uint32_t kTaskPoolBlockRows = kPoolBlockRows;
    static constexpr bool kTaskHasThreeSegmentK = true;
    static constexpr bool kTaskPairedN = kPairAdjacentN;

    using Decoder = ExternalKGroupedThreeSegmentRangeDecoder<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        SHAPE_M, SHAPE_N, kNum1DBlocksPerGroup,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kPoolPrefixWord, kPoolPrefixWord, false,
        kPairAdjacentN>;

    // Two TMA roles, the leader-CTA MMA role, and four epilogue roles in
    // each CTA consume every published batch.
    static constexpr uint32_t kNumSchedulerRoles =
        3u + 2u * kNumEpilogueWarps;
    static constexpr uint32_t kCompleteAcquireMask =
        (1u << kNumSchedulerRoles) - 1u;

    static_assert(
        kNumMulticast == 2u,
        "terminal three-segment provider requires two-CTA clusters");
    static_assert(
        kNumSchedulerRoles < 32u,
        "terminal three-segment scheduler role mask exceeds u32");
    static_assert(
        kNumSMs % kNumMulticast == 0u,
        "terminal three-segment provider requires complete clusters");
    static_assert(
        kBatchTasks != 0u && kBatchTasks % 4u == 0u,
        "terminal three-segment batches must reset BF16 epilogue phases");
    static_assert(
        kTasksPerExpert % kBatchTasks == 0u,
        "terminal three-segment batches must not cross experts");
    static_assert(
        kTasksPerExpert == Decoder::kNumClusterTasksPerGroup,
        "terminal three-segment task count must match paired-N decode geometry");
    static_assert(
        kPoolBlockRows % kSFKSpan == 0u,
        "terminal three-segment prefixes must be TMA-tile aligned");
    static_assert(
        kPhysicalRangeStateStride != 0u,
        "terminal three-segment provider requires contiguous range state");

    int current_iter = -1;
    uint32_t current_group_idx = 0u;
    uint32_t current_shape_k = 0u;
    uint32_t current_k_cumsum = 0u;
    uint32_t current_first_segment_shape_k = 0u;
    uint32_t current_second_segment_shape_k = 0u;
    uint32_t current_second_segment_k_cumsum = 0u;
    uint32_t current_third_segment_k_cumsum = 0u;

    const uint32_t* first_segment_state_words;
    const uint32_t* second_segment_state_words;
    uint32_t* task_cursor;
    uint32_t task_limit;
    uint32_t* cluster_mailbox;

    uint32_t batch_sequence = 0u;
    uint32_t batch_first = 0u;
    uint32_t batch_count = 0u;
    uint32_t batch_offset = 0u;
    bool terminal = false;

    CUTLASS_DEVICE explicit
    ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider(
            const ExternalKGroupedTerminalThreeSegmentRangeStream& stream):
        first_segment_state_words(stream.first_segment_state_words),
        second_segment_state_words(stream.second_segment_state_words),
        task_cursor(stream.task_cursor),
        task_limit(stream.task_limit),
        cluster_mailbox(stream.cluster_mailbox) {
        DG_DEVICE_ASSERT(first_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(second_segment_state_words != nullptr);
        DG_DEVICE_ASSERT(task_cursor != nullptr);
        DG_DEVICE_ASSERT(cluster_mailbox != nullptr);
        DG_DEVICE_ASSERT(stream.batch_tasks == kBatchTasks);
        DG_DEVICE_ASSERT(stream.tasks_per_expert == kTasksPerExpert);
        DG_DEVICE_ASSERT(task_limit % kBatchTasks == 0u);
    }

    CUTLASS_DEVICE explicit
    ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider(
            const uint32_t&, const uint32_t&, const uint32_t&,
            int* opaque_task_stream):
        ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider(
            *reinterpret_cast<const
                ExternalKGroupedTerminalThreeSegmentRangeStream*>(
                    opaque_task_stream)) {}

    CUTLASS_DEVICE uint32_t get_scheduler_role_bit(
            const uint32_t warp_idx,
            const uint32_t cluster_rank) const {
        if (warp_idx == 0u)
            return 1u << cluster_rank;
        if (warp_idx == 1u) {
            DG_DEVICE_ASSERT(cluster_rank == 0u);
            return 1u << 2u;
        }
        DG_DEVICE_ASSERT(
            warp_idx >= kFirstEpilogueWarp &&
            warp_idx < kFirstEpilogueWarp + kNumEpilogueWarps);
        return 1u <<
            (3u + cluster_rank * kNumEpilogueWarps +
             warp_idx - kFirstEpilogueWarp);
    }

    CUTLASS_DEVICE bool acquire_next_batch() {
        const uint32_t warp_idx = cutlass::canonical_warp_idx();
        const uint32_t lane_idx = threadIdx.x & 31u;
        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const uint32_t active_mask = __activemask();
        const int elected_lane = __ffs(static_cast<int>(active_mask)) - 1;
        const bool elected =
            lane_idx == static_cast<uint32_t>(elected_lane);
        const bool publisher =
            elected && cluster_rank == 0u && warp_idx == 0u;
        const uint32_t next_sequence = batch_sequence + 1u;

        uint32_t first = 0u;
        uint32_t count = 0u;
        if (publisher) {
            while (ptx::ld_acq(cluster_mailbox) !=
                   kCompleteAcquireMask) {
                __nanosleep(64);
            }
            cluster_mailbox[0] = 0u;
            first = atomicAdd(task_cursor, kBatchTasks);
            count = first < task_limit ? kBatchTasks : 0u;
            cluster_mailbox[1] = first;
            cluster_mailbox[2] = count;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(cluster_mailbox + 3), "r"(next_sequence)
                : "memory");
        }

        if (elected) {
            while (ptx::ld_acq(cluster_mailbox + 3) !=
                   next_sequence) {
                __nanosleep(64);
            }
            first = cluster_mailbox[1];
            count = cluster_mailbox[2];
            ptx::red_or_rel_gpu(
                cluster_mailbox,
                get_scheduler_role_bit(warp_idx, cluster_rank));
        }

        first = __shfl_sync(active_mask, first, elected_lane);
        count = __shfl_sync(active_mask, count, elected_lane);
        batch_sequence = next_sequence;
        batch_first = first;
        batch_count = count;
        batch_offset = 0u;
        terminal = count == 0u;
        return !terminal;
    }

    CUTLASS_DEVICE bool get_next_block(
            uint32_t& m_block_idx, uint32_t& n_block_idx) {
        ++current_iter;
        if (terminal)
            return false;
        if (batch_offset == batch_count && !acquire_next_batch())
            return false;

        const uint32_t cluster_rank =
            static_cast<uint32_t>(blockIdx.x) % kNumMulticast;
        const auto decoded = Decoder::decode_range_task(
            first_segment_state_words,
            second_segment_state_words,
            second_segment_state_words - kPhysicalRangeStateStride,
            batch_first, batch_offset++, cluster_rank);
        const auto& task = decoded.output_task;
        current_group_idx = task.group_idx;
        current_shape_k = task.shape_k;
        current_k_cumsum = task.k_cumsum;
        current_first_segment_shape_k =
            decoded.first_segment_shape_k;
        current_second_segment_shape_k =
            decoded.second_segment_shape_k;
        current_second_segment_k_cumsum =
            decoded.second_segment_k_cumsum;
        current_third_segment_k_cumsum =
            decoded.third_segment_k_cumsum;
        DG_DEVICE_ASSERT(
            external_k_grouped_three_segment_is_tile_aligned(
                decoded.first_segment_shape_k,
                decoded.second_segment_shape_k,
                decoded.third_segment_shape_k,
                kSFKSpan));
        m_block_idx = task.m_block_idx;
        n_block_idx = task.n_block_idx;
        return true;
    }

    template <bool kWithGroupOffset,
              IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
            const uint32_t shape_dim, const uint32_t block_size,
            const uint32_t& block_idx, const uint32_t& = 0u) const {
        static_assert(
            kIndexType != IndexType::SF_K,
            "terminal three-segment BF16 provider has no scale-factor operand");
        if constexpr (kIndexType == IndexType::K) {
            static_assert(
                kWithGroupOffset,
                "three-segment BF16 K operands require physical offsets");
            return external_k_grouped_three_segment_physical_index(
                block_idx * block_size,
                current_first_segment_shape_k,
                current_second_segment_shape_k,
                current_k_cumsum,
                current_second_segment_k_cumsum,
                current_third_segment_k_cumsum);
        }
        const uint32_t offset = kWithGroupOffset
            ? current_group_idx * shape_dim : 0u;
        return offset + block_idx * block_size;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(
            const uint32_t&) const {
        return BLOCK_M;
    }

    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t&) const {
        return true;
    }

    CUTLASS_DEVICE bool is_computation_valid(
            const uint32_t&, const uint32_t&) const {
        return true;
    }
};

/** Exact Kimi K3 two-segment BF16 dW2/dW13 work-conserving provider.
 *
 * One global dW2 cursor and the per-expert W13 retirement/cursor plane feed a
 * single cluster mailbox.  The mailbox high bit selects dW13's descriptor and
 * 6144x3584 output geometry; an untagged task selects dW2's 3584x3072
 * geometry.  Both phases preserve segment-zero then segment-one accumulation
 * in one FP32 accumulator, and every claim is a four-task phase-restoring
 * quantum.  The specialization deliberately exposes no general shape knobs:
 * its scheduler protocol is valid only behind the exact K3 host gate.
 */
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs, false>()>
using ExternalKGroupedK3TwoSegmentBF16WgradDynamicRangeProvider =
    ExternalKGroupedDynamicRangeProvider<
        BLOCK_M, BLOCK_N, 2u, false,
        kNumSMs, 3584u, 3072u,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kFirstEpilogueWarp, kNumEpilogueWarps,
        kNum1DBlocksPerGroup, true,
        0u, kPoolPrefixWord, kPoolPrefixWord, false,
        0u, kPoolPrefixWord,
        kSFKSpan, kSFKSpan,
        false, false, 0u,
        false, 0u, 0u, 0u,
        false, true>;

/** Ready-first dynamic provider for a fixed-order two-segment K reduction.
 *
 * The optional retirement plane is both the producer credit and, after the
 * exact aggregate target is reached, the expert-local task cursor.  A task is
 * therefore never reserved ahead of its operands, while each claimed output
 * tile still traverses segment zero followed by segment one in one FP32
 * accumulator.  The provider reuses the ordinary four-word cluster mailbox
 * and does not require a task list or a global workspace allocation.
 */
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t SHAPE_M, uint32_t SHAPE_N,
          uint32_t kPoolBlockRows = 192u,
          uint32_t kSFKSpan = 64u,
          uint32_t kPoolPrefixWord = 31u,
          uint32_t kActiveExpertWord = 144u,
          uint32_t kFirstEpilogueWarp = 4u,
          uint32_t kNumEpilogueWarps = 4u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<
                  GemmType::KGroupedContiguous,
                  BLOCK_M, BLOCK_N, kNumSMs,
                  kIsMulticastOnA>(),
          uint32_t kRetirementBias = 0u,
          uint32_t kReadinessPrefixWord = kPoolPrefixWord,
          uint32_t kValueAlignment = kSFKSpan,
          uint32_t kTmaTileK = kSFKSpan>
using ExternalKGroupedTwoSegmentReadyDynamicRangeProvider =
    ExternalKGroupedDynamicRangeProvider<
        BLOCK_M, BLOCK_N, kNumMulticast, kIsMulticastOnA,
        kNumSMs, SHAPE_M, SHAPE_N,
        kPoolBlockRows, kSFKSpan,
        kPoolPrefixWord, kActiveExpertWord,
        kFirstEpilogueWarp, kNumEpilogueWarps,
        kNum1DBlocksPerGroup, true,
        0u, kPoolPrefixWord, kPoolPrefixWord, false,
        kRetirementBias, kReadinessPrefixWord,
        kValueAlignment, kTmaTileK,
        true, false, 0u, false, 0u, 0u, 0u, false>;

#pragma clang diagnostic pop

} // namespace deep_gemm::sched
