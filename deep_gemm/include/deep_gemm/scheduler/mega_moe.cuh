#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

// Fixed training reverse for the GLM large-M specialization. It is the same
// two-phase wave state machine as MegaMoEScheduler, with immutable expert
// counts saved by forward instead of live dispatch counters. Reverse L1 is W2
// dgrad; its fused epilogue publishes quantized dpreact into the L2 ring, and
// reverse L2 is W13 dgrad. Ring readiness counters carry the dependency, so
// CTAs never serialize phases behind a grid-wide barrier.
enum class BackwardBlockPhase {
    None = 0,
    W2Dgrad = 1,
    W13Dgrad = 2,
    // Legacy helper selector only. The persistent scheduler never emits a
    // recompute phase; forward-saved BF16 preactivation is authoritative.
    RecomputeW13 = 3,
};

template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExpertsPerRank, uint32_t kNumExpertsPerWave,
    uint32_t kNumSMs,
    uint32_t kNumExpertsPerLane =
        math::constexpr_ceil_div(kNumExpertsPerRank, 32u)>
struct MegaMoEBackwardScheduler {
    static constexpr uint32_t kW2DgradBlockNs =
        kIntermediateHidden / BLOCK_N;
    static constexpr uint32_t kW13DgradBlockNs =
        kHidden / BLOCK_N;
    static constexpr uint32_t kW2DgradBlockKs = kHidden / BLOCK_K;
    static constexpr uint32_t kW13DgradBlockKs =
        (2 * kIntermediateHidden) / BLOCK_K;

    DG_STATIC_ASSERT(kNumSMs % 2 == 0,
                     "Backward 2-CTA scheduler requires an even SM count");
    DG_STATIC_ASSERT(kW2DgradBlockNs % 2 == 0 &&
                         kW13DgradBlockNs % 2 == 0,
                     "Every backward phase must assign adjacent N blocks to a cluster");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 &&
                         kNumExpertsPerWave <= kNumExpertsPerRank,
                     "Invalid backward wave size");

    const int* expert_counts;
    BackwardBlockPhase phase = BackwardBlockPhase::W2Dgrad;
    uint32_t current_local_expert_idx = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t current_num_tokens = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEBackwardScheduler(const int* counts)
        : expert_counts(counts), block_idx(blockIdx.x) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            const uint32_t expert_idx = i * 32 + ptx::get_lane_idx();
            stored_num_tokens_per_expert[i] =
                expert_idx < kNumExpertsPerRank
                    ? static_cast<uint32_t>(__ldg(expert_counts + expert_idx))
                    : 0u;
        }
        __syncwarp();
        set_expert_idx(0);
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        return cute::min(
            math::align(current_local_expert_idx + 1,
                        kNumExpertsPerWave),
            kNumExpertsPerRank);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(
        const uint32_t expert_idx) const {
        uint32_t value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            if (expert_idx == i * 32 + ptx::get_lane_idx())
                value = stored_num_tokens_per_expert[i];
        }
        return ptx::exchange(value, expert_idx % 32);
    }

    CUTLASS_DEVICE uint32_t get_pool_block_offset(
        const uint32_t expert_idx) const {
        uint32_t blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx) {
                blocks += math::ceil_div(
                    stored_num_tokens_per_expert[i], BLOCK_M);
            }
        }
        return __reduce_add_sync(0xffffffff, blocks);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        ++current_local_expert_idx;
        if (current_local_expert_idx < kNumExpertsPerRank)
            current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto value = cute::min(
            current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(value, 16u) : value;
    }

    CUTLASS_DEVICE bool fetch_next_w2_block() {
        const uint32_t wave_end = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end) {
            const uint32_t num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kW2DgradBlockNs;
            if (m_block_idx < num_m_blocks)
                return true;
            block_idx -= num_m_blocks * kW2DgradBlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_w13_block() {
        const uint32_t wave_end = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end) {
            const uint32_t num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kW13DgradBlockNs) {
                m_block_idx = block_idx / kW13DgradBlockNs;
                return true;
            }
            block_idx -= num_m_blocks * kW13DgradBlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE cute::tuple<BackwardBlockPhase, uint32_t, uint32_t, uint32_t>
    get_next_block() {
        while (current_local_expert_idx < kNumExpertsPerRank) {
            if (phase == BackwardBlockPhase::W2Dgrad) {
                if (fetch_next_w2_block()) {
                    n_block_idx =
                        block_idx - m_block_idx * kW2DgradBlockNs;
                    block_idx += kNumSMs;
                    return {phase, current_local_expert_idx,
                            m_block_idx, n_block_idx};
                }
                phase = BackwardBlockPhase::W13Dgrad;
                set_expert_idx(math::align<uint32_t, false>(
                    current_local_expert_idx - 1,
                    kNumExpertsPerWave));
            } else if (fetch_next_w13_block()) {
                n_block_idx =
                    block_idx - m_block_idx * kW13DgradBlockNs;
                block_idx += kNumSMs;
                return {phase, current_local_expert_idx,
                        m_block_idx, n_block_idx};
            } else {
                phase = BackwardBlockPhase::W2Dgrad;
            }
        }
        return {BackwardBlockPhase::None, 0, 0, 0};
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        while (true) {
            CUTE_TIE_DECL(get_next_block(), current_phase, expert_idx,
                          current_m_block_idx, current_n_block_idx);
            if (current_phase == BackwardBlockPhase::None)
                break;
            const uint32_t num_k_blocks =
                current_phase == BackwardBlockPhase::W2Dgrad
                    ? kW2DgradBlockKs
                    : kW13DgradBlockKs;
            func(current_phase, expert_idx, num_k_blocks,
                 current_m_block_idx, current_n_block_idx);
        }
    }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerWave > 0 and kNumExpertsPerWave <= kNumExpertsPerRank, "Invalid wave config");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const layout::Workspace& workspace;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        // Align up to wave boundary, clamped for the last partial wave
        const auto aligned = math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        return cute::min(aligned, kNumExpertsPerRank);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                m_block_idx = block_idx / kNumL2BlockNs;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    next_phase = BlockPhase::Linear2;
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }
            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }

        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched
