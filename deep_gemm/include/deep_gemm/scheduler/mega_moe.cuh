#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0,
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks, uint32_t kNumThreads,
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
    const layout::SymBuffer<kNumRanks>& sym_buffer;
    uint32_t sm_idx, thread_idx;

    // Scheduler state
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0;
    uint32_t current_num_tokens = 0;
    uint32_t current_pool_block_offset = 0;
    uint32_t block_idx = 0;
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;

    // Cycle chunking: exclusive pool-block end of the current dispatch cycle. The scheduler
    // stops iterating and clamps the last expert's m-block count at this bound. Default is
    // unbounded (the whole pool) so num_cycles == 1 reproduces the original behavior.
    uint32_t cycle_pool_block_end = 0xffffffffu;
    // Cycle chunking: per-cycle pool-block capacity (kNumRingBlocks) when chunking is enabled;
    // 0 = single-cycle path (original behavior).
    uint32_t cap_blocks = 0u;
    // Set by fetch_next_l1/l2_block when it returns false BECAUSE the next block is beyond the
    // cycle end (vs. because the wave's blocks are done). get_next_block uses this to break the
    // cycle instead of transitioning L1->L2 / advancing the wave.
    bool cycle_ended = false;
    // Counted-arrival mbarrier for the cycle boundary (tolerates the pipelined for_each_block
    // roles arriving at different times, unlike bar.sync which needs all threads simultaneously).
    // Set by the kernel (`set_cycle_barrier`) to `&shared_storage.cycle_barrier` (init kNumThreads).
    cutlass::arch::ClusterTransactionBarrier* cycle_barrier_ptr = nullptr;
    uint32_t cycle_barrier_phase = 0;
    CUTLASS_DEVICE void set_cycle_barrier(cutlass::arch::ClusterTransactionBarrier* b) { cycle_barrier_ptr = b; }

    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace,
                                             const layout::SymBuffer<kNumRanks>& sym_buffer,
                                             const uint32_t& sm_idx, const uint32_t& thread_idx)
        : workspace(workspace), sym_buffer(sym_buffer), sm_idx(sm_idx), thread_idx(thread_idx) {
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

    CUTLASS_DEVICE void set_cycle_pool_block_end(const uint32_t& end) { cycle_pool_block_end = end; }
    CUTLASS_DEVICE void set_cap_blocks(const uint32_t& cap) { cap_blocks = cap; }

    // Cycle-boundary barrier: sync all warp roles within the SM, then cross-rank via NVLink.
    // All threads (kNumThreads) of every SM must call this once per cycle, in lockstep.
    // Indices/tags are local constexprs (NVCC can't ODR-use static constexpr members in device code).
    CUTLASS_DEVICE void cycle_barrier() {
        // Cycle boundary: (1) intra-CTA counted-arrival mbarrier (all kNumThreads threads,
        // tolerates the pipelined for_each_block roles arriving at different times), then
        // (2) cross-CTA cluster_sync (the 2-CTA cluster). For multi-rank, a cross-rank NVLink
        // barrier is also needed (TODO when testing multi-rank). No-op if no barrier is set.
        if (cycle_barrier_ptr == nullptr) return;
        ptx::mbarrier_arrive(cycle_barrier_ptr);
        ptx::mbarrier_wait_and_flip_phase(cycle_barrier_ptr, cycle_barrier_phase);
        comm::cluster_sync_with_relaxed_arrive();
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
            // Cycle chunking: stop at the cycle's pool-block end (block-aligned). The block at
            // `current_pool_block_offset + m_block_idx` is beyond this cycle -> return false so
            // the cycle loop drains + barriers, then the next cycle continues this expert.
            // No-op when `cycle_pool_block_end` is unbounded (num_cycles == 1).
            if (current_pool_block_offset + m_block_idx >= cycle_pool_block_end) {
                cycle_ended = true;
                return false;
            }
            if (m_block_idx < num_m_blocks) {
                cycle_ended = false;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL1BlockNs;
            advance_expert_idx();
        }
        cycle_ended = false;
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL2BlockNs;
            // Cycle chunking: stop at the cycle's pool-block end (see fetch_next_l1_block).
            if (current_pool_block_offset + m_block_idx >= cycle_pool_block_end) {
                cycle_ended = true;
                return false;
            }
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                cycle_ended = false;
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;
            advance_expert_idx();
        }
        cycle_ended = false;
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;
            // Cycle chunking: stop once we've reached the current cycle's pool-block end.
            // No-op when `cycle_pool_block_end` is unbounded (num_cycles == 1).
            if (current_pool_block_offset >= cycle_pool_block_end)
                break;

            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) {
                    // Found a new L1 block
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // fetch returned false: either the wave's L1 is done, or the cycle ended
                    // (cycle_ended flag distinguishes — set by the fetch's cycle-pool-block check).
                    if (cycle_ended)
                        break;  // cycle ended — stop, don't transition to L2
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
                    if (cycle_ended)
                        break;  // cycle ended — stop, don't advance to the next wave
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

    template <typename Func, typename BarrierFunc>
    CUTLASS_DEVICE void for_each_block_impl(Func&& func, BarrierFunc&& cycle_barrier, const uint32_t& cap_blocks) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Cycle chunking: split the pool into `num_cycles` chunks of `cap_blocks` pool blocks
        // each (the ring holds one cycle). `cap_blocks == 0` is the single-cycle sentinel
        // (eff_cap = total_pool_blocks -> num_cycles == 1, reproducing the original behavior).
        const uint32_t total_pool_blocks = get_pool_block_offset(kNumExpertsPerRank);
        const uint32_t eff_cap_blocks = (cap_blocks == 0u) ? total_pool_blocks : cap_blocks;
        const uint32_t num_cycles = total_pool_blocks == 0 ? 1u
            : (total_pool_blocks + eff_cap_blocks - 1) / eff_cap_blocks;

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        for (uint32_t cycle = 0; cycle < num_cycles; ++ cycle) {
            // Only activate the cycle-pool-block bound when actually chunking; the single-cycle
            // path (cap_blocks == 0) leaves `cycle_pool_block_end` unbounded = original behavior.
            if (cap_blocks != 0u)
                set_cycle_pool_block_end(cute::min((cycle + 1) * eff_cap_blocks, total_pool_blocks));
            // `block_idx` (and thus expert-relative `m_block_idx`) carries across cycles so
            // `get_valid_m` stays correct; the cycle end is enforced inside fetch_next_l1/l2_block.
            while (true) {
                CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
                if (block_phase == BlockPhase::None)
                    break;

                func(block_phase, current_local_expert_idx,
                     block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                     m_block_idx, n_block_idx);
            }
            // Cycle boundary: sync all warp roles + ranks (caller-supplied). MUST be called
            // every cycle by every thread (even SMs with no blocks this cycle) so the barrier
            // gets all kNumThreads arrivals — do NOT break early when an SM finishes its blocks.
            cycle_barrier();
        }
    }

    // Original single-cycle entry point. Uses the chunking path (with the cross-warp/cross-rank
    // cycle barrier) when `cap_blocks > 0` has been set; otherwise the single-cycle no-op path
    // (original behavior). Callers (the 5 warp-role for_each_block sites) are unchanged.
    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        if (cap_blocks > 0u) {
            for_each_block_impl(std::forward<Func>(func), [this]() { cycle_barrier(); }, cap_blocks);
        } else {
            struct NoOpBarrier { CUTLASS_DEVICE void operator()() const {} };
            for_each_block_impl(std::forward<Func>(func), NoOpBarrier{}, 0u);
        }
    }
};

} // namespace deep_gemm::sched
