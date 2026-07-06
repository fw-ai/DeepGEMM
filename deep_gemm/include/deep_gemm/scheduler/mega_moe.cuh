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
    // Init to 1: the mbarrier starts at parity 0, and `try_wait.parity(phase)` returns when
    // parity == phase. To BLOCK until the first round of kNumThreads arrivals flips parity 0->1,
    // the first wait must pass phase=1 (not 0, which would match the init parity and return
    // immediately). `mbarrier_wait_and_flip_phase` flips this each call (1<->0).
    uint32_t cycle_barrier_phase = 1;
    CUTLASS_DEVICE void set_cycle_barrier(cutlass::arch::ClusterTransactionBarrier* b) { cycle_barrier_ptr = b; }

    // Current dispatch cycle index (set by for_each_block_impl per cycle). Read by the kernel's
    // per-block lambdas to gate debug printf on cycle >= 1 (avoid flooding cycle 0's blocks).
    uint32_t current_cycle = 0;
    CUTLASS_DEVICE uint32_t get_current_cycle() const { return current_cycle; }

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

    // Cycle-boundary barrier, three steps:
    //  (1) Within-SM counted-arrival mbarrier: all 16 warps (5 pipelined for_each_block roles +
    //      the dispatch pull) arrive once. Tolerates staggered arrivals (unlike bar.sync). This
    //      guarantees the SM's GEMM+pull for this cycle are done before signalling cross-SM.
    //  (2) Cross-SM grid_sync driven by ONLY warp 0 (one atomic-add per SM), so all SMs reach the
    //      same cycle boundary before any SM starts the next cycle's pull. Without this, SMs
    //      desync across cycles and the global ring full/empty counts deadlock.
    //  (3) Within-SM mbarrier again: warps 1-15 wait for warp 0's grid_sync to complete before
    //      proceeding to the next cycle.
    // Each thread calls this once per cycle (via its role's for_each_block / the pull loop). Only
    // warp 0 does the cross-SM atomic; the others skip (2) and wait at (3).
    CUTLASS_DEVICE void cycle_barrier() {
        if (cycle_barrier_ptr == nullptr) return;
        const uint32_t warp_idx_local = thread_idx / 32;
        if (sm_idx == 0u && ptx::get_lane_idx() == 0u && (warp_idx_local == 0u || warp_idx_local == 4u))
            printf("[cb] c=%u w=%u pre-mbar ph=%u\n", current_cycle, warp_idx_local, cycle_barrier_phase);
        // (1) Within-SM counted-arrival mbarrier: all 16 warps arrive ONCE per cycle.
        ptx::mbarrier_arrive(cycle_barrier_ptr);
        ptx::mbarrier_wait_and_flip_phase(cycle_barrier_ptr, cycle_barrier_phase);
        if (sm_idx == 0u && ptx::get_lane_idx() == 0u && (warp_idx_local == 0u || warp_idx_local == 4u))
            printf("[cb] c=%u w=%u post-mbar ph=%u\n", current_cycle, warp_idx_local, cycle_barrier_phase);
        // (2) Cross-SM grid sync, driven by warp 0 only.
        if (warp_idx_local == 0u) {
            if (sm_idx == 0u && ptx::get_lane_idx() == 0u)
                printf("[cb] c=%u w=0 pre-grid\n", current_cycle);
            constexpr uint32_t kCycleGridSyncIndex = 2u;   // 0=dispatch, 1=epilogue, 2=cycle
            constexpr uint32_t kCycleBarrierIdx = 8u;      // free named barrier (0-2, 3+ epilogue WG)
            comm::grid_sync<kNumSMs, kCycleGridSyncIndex>(
                workspace, sm_idx, thread_idx,
                [=]() { ptx::sync_aligned(32u, kCycleBarrierIdx); });
            if (sm_idx == 0u && ptx::get_lane_idx() == 0u)
                printf("[cb] c=%u w=0 post-grid\n", current_cycle);
        }
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            // Cycle chunking: if this expert's pool-block start is already at/after the cycle end,
            // the cycle is done for this SM. This stops the consume-walk from overshooting the
            // cycle boundary when the stripe counter (`block_idx`) has no real block in this cycle
            // (e.g. SM 0 got one block early, then the consume-walk advances through experts whose
            // pool range is beyond this cycle). No-op when unbounded (num_cycles == 1).
            if (current_pool_block_offset >= cycle_pool_block_end) {
                cycle_ended = true;
                return false;
            }
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL1BlockNs;
            if (m_block_idx < num_m_blocks) {
                // Real block within the expert — check it's within the cycle's pool-block range.
                // No-op when `cycle_pool_block_end` is unbounded (num_cycles == 1).
                if (current_pool_block_offset + m_block_idx >= cycle_pool_block_end) {
                    cycle_ended = true;
                    return false;
                }
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
            // Cycle chunking: see fetch_next_l1_block.
            if (current_pool_block_offset >= cycle_pool_block_end) {
                cycle_ended = true;
                return false;
            }
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kNumL2BlockNs;
            if (block_idx < num_m_blocks * kNumL2BlockNs) {
                if (current_pool_block_offset + m_block_idx >= cycle_pool_block_end) {
                    cycle_ended = true;
                    return false;
                }
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
            current_cycle = cycle;
            // Only activate the cycle-pool-block bound when actually chunking; the single-cycle
            // path (cap_blocks == 0) leaves `cycle_pool_block_end` unbounded = original behavior.
            if (cap_blocks != 0u)
                set_cycle_pool_block_end(cute::min((cycle + 1) * eff_cap_blocks, total_pool_blocks));
            // Reset the persistent stripe counter per cycle. The expert/pool-offset state carries
            // across cycles (so we resume at the right expert), but `block_idx` must restart at
            // `blockIdx.x` for each cycle's range — otherwise the leftover stripe value from the
            // previous cycle's consume-walk yields a bogus large `m_block_idx` that trips the
            // cycle-end check before any real block is found (GEMM did no work for cycle >= 1).
            block_idx = blockIdx.x;
            // The cycle end is enforced inside fetch_next_l1/l2_block.
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

    // Single-cycle iteration: iterate blocks up to `cycle_pool_block_end` (set by the caller),
    // NO internal cycle loop, NO cycle_barrier. Used by the "caller loops over cycles" chunking
    // pattern so the caller can reset the pipeline state (stage_idx/phase/current_iter_idx)
    // between cycles. Calls fetch_expert_recv_count + set_expert_idx(0) once.
    template <typename Func>
    CUTLASS_DEVICE void for_each_block_single_cycle(Func&& func) {
        fetch_expert_recv_count();
        set_expert_idx(0);
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;
            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }

    CUTLASS_DEVICE uint32_t get_total_pool_blocks() {
        return get_pool_block_offset(kNumExpertsPerRank);
    }
    CUTLASS_DEVICE uint32_t get_num_cycles() {
        const auto total = get_total_pool_blocks();
        const auto cap = (cap_blocks == 0u) ? total : cap_blocks;
        return total == 0 ? 1u : (total + cap - 1) / cap;
    }
};

} // namespace deep_gemm::sched
