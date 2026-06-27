#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

// Standalone packed MXFP4 x MXFP4 (E2M1 data, UE8M0 SF, gran-K 32), 2-CTA UMMA.
//
// NOTES: de-risking vehicle for the packed `mxf4` 2-CTA path before porting it
// into the mega-MoE kernel. This deliberately mirrors the proven
// `sm100_fp8_fp4_gemm_1d1d` kernel structure (scheduler, warp roles, barriers,
// SF warp-transpose + with-SF barrier, and the swap-AB BF16 epilogue), changing
// ONLY what packed `mxf4` requires:
//   - both operands packed E2M1 (2 elems/byte), byte-addressed smem (`/2`).
//   - `UMMA_K = 64` and the `tcgen05.mma.kind::mxf4` 2-CTA wrapper.
//   - UE8M0 SF fixed at gran-K 32 (one gran-32 SF per 32 K, `sf_id = k * 2`).
//   - K-major packed swizzle == `BLOCK_K / 2` bytes; UMMA descriptors via
//     `make_smem_desc` with `BLOCK_K/2` byte strides (mirrors `sm100_fp4_mqa_logits`).
// Spots needing on-SM100 confirmation are marked `// VALIDATE`.

namespace deep_gemm {

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs,
          // NVFP4 (E4M3/UE4M3 SF gran-16 + per-tensor global scale) vs MXFP4 (UE8M0 SF gran-32)
          bool kIsNVFP4 = false>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_mxfp4_gemm_impl(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_a,    // acts,    [M, K] K-major (packed E2M1)
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,  // acts SF (int32-packed): UE8M0 gran-32 (mxfp4) / E4M3 gran-16 (nvfp4)
                      const __grid_constant__ cute::TmaDescriptor tensor_map_b,    // weights, [N, K] K-major (packed E2M1)
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,  // weights SF
                      const __grid_constant__ cute::TmaDescriptor tensor_map_cd,   // out,     [M, N] BF16
                      // NVFP4 output dequant scale = gs_a * gs_b (CPU scalar; 1.0 for MXFP4)
                      const float ab_global_scale = 1.0f) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Packed FP4 data type (2 elements per byte) and UE8M0 scale factor
    using ab_dtype_t = cutlass::float_e2m1_t;
    using cd_dtype_t = cutlass::bfloat16_t;

    // Fixed configuration for this de-risk kernel
    constexpr cute::UMMA::Major kMajorA = cute::UMMA::Major::K;
    constexpr cute::UMMA::Major kMajorB = cute::UMMA::Major::K;
    constexpr uint32_t kGemmType = static_cast<uint32_t>(GemmType::Normal);
    constexpr uint32_t kNumGroups = 1;
    constexpr uint32_t kNumMulticast = 2;

    // MMA configs (swap-AB: weights take UMMA "A" / UMMA_M, acts take UMMA "B" / UMMA_N)
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_M;
    constexpr uint32_t UMMA_K = 64;                          // packed FP4 contracts K=64 per instruction
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / kNumMulticast; // acts split on M across the cluster (cluster_n)
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;                 // weights full per CTA
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Swap-AB requires BLOCK_N == 128");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Invalid K tiling");

    // Packed-FP4 byte math: 2 elements per byte
    constexpr uint32_t BLOCK_K_BYTES = BLOCK_K / 2;
    constexpr uint32_t UMMA_K_BYTES = UMMA_K / 2;
    // K-major packed swizzle == K extent in bytes (matches `sm100_fp4_mqa_logits`)
    constexpr uint32_t kSwizzleABMode = BLOCK_K_BYTES;       // 64 for BLOCK_K = 128
    constexpr uint32_t kSwizzleCDMode = 128;

    // SF configs: MXFP4 -> UE8M0 gran-32, NVFP4 -> E4M3/UE4M3 gran-16. UTCCP 128-aligned.
    constexpr uint32_t kGranK = kIsNVFP4 ? 16 : 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    // One int32 packs 4 SFs along K: gran-32 -> covers 128 K (1/load), gran-16 -> covers 64 K (2/load)
    constexpr uint32_t kNumSFKPerLoad = BLOCK_K / (kGranK * 4);
    DG_STATIC_ASSERT(kNumSFKPerLoad == 1 or kNumSFKPerLoad == 2, "Invalid packed SF int count per load");

    // Epilogue configs (swap-AB)
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_M = 16;                  // swap-AB stores `umma_step_n = 16` rows at a time
    constexpr uint32_t STORE_BLOCK_N = BLOCK_N;
    constexpr uint32_t kNumUMMAStoreThreads = kNumEpilogueThreads;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Shared memory sizes (data in bytes via uint8, since packed FP4 is sub-byte)
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K_BYTES;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K_BYTES;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * kNumSFKPerLoad * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * kNumSFKPerLoad * sizeof(uint32_t);

    // Tensor memory size and offsets. Each K-uint32 occupies (SF_BLOCK/32) cols (the MN rows);
    // gran-16 has 2 K-uint32s per K-block. The 2-bit `sf_id` selects within a K-uint32; crossing
    // K-uint32s is done via the SF tmem ADDRESS.
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kSFAColsPerKUint = SF_BLOCK_M / 32;
    constexpr uint32_t kSFBColsPerKUint = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFATmemCols = kSFAColsPerKUint * kNumSFKPerLoad;
    constexpr uint32_t kNumSFBTmemCols = kSFBColsPerKUint * kNumSFKPerLoad;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shapes if the compiler provides them
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sf_k = math::ceil_div(shape_k, kGranK * 4);

    // Align to 1024 bytes for swizzle-128B (C/D)
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D / A / B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA / SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Both CTAs' transposers arrive on the leader's `with_sf` (cross-CTA loads-done sync)
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    // Wait for primary kernel completion (PDL)
    cudaGridDependencySynchronize();

    // Block scheduler (reuse the proven scheduler in Normal mode)
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, true, kNumSMs, kGranK * 4>(
        shape_m, shape_n, shape_k, nullptr);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp — TRUE 2-CTA (cta_group::2), issued by BOTH CTAs. Matches CUTLASS for a
        // 2x1 cluster (verified by instrumenting cute's SM100_TMA_2SM_LOAD on a real GEMM):
        //   - each CTA loads its OWN per-CTA box at its OWN coord (acts split on M via the rank
        //     offset; weights replicated by using the same n coord on both CTAs),
        //   - the 2-SM atom's peer bit routes ALL tx to the LEADER's `full` barrier automatically
        //     (cluster smem addressing sets bit 24 on the peer; the atom masks it to CTA0),
        //   - leader sets `expect_tx` for BOTH CTAs' contributions + loads SF (SM90, leader-resident).
        const auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL);
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                uint32_t m_idx = scheduler.template get_global_idx<false, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<true, sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                uint32_t k_idx = k_block_idx * BLOCK_K;
                auto* mbar = reinterpret_cast<uint64_t*>(full_barriers[stage_idx]);

                // acts: per-CTA M offset (each CTA loads its `LOAD_BLOCK_M`-row half) — 2-SM routes tx to leader
                m_idx += cute::block_rank_in_cluster() * LOAD_BLOCK_M;
                cute::SM100_TMA_2SM_LOAD_2D::copy(&tensor_map_a, mbar, cache_hint, smem_a[stage_idx], k_idx, m_idx);
                // weights: same n coord on both CTAs (each loads the full BLOCK_N replica) — 2-SM routes tx to leader
                cute::SM100_TMA_2SM_LOAD_2D::copy(&tensor_map_b, mbar, cache_hint, smem_b[stage_idx], k_idx, n_idx);

                // SF: per-CTA SM90 load into THIS CTA's smem (the 2-CTA UTCCP reads both CTAs'
                // SF), signaling THIS CTA's own `full` barrier (NOT routed to the leader). So both
                // CTAs' transposers can wait their own `full` and the cross-CTA `with_sf` sync holds.
                // K-uint32 stride per K-block: gran-32 -> 1, gran-16 -> 2
                uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                uint32_t sfa_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_sf_k, kNumSFKPerLoad, k_block_idx);
                tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_sf_k, kNumSFKPerLoad, k_block_idx, m_block_idx);
                tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);

                // Expect: the leader collects BOTH CTAs' data (2-SM routed) + its own SF; the
                // non-leader's `full` only sees its own SF (its data tx went to the leader).
                const auto sf_bytes = (BLOCK_M + BLOCK_N) * kNumSFKPerLoad * sizeof(uint32_t);
                if (is_leader_cta)
                    full_barriers[stage_idx]->arrive_and_expect_tx(
                        SMEM_A_SIZE_PER_STAGE * kNumMulticast + SMEM_B_SIZE_PER_STAGE * kNumMulticast + sf_bytes);
                else
                    full_barriers[stage_idx]->arrive_and_expect_tx(sf_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp (leader CTA only)
        // Swap-AB: weights -> UMMA "A" (UMMA_M), acts -> UMMA "B" (UMMA_N)
        using sf_dtype_t = cute::conditional_t<kIsNVFP4, cutlass::float_ue4m3_t, cutlass::float_ue8m0_t>;
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            ab_dtype_t, ab_dtype_t, float, sf_dtype_t,
            UMMA_M, UMMA_N, kMajorB, kMajorA>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // VALIDATE: make_smem_desc layout/stride for packed E2M1 (mirrors mqa-logits)
        constexpr auto kFP4Layout = mma::sm100::to_umma_layout_type<cute::UMMA::Major::K, kSwizzleABMode, false, ab_dtype_t>();

        DG_STATIC_ASSERT((UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic UMMA N based on effective M (swap-AB)
            {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            #pragma unroll 2
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                if (cute::elect_one_sync()) {
                    // UTCCP copy SFA / SFB into TMEM (transposed in warp 2 already).
                    // gran-16 has `kNumSFKPerLoad` K-uint32s per K-block, each at kSF*ColsPerKUint apart.
                    using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                    #pragma unroll
                    for (uint32_t ku = 0; ku < kNumSFKPerLoad; ++ ku) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto sf_desc = mma::sm100::make_sf_desc(smem_sfa[stage_idx] + ku * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + ku * kSFAColsPerKUint + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto sf_desc = mma::sm100::make_sf_desc(smem_sfb[stage_idx] + ku * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + ku * kSFBColsPerKUint + i * 4);
                        }
                    }

                    // Issue UMMA over UMMA_K (=64) sub-tiles. The 2-bit `sf_id` selects WITHIN a
                    // K-uint32; crossing K-uint32s (gran-16) uses the SF tmem ADDRESS.
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const uint32_t global_sf_idx = k * (UMMA_K / kGranK);
                        const uint32_t sf_kuint = global_sf_idx / 4;
                        const uint32_t sf_id = global_sf_idx % 4;
                        const uint32_t tmem_sfa = kTmemStartColOfSFA + sf_kuint * kSFAColsPerKUint;
                        const uint32_t tmem_sfb = kTmemStartColOfSFB + sf_kuint * kSFBColsPerKUint;
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sf_id, sf_id);
                        auto a_desc = mma::sm100::make_smem_desc(
                            kFP4Layout, reinterpret_cast<ab_dtype_t*>(smem_b[stage_idx]) + k * UMMA_K_BYTES,
                            8 * kSwizzleABMode, 0);
                        auto b_desc = mma::sm100::make_smem_desc(
                            kFP4Layout, reinterpret_cast<ab_dtype_t*>(smem_a[stage_idx]) + k * UMMA_K_BYTES,
                            8 * kSwizzleABMode, 0);
                        // Swap-AB: weights (b_desc) first, SFB id first
                        if constexpr (kIsNVFP4)
                            ptx::SM100_MMA_NVF4_2x1SM_SS::fma(
                                a_desc, b_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                tmem_sfb, tmem_sfa);
                        else
                            ptx::SM100_MMA_MXF4_2x1SM_SS::fma(
                                a_desc, b_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                tmem_sfb, tmem_sfa);
                    }
                }
                __syncwarp();
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        const auto iter_idx = scheduler.current_iter - 1;
        if (iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx == 2) {
        // UTCCP transposer (BOTH CTAs): each waits its own `full` (its SF tx), transposes its SF,
        // then arrives on the leader's `with_sf`. The leader's `full` also gates the 2-SM data.
        auto utccp_required_smem_warp_transpose = [&](uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                #pragma unroll
                for (uint32_t ku = 0; ku < kNumSFKPerLoad; ++ ku)
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + ku * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t ku = 0; ku < kNumSFKPerLoad; ++ ku)
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + ku * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups (swap-AB BF16 store)
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<true, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;
            const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
            epilogue::sm100_store_cd_swap_ab<
                BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cd_dtype_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, tmem_base_addr,
             base_m_idx, base_n_idx, scheduler.current_group_idx,
             effective_m, epilogue_warp_idx, lane_idx,
             tmem_empty_barriers[accum_stage_idx], tensor_map_cd,
             ab_global_scale);
        }
    }

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
