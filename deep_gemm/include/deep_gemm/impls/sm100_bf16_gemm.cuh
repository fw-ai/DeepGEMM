#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm {

// A fused caller may isolate this large persistent body from an earlier phase
// by supplying ``__noinline__`` before including the header. Standalone BF16
// GEMM translation units leave it empty so their existing inlining contract is
// unchanged.
#ifndef DG_BF16_GEMM_BODY_ATTRIBUTE
#define DG_BF16_GEMM_BODY_ATTRIBUTE
#define DG_BF16_GEMM_BODY_ATTRIBUTE_DEFINED_HERE
#endif

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t kKAlignment,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl,
          uint32_t kCombineNumRanks, bool kFuseCombine,
          CombineOrderMode kCombineOrderMode,
          uint32_t kNumExtraCombineThreads>
CUTLASS_DEVICE DG_BF16_GEMM_BODY_ATTRIBUTE void
sm100_bf16_gemm_body(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     // Preserve the enclosing kernel's grid-constant tensor
                     // map storage. TMA instructions require a descriptor
                     // address; passing these 128-byte values by value can
                     // materialize an invalid thread-local descriptor copy.
                     const cute::TmaDescriptor& tensor_map_a,
                     const cute::TmaDescriptor& tensor_map_b,
                     const cute::TmaDescriptor& tensor_map_cd,
                     const layout::SymBuffer<kCombineNumRanks>& combine_sym_buffer,
                     const layout::Workspace& combine_workspace,
                     cutlass::bfloat16_t* grad_x_output,
                     cutlass::bfloat16_t* combine_buffer,
                     const int64_t* combine_topk_ids,
                     uint32_t combine_num_tokens,
                     uint32_t combine_num_max_tokens,
                     uint32_t combine_num_topk,
                     uint32_t combine_hidden,
                     bool combine_reduce,
                     uint8_t* smem_buffer,
                     bool wait_for_primary_kernel) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(
        BLOCK_K_ == 16 || BLOCK_K_ == 32 || BLOCK_K_ == 64,
        "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(kKAlignment % UMMA_K == 0, "K alignment must be divisible by UMMA K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory out of bound for UMMA");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers              = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Standalone launches may be programmatically dependent on a primary
    // kernel. An in-kernel caller has already established operand readiness.
    if (wait_for_primary_kernel)
        cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    // NOTES: BF16 has no SF, so `kSFKSpan` is unused here; pass `kKAlignment` explicitly to avoid relying on the default.
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kEnsureZeroPadding, kKAlignment, kKAlignment>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, kNumMulticast, batch_idx);

                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // UMMA and empty barrier arrival alias
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            constexpr bool kMayHaveTailKBlock = is_k_grouped_contiguous(kGemmType) ? (kKAlignment % BLOCK_K != 0) : (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                        constexpr uint32_t kAtomKIdx = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;
                        constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
                        if (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                        }
                    };
                    auto issue_full_k_block = [&]() {
                        utils::for_each_static_until<BLOCK_K / UMMA_K>(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
                    };

                    if constexpr (kMayHaveTailKBlock) {
                        auto issue_tail_k_block = [&](const uint32_t& remaining_k) {
                            const auto num_valid_umma_k = math::ceil_div(remaining_k, UMMA_K);
                            // Prefix expansion uses switch only for small cases to avoid long SASS.
                            utils::for_each_static_prefix(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), num_valid_umma_k, issue_umma);
                        };
                        const auto is_last_k_block = k_block_idx == num_total_k_blocks - 1;
                        if (is_last_k_block) {
                            const auto remaining_k = scheduler.current_shape_k - k_block_idx * BLOCK_K;
                            if (remaining_k < BLOCK_K)
                                issue_tail_k_block(remaining_k);
                            else
                                issue_full_k_block();
                        } else {
                            issue_full_k_block();
                        }
                    } else {
                        issue_full_k_block();
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    __syncwarp();

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (
        kFuseCombine and
        ((warp_idx >= 2 and
          warp_idx < kNumNonEpilogueThreads / 32) or
         (warp_idx >=
              (kNumNonEpilogueThreads +
               kNumEpilogueThreads) /
                  32 and
          warp_idx <
              (kNumNonEpilogueThreads +
               kNumEpilogueThreads +
               kNumExtraCombineThreads) /
                  32))) {
        if constexpr (kFuseCombine) {
            constexpr uint32_t kNumCombineThreads =
                kNumNonEpilogueThreads - 64 +
                kNumExtraCombineThreads;
            DG_STATIC_ASSERT(
                kNumCombineThreads == 64 or
                    kNumCombineThreads == 128,
                "Fused combine requires two or four warps");
            constexpr uint32_t kCombineNamedBarrierIdx = 15;
            constexpr uint32_t kBeforeCombineReuseGridSyncIdx = 2;
            constexpr uint32_t kBeforeCombineReuseTag = 7;
            constexpr uint32_t kExtraCombineWarpBegin =
                (kNumNonEpilogueThreads +
                 kNumEpilogueThreads) /
                32;
            const uint32_t combine_thread_idx =
                warp_idx < kNumNonEpilogueThreads / 32
                    ? (warp_idx - 2) * 32 + lane_idx
                    : kNumNonEpilogueThreads - 64 +
                          (warp_idx -
                           kExtraCombineWarpBegin) *
                              32 +
                          lane_idx;
            const auto combine_sync = [=]() {
                ptx::sync_aligned(
                    kNumCombineThreads,
                    kCombineNamedBarrierIdx);
            };

            constexpr uint32_t kValuesPerVec =
                sizeof(uint4) /
                sizeof(cutlass::bfloat16_t);

            if (!combine_reduce) {
                // Kernel A remotely pulls grad-y from this same symmetric
                // region. W2's combine warps protect its reuse while the
                // independent W2 tensor-core work runs.
                comm::nvlink_barrier<
                    kCombineNumRanks, kNumSMs,
                    kNumCombineThreads,
                    kBeforeCombineReuseGridSyncIdx,
                    kBeforeCombineReuseTag>(
                    combine_workspace, combine_sym_buffer,
                    blockIdx.x, combine_thread_idx,
                    combine_sync,
                    /* Inline callers need a device-wide publication edge;
                       standalone launch order already provides one. */
                    !wait_for_primary_kernel,
                    /* Kernel completion joins SM 0 after cross-rank sync. */ false);
            } else {
                // Kernel A has already written every remote top-k plane.
                // W13's combine warps reduce those planes while the
                // independent W13 tensor-core work runs.
                constexpr uint32_t kReduceValuesPerVec =
                    2 * kValuesPerVec;
                const uint64_t num_output_vecs =
                    static_cast<uint64_t>(
                        combine_num_tokens) *
                    combine_hidden / kReduceValuesPerVec;
                for (uint64_t linear =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumCombineThreads +
                         combine_thread_idx;
                     linear < num_output_vecs;
                     linear +=
                         static_cast<uint64_t>(kNumSMs) *
                         kNumCombineThreads) {
                    const uint32_t num_vecs_per_token =
                        combine_hidden /
                        kReduceValuesPerVec;
                    const uint32_t token_idx =
                        linear / num_vecs_per_token;
                    const uint32_t vec_idx =
                        linear -
                        static_cast<uint64_t>(token_idx) *
                            num_vecs_per_token;
                    float values[kReduceValuesPerVec] = {
                        0.0f};
                    const auto accumulate_slot =
                        [&](float* dst,
                            const uint32_t topk_idx) {
                            const uint64_t packed_idx =
                                ((static_cast<uint64_t>(
                                      topk_idx) *
                                  combine_num_max_tokens +
                                  token_idx) *
                                     num_vecs_per_token +
                                 vec_idx) *
                                2;
                            uint4 packed[2];
                            packed[0] =
                                reinterpret_cast<
                                    const uint4*>(
                                    combine_buffer)[
                                    packed_idx];
                            packed[1] =
                                reinterpret_cast<
                                    const uint4*>(
                                    combine_buffer)[
                                    packed_idx + 1];
                            const auto* packed_values =
                                reinterpret_cast<
                                    const cutlass::bfloat16_t*>(
                                    packed);
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < kReduceValuesPerVec;
                                 ++i) {
                                dst[i] +=
                                    static_cast<float>(
                                        packed_values[i]);
                            }
                        };
                    if constexpr (
                        kCombineOrderMode ==
                        CombineOrderMode::FixedTopK) {
                        for (uint32_t topk_idx = 0;
                             topk_idx <
                                 combine_num_topk;
                             ++topk_idx) {
                            accumulate_slot(
                                values, topk_idx);
                        }
                    } else {
                        DG_DEVICE_ASSERT(
                            combine_topk_ids != nullptr);
                        DG_DEVICE_ASSERT(
                            combine_num_topk <= 32);
                        const uint32_t
                            num_rank_iterations =
                                kCombineOrderMode ==
                                        CombineOrderMode::
                                            DeepEPV1
                                ? kCombineNumRanks
                                : combine_num_topk;
                        for (uint32_t rank_iteration = 0;
                             rank_iteration <
                                 num_rank_iterations;
                             ++rank_iteration) {
                            int dst_rank;
                            if constexpr (
                                kCombineOrderMode ==
                                CombineOrderMode::
                                    DeepEPV1) {
                                dst_rank =
                                    static_cast<int>(
                                        rank_iteration);
                                bool rank_is_present =
                                    false;
                                for (uint32_t i = 0;
                                     i <
                                         combine_num_topk;
                                     ++i) {
                                    const int64_t
                                        expert_idx =
                                            combine_topk_ids[
                                                static_cast<
                                                    uint64_t>(
                                                    token_idx) *
                                                    combine_num_topk +
                                                i];
                                    rank_is_present |=
                                        expert_idx >= 0 &&
                                        expert_idx /
                                                kNumGroups ==
                                            dst_rank;
                                }
                                if (!rank_is_present)
                                    continue;
                            } else {
                                const int64_t
                                    expert_idx =
                                        combine_topk_ids[
                                            static_cast<
                                                uint64_t>(
                                                token_idx) *
                                                combine_num_topk +
                                            rank_iteration];
                                if (expert_idx < 0)
                                    continue;
                                dst_rank =
                                    static_cast<int>(
                                        expert_idx /
                                        kNumGroups);
                                bool appears_later =
                                    false;
                                for (uint32_t i =
                                         rank_iteration +
                                         1;
                                     i <
                                         combine_num_topk;
                                     ++i) {
                                    const int64_t
                                        later_expert_idx =
                                            combine_topk_ids[
                                                static_cast<
                                                    uint64_t>(
                                                    token_idx) *
                                                    combine_num_topk +
                                                i];
                                    appears_later |=
                                        later_expert_idx >=
                                            0 &&
                                        later_expert_idx /
                                                kNumGroups ==
                                            dst_rank;
                                }
                                if (appears_later)
                                    continue;
                            }

                            float rank_values[
                                kReduceValuesPerVec] = {
                                0.0f};
                            uint32_t rank_mask = 0;
                            for (uint32_t topk_idx = 0;
                                 topk_idx <
                                     combine_num_topk;
                                 ++topk_idx) {
                                const int64_t
                                    expert_idx =
                                        combine_topk_ids[
                                            static_cast<
                                                uint64_t>(
                                                token_idx) *
                                                combine_num_topk +
                                            topk_idx];
                                if (expert_idx >= 0 &&
                                    expert_idx /
                                            kNumGroups ==
                                        dst_rank) {
                                    rank_mask |=
                                        1u << topk_idx;
                                }
                            }
                            // TorchTitan's deterministic DeepEP V1 and V2
                            // paths both reduce each rank partial in source
                            // top-k slot order. Their cross-rank orders remain
                            // distinct.
                            while (rank_mask) {
                                uint32_t selected_slot =
                                    __ffs(rank_mask) - 1;
                                rank_mask &=
                                    ~(1u << selected_slot);
                                accumulate_slot(
                                    rank_values,
                                    selected_slot);
                            }
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i <
                                     kReduceValuesPerVec;
                                 ++i) {
                                const cutlass::
                                    bfloat16_t
                                        rank_partial(
                                            rank_values[
                                                i]);
                                values[i] +=
                                    static_cast<float>(
                                        rank_partial);
                            }
                        }
                    }
                    uint4 packed_output[2];
                    auto* output_values =
                        reinterpret_cast<
                            cutlass::bfloat16_t*>(
                            packed_output);
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kReduceValuesPerVec; ++i)
                        output_values[i] =
                            cutlass::bfloat16_t(values[i]);
                    auto* output =
                        reinterpret_cast<uint4*>(
                            grad_x_output) +
                        linear * 2;
                    output[0] = packed_output[0];
                    output[1] = packed_output[1];
                }
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<
                (not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode, uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs, uint32_t kKAlignment,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl,
          uint32_t kCombineNumRanks, bool kFuseCombine,
          CombineOrderMode kCombineOrderMode,
          uint32_t kNumExtraCombineThreads>
CUTLASS_GLOBAL void __launch_bounds__(
    kNumNonEpilogueThreads + kNumEpilogueThreads +
        kNumExtraCombineThreads,
    1)
sm100_bf16_gemm_impl(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                     const __grid_constant__ layout::SymBuffer<kCombineNumRanks> combine_sym_buffer,
                     const __grid_constant__ layout::Workspace combine_workspace,
                     cutlass::bfloat16_t* grad_x_output,
                     cutlass::bfloat16_t* combine_buffer,
                     const int64_t* combine_topk_ids,
                     uint32_t combine_num_tokens,
                     uint32_t combine_num_max_tokens,
                     uint32_t combine_num_topk,
                     uint32_t combine_hidden,
                     bool combine_reduce) {
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    sm100_bf16_gemm_body<
        kMajorA, kMajorB,
        SHAPE_M, SHAPE_N, SHAPE_K,
        BLOCK_M, BLOCK_N, BLOCK_K_,
        kNumGroups,
        kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
        kNumStages_,
        kNumNonEpilogueThreads, kNumEpilogueThreads,
        kNumMulticast, kIsMulticastOnA,
        kNumSMs, kKAlignment,
        kSwapAB, kEnsureZeroPadding,
        kGemmType, kWithAccumulation, cd_dtype_t,
        kTensorCoreUtilControl,
        kCombineNumRanks, kFuseCombine,
        kCombineOrderMode, kNumExtraCombineThreads>(
            grouped_layout,
            shape_m, shape_n, shape_k,
            tensor_map_a, tensor_map_b, tensor_map_cd,
            combine_sym_buffer, combine_workspace,
            grad_x_output, combine_buffer, combine_topk_ids,
            combine_num_tokens, combine_num_max_tokens,
            combine_num_topk, combine_hidden, combine_reduce,
            smem_buffer, true);
}

};  // namespace deep_gemm

#ifdef DG_BF16_GEMM_BODY_ATTRIBUTE_DEFINED_HERE
#undef DG_BF16_GEMM_BODY_ATTRIBUTE_DEFINED_HERE
#undef DG_BF16_GEMM_BODY_ATTRIBUTE
#endif

#pragma clang diagnostic pop
