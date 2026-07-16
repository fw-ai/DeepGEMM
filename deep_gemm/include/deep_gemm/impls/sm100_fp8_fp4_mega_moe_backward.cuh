#pragma once

#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Production MegaMoE backward wave. This persistent kernel consumes the
// forward kernel's block-padded expert pool directly and replays gate and up
// together as one W13 FP8xFP4 mainloop before computing the retained dgrads.
template <
    uint32_t kHidden, uint32_t kIntermediateHidden,
    uint32_t kNumExperts,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
    uint32_t kNumStages,
    uint32_t kNumSMs,
    uint32_t kNumRanks = 1,
    bool kCompileW13Dgrad = true,
    bool kBF16Mode = false,
    ActivationType kActivationType = ActivationType::SwiGLU,
    bool kFastMath = false,
    RouteWeightMode kRouteWeightMode = RouteWeightMode::PreDown,
    uint32_t kNumNonEpilogueThreads = 128,
    uint32_t kNumEpilogueThreads = 128,
    uint32_t kNumThreads =
        kNumNonEpilogueThreads + kNumEpilogueThreads +
        768>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_backward_wave_impl(
    const int* expert_counts,
    const __grid_constant__ layout::SymBuffer<kNumRanks> backward_sym_buffer,
    const __grid_constant__ layout::Workspace backward_workspace,
    const cutlass::bfloat16_t* backward_grad_y,
    const cutlass::bfloat16_t* backward_x,
    const float* backward_topk_weights,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_topk,
    const uint32_t acts_sf_stride,
    const __grid_constant__ cute::TmaDescriptor tensor_map_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_ye,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_dequant,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_scales,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_dequant,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_scales,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_gate_up,
    const cutlass::float_e4m3_t* acts_ptr,
    const uint32_t* acts_sf_ptr,
    const int8_t* w2_weights,
    const float* w2_scales,
    cutlass::bfloat16_t* w2_dequant_scratch,
    const int8_t* w13_weights,
    const float* w13_scales,
    cutlass::bfloat16_t* w13_dequant_scratch,
    const cutlass::bfloat16_t* gate_up_output,
    cutlass::bfloat16_t* grad_ye_output,
    cutlass::bfloat16_t* grad_y_unweighted_output,
    cutlass::bfloat16_t* route_weights,
    float* route_weights_fp32,
    cutlass::bfloat16_t* grad_h_output,
    cutlass::bfloat16_t* grad_gate_up_output,
    cutlass::bfloat16_t* h_act_output,
    cutlass::bfloat16_t* h_weighted_output,
    cutlass::bfloat16_t* x_pool_output,
    cutlass::bfloat16_t* grad_x_pool_output,
    const cutlass::bfloat16_t* down_unweighted_output,
    float* grad_route_output,
    uint32_t* weight_tile_states,
    const uint32_t launch_epoch,
    const float activation_limit,
    const bool compute_w13_dgrad,
    const bool direct_remote_grad_x,
    const bool write_grad_x_pool,
    const bool clear_wgrad_padding) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    using cd_dtype_t = cutlass::bfloat16_t;
    using dgrad_b_dtype_t = cutlass::bfloat16_t;

    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumDispatchThreads =
        kNumRanks > 1 ? 128 : 0;
    constexpr uint32_t kNumDispatchWarps =
        kNumDispatchThreads / 32;
    constexpr uint32_t kDispatchWarpStart =
        (kNumNonEpilogueThreads + kNumEpilogueThreads) / 32;
    constexpr uint32_t kNumDgradEpilogueThreads =
        kNumThreads - kNumNonEpilogueThreads;
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kNumBlockNs = (2 * kIntermediateHidden) / BLOCK_N;
    constexpr uint32_t kNumDgradBlockNs = kIntermediateHidden / BLOCK_N;
    constexpr uint32_t kNumW13DgradBlockNs = kHidden / BLOCK_N;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t DGRAD_BLOCK_K = 64;
    constexpr uint32_t DGRAD_UMMA_K = 16;
    constexpr uint32_t kNumW2WeightTileStates =
        kNumExperts * (kHidden / DGRAD_BLOCK_K) *
        kNumDgradBlockNs;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = 16;
    constexpr uint32_t STORE_BLOCK_N = BLOCK_N;
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;

    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid producer thread count");
    DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Invalid epilogue thread count");
    DG_STATIC_ASSERT(kNumRanks == 1 || kNumDispatchThreads == 128,
                     "Invalid backward dispatch thread count");
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0 && BLOCK_N == 128 && BLOCK_K == 128,
                     "Invalid backward wave tile");
    DG_STATIC_ASSERT(kNumBlockNs % 2 == 0, "Cluster peers must receive adjacent N blocks");
    DG_STATIC_ASSERT(kNumDgradBlockNs % 2 == 0,
                     "Dgrad cluster peers must receive adjacent N blocks");
    DG_STATIC_ASSERT(kNumW13DgradBlockNs % 2 == 0,
                     "W13 dgrad cluster peers must receive adjacent N blocks");
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems),
                     "Invalid SFA block");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "Invalid SFB block");
    DG_STATIC_ASSERT(kHidden % BLOCK_K == 0, "Invalid hidden size");
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "2-CTA clusters require an even SM count");

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_acts);
        cute::prefetch_tma_descriptor(&tensor_map_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_output);
        cute::prefetch_tma_descriptor(&tensor_map_grad_ye);
        cute::prefetch_tma_descriptor(&tensor_map_w2_dequant);
        cute::prefetch_tma_descriptor(&tensor_map_w2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_w2_scales);
        cute::prefetch_tma_descriptor(&tensor_map_w13_dequant);
        cute::prefetch_tma_descriptor(&tensor_map_w13_weights);
        cute::prefetch_tma_descriptor(&tensor_map_w13_scales);
        cute::prefetch_tma_descriptor(&tensor_map_grad_gate_up);
    }

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE =
        STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE =
        LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
        LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_DISPATCH_SIZE =
        kNumDispatchWarps * kHidden * sizeof(cd_dtype_t);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto* smem_gemm_base = smem_buffer + SMEM_DISPATCH_SIZE;
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(
            smem_gemm_base + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(
            smem_gemm_base + SMEM_CD_SIZE +
            i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(
            smem_gemm_base + SMEM_CD_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE +
            i * SMEM_B_SIZE_PER_STAGE);
    });
    // The dgrad phase aliases the recompute mainloop storage exactly:
    //   FP8 A [BLOCK_M/2, 128] == BF16 A [BLOCK_M/2, 64]
    //   packed-FP4 B [128, 128] == BF16 B [128, 64].
    // W2 is dequantized and transposed directly into the latter by a producer
    // warp, so no persistent weight copy or host-side packing is needed.
    auto smem_dgrad_a = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(
            smem_gemm_base + SMEM_CD_SIZE +
            i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_dgrad_b = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<dgrad_b_dtype_t*>(
            smem_gemm_base + SMEM_CD_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE +
            i * SMEM_B_SIZE_PER_STAGE);
    });
    DG_STATIC_ASSERT(
        LOAD_BLOCK_M * DGRAD_BLOCK_K * sizeof(cd_dtype_t) ==
            SMEM_A_SIZE_PER_STAGE,
        "Dgrad A alias size mismatch");
    DG_STATIC_ASSERT(
        LOAD_BLOCK_N * DGRAD_BLOCK_K * sizeof(dgrad_b_dtype_t) ==
            SMEM_B_SIZE_PER_STAGE,
        "Dgrad B alias size mismatch");
    auto sf_start_ptr = smem_gemm_base + SMEM_CD_SIZE +
                        kNumStages *
                            (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(
            sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(
            sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
            i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);
    auto full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages + i;
    });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages + kNumEpilogueStages + i;
    });
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages +
            2 * kNumEpilogueStages + i;
    });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + 2 * kNumStages +
        2 * kNumEpilogueStages + kNumDispatchWarps);

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<
            kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB =
        kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Backward recompute exceeds TMEM");

    if constexpr (!kBF16Mode) {
        // Dequantize W2 exactly once per launch into an ephemeral
        // [expert, dim, H] BF16 workspace.  Keeping the source orientation
        // makes both packed-FP4 reads and BF16 writes coalesced; dgrad consumes
        // it as an MN-major transposed operand.
        constexpr uint32_t kDequantTileK = 256;
        constexpr uint32_t kDequantTileN = LOAD_BLOCK_N;
        constexpr uint32_t kDequantPairsPerTile =
            kDequantTileK * kDequantTileN / 2;
        constexpr uint32_t kDequantSFsPerK =
            kDequantTileN / 32;
        constexpr uint32_t kDequantSFsPerTile =
            kDequantTileK * kDequantSFsPerK;
        constexpr uint32_t kDequantWeightBytes =
            kDequantTileK * (kDequantTileN / 2);
        constexpr uint32_t kDequantScaleBytes =
            kDequantSFsPerTile * sizeof(float);
        constexpr uint32_t kNumDequantKTiles =
            kHidden / kDequantTileK;
        constexpr uint32_t kNumDequantNTiles =
            kIntermediateHidden / kDequantTileN;
        constexpr uint32_t kNumDequantTiles =
            kNumExperts * kNumDequantKTiles *
            kNumDequantNTiles;
        auto* dequant_weights =
            reinterpret_cast<int8_t*>(smem_buffer);
        auto* dequant_scales =
            reinterpret_cast<float*>(
                smem_buffer + kDequantWeightBytes);
        auto* dequant_scale_half2 =
            reinterpret_cast<uint32_t*>(
                smem_buffer + kDequantWeightBytes +
                kDequantScaleBytes);
        comm::cluster_sync_with_relaxed_arrive();
        if (warp_idx == 0 && cute::elect_one_sync()) {
            full_barriers[0]->init(1);
            if constexpr (kCompileW13Dgrad)
                full_barriers[1]->init(1);
            cutlass::arch::fence_barrier_init();
        }
        comm::cluster_sync_with_relaxed_arrive();
        uint32_t dequant_phase = 0;

        for (uint32_t tile_idx = blockIdx.x;
             tile_idx < kNumDequantTiles;
             tile_idx += kNumSMs) {
            const uint32_t n_tile_idx =
                tile_idx % kNumDequantNTiles;
            const uint32_t k_expert_tile_idx =
                tile_idx / kNumDequantNTiles;
            const uint32_t k_tile_idx =
                k_expert_tile_idx % kNumDequantKTiles;
            const uint32_t expert_idx =
                k_expert_tile_idx / kNumDequantKTiles;
            const uint32_t global_k_base =
                k_tile_idx * kDequantTileK;
            const uint32_t global_n_base =
                n_tile_idx * kDequantTileN;

            if (warp_idx == 0 && cute::elect_one_sync()) {
                tma::copy<
                    kDequantTileN / 2, kDequantTileK, 0,
                    int8_t>(
                    &tensor_map_w2_weights,
                    full_barriers[0], dequant_weights,
                    global_n_base / 2,
                    expert_idx * kHidden + global_k_base);
                tma::copy<
                    kDequantSFsPerK, kDequantTileK, 0,
                    float>(
                    &tensor_map_w2_scales,
                    full_barriers[0], dequant_scales,
                    global_n_base / 32,
                    expert_idx * kHidden + global_k_base);
                full_barriers[0]->arrive_and_expect_tx(
                    kDequantWeightBytes +
                    kDequantScaleBytes);
            }
            full_barriers[0]->wait(dequant_phase);
            __syncthreads();

            for (uint32_t scale_idx = threadIdx.x;
                 scale_idx < kDequantSFsPerTile;
                 scale_idx += kNumThreads) {
                const auto scale_half2 =
                    __float2half2_rn(
                        dequant_scales[scale_idx]);
                dequant_scale_half2[scale_idx] =
                    *reinterpret_cast<const uint32_t*>(
                        &scale_half2);
            }
            __syncthreads();

            for (uint32_t pair_idx = threadIdx.x;
                 pair_idx < kDequantPairsPerTile;
                 pair_idx += kNumThreads) {
                const uint32_t local_k =
                    pair_idx / (kDequantTileN / 2);
                const uint32_t local_n_pair =
                    pair_idx % (kDequantTileN / 2);
                const uint32_t global_k =
                    global_k_base + local_k;
                const uint32_t global_n_pair =
                    global_n_base / 2 + local_n_pair;
                const uint8_t packed =
                    static_cast<uint8_t>(
                        dequant_weights[
                            local_k *
                                (kDequantTileN / 2) +
                            local_n_pair]);
                uint32_t fp16x2;
                asm volatile(
                    "{\n"
                    ".reg .b8 fp4;\n"
                    ".reg .b8 unused1, unused2, unused3;\n"
                    "mov.b32 {fp4, unused1, unused2, unused3}, %1;\n"
                    "cvt.rn.f16x2.e2m1x2 %0, fp4;\n"
                    "}\n"
                    : "=r"(fp16x2)
                    : "r"(static_cast<uint32_t>(packed)));
                auto value_pair =
                    *reinterpret_cast<__half2*>(&fp16x2);
                const uint32_t scale_half2_bits =
                    dequant_scale_half2[
                        local_k * kDequantSFsPerK +
                        (local_n_pair * 2) / 32];
                const auto scale_half2 =
                    *reinterpret_cast<const __half2*>(
                        &scale_half2_bits);
                value_pair =
                    __hmul2(value_pair, scale_half2);
                const auto value_pair_bf16 =
                    __float22bfloat162_rn(
                        __half22float2(value_pair));
                const uint32_t scaled_pair =
                    *reinterpret_cast<const uint32_t*>(
                        &value_pair_bf16);
                *reinterpret_cast<uint32_t*>(
                    w2_dequant_scratch +
                    (static_cast<uint64_t>(expert_idx) *
                         kHidden +
                     global_k) *
                        kIntermediateHidden +
                    global_n_pair * 2) =
                    scaled_pair;
            }
            __syncthreads();
            if (threadIdx.x <
                kDequantTileK / DGRAD_BLOCK_K) {
                const uint32_t dgrad_k_block_idx =
                    k_tile_idx *
                        (kDequantTileK /
                         DGRAD_BLOCK_K) +
                    threadIdx.x;
                const uint32_t weight_tile_idx =
                    (expert_idx *
                         (kHidden / DGRAD_BLOCK_K) +
                     dgrad_k_block_idx) *
                        kNumDgradBlockNs +
                    n_tile_idx;
                asm volatile(
                    "st.release.gpu.global.u32 [%0], %1;"
                    :: "l"(weight_tile_states +
                           weight_tile_idx),
                       "r"(launch_epoch)
                    : "memory");
            }
            __syncthreads();
            dequant_phase ^= 1;
        }

        if constexpr (kCompileW13Dgrad) {
            constexpr uint32_t kW13DequantTileK = 256;
            constexpr uint32_t kW13DequantTileN = LOAD_BLOCK_N;
            constexpr uint32_t kW13DequantPairsPerTile =
                kW13DequantTileK * kW13DequantTileN / 2;
            constexpr uint32_t kW13DequantSFsPerK =
                kW13DequantTileN / 32;
            constexpr uint32_t kW13DequantSFsPerTile =
                kW13DequantTileK * kW13DequantSFsPerK;
            constexpr uint32_t kW13DequantWeightBytes =
                kW13DequantTileK * (kW13DequantTileN / 2);
            constexpr uint32_t kW13DequantScaleBytes =
                kW13DequantSFsPerTile * sizeof(float);
            constexpr uint32_t kNumW13DequantKTiles =
                (2 * kIntermediateHidden) / kW13DequantTileK;
            constexpr uint32_t kNumW13DequantNTiles =
                kHidden / kW13DequantTileN;
            constexpr uint32_t kNumW13DequantTiles =
                kNumExperts * kNumW13DequantKTiles *
                kNumW13DequantNTiles;
            const uint32_t w13_launch_epoch =
                launch_epoch ^ 0x80000000u;
            uint32_t w13_dequant_phase = 0;

            for (uint32_t tile_idx = blockIdx.x;
                 tile_idx < kNumW13DequantTiles;
                 tile_idx += kNumSMs) {
                const uint32_t n_tile_idx =
                    tile_idx % kNumW13DequantNTiles;
                const uint32_t k_expert_tile_idx =
                    tile_idx / kNumW13DequantNTiles;
                const uint32_t k_tile_idx =
                    k_expert_tile_idx % kNumW13DequantKTiles;
                const uint32_t expert_idx =
                    k_expert_tile_idx / kNumW13DequantKTiles;
                const uint32_t global_k_base =
                    k_tile_idx * kW13DequantTileK;
                const uint32_t global_n_base =
                    n_tile_idx * kW13DequantTileN;

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    tma::copy<
                        kW13DequantTileN / 2,
                        kW13DequantTileK, 0, int8_t>(
                        &tensor_map_w13_weights,
                        full_barriers[1],
                        dequant_weights,
                        global_n_base / 2,
                        expert_idx *
                                (2 * kIntermediateHidden) +
                            global_k_base);
                    tma::copy<
                        kW13DequantSFsPerK,
                        kW13DequantTileK, 0, float>(
                        &tensor_map_w13_scales,
                        full_barriers[1],
                        dequant_scales,
                        global_n_base / 32,
                        expert_idx *
                                (2 * kIntermediateHidden) +
                            global_k_base);
                    full_barriers[1]->arrive_and_expect_tx(
                        kW13DequantWeightBytes +
                        kW13DequantScaleBytes);
                }
                full_barriers[1]->wait(
                    w13_dequant_phase);
                __syncthreads();

                for (uint32_t scale_idx = threadIdx.x;
                     scale_idx < kW13DequantSFsPerTile;
                     scale_idx += kNumThreads) {
                    const auto scale_half2 =
                        __float2half2_rn(
                            dequant_scales[scale_idx]);
                    dequant_scale_half2[scale_idx] =
                        *reinterpret_cast<
                            const uint32_t*>(
                            &scale_half2);
                }
                __syncthreads();

                for (uint32_t pair_idx = threadIdx.x;
                     pair_idx <
                         kW13DequantPairsPerTile;
                     pair_idx += kNumThreads) {
                    const uint32_t local_k =
                        pair_idx /
                        (kW13DequantTileN / 2);
                    const uint32_t local_n_pair =
                        pair_idx %
                        (kW13DequantTileN / 2);
                    const uint32_t global_k =
                        global_k_base + local_k;
                    const uint32_t global_n_pair =
                        global_n_base / 2 +
                        local_n_pair;
                    const uint8_t packed =
                        static_cast<uint8_t>(
                            dequant_weights[
                                local_k *
                                    (kW13DequantTileN / 2) +
                                local_n_pair]);
                    uint32_t fp16x2;
                    asm volatile(
                        "{\n"
                        ".reg .b8 fp4;\n"
                        ".reg .b8 unused1, unused2, unused3;\n"
                        "mov.b32 {fp4, unused1, unused2, unused3}, %1;\n"
                        "cvt.rn.f16x2.e2m1x2 %0, fp4;\n"
                        "}\n"
                        : "=r"(fp16x2)
                        : "r"(
                              static_cast<uint32_t>(
                                  packed)));
                    auto value_pair =
                        *reinterpret_cast<__half2*>(
                            &fp16x2);
                    const uint32_t scale_half2_bits =
                        dequant_scale_half2[
                            local_k *
                                kW13DequantSFsPerK +
                            (local_n_pair * 2) / 32];
                    value_pair = __hmul2(
                        value_pair,
                        *reinterpret_cast<
                            const __half2*>(
                            &scale_half2_bits));
                    const auto value_pair_bf16 =
                        __float22bfloat162_rn(
                            __half22float2(
                                value_pair));
                    *reinterpret_cast<uint32_t*>(
                        w13_dequant_scratch +
                        (static_cast<uint64_t>(
                             expert_idx) *
                             (2 *
                              kIntermediateHidden) +
                         global_k) *
                            kHidden +
                        global_n_pair * 2) =
                        *reinterpret_cast<
                            const uint32_t*>(
                            &value_pair_bf16);
                }
                __syncthreads();

                if (threadIdx.x <
                    kW13DequantTileK /
                        DGRAD_BLOCK_K) {
                    const uint32_t
                        dgrad_k_block_idx =
                            k_tile_idx *
                                (kW13DequantTileK /
                                 DGRAD_BLOCK_K) +
                            threadIdx.x;
                    const uint32_t weight_tile_idx =
                        (expert_idx *
                             ((2 *
                               kIntermediateHidden) /
                              DGRAD_BLOCK_K) +
                         dgrad_k_block_idx) *
                            kNumW13DgradBlockNs +
                        n_tile_idx;
                    asm volatile(
                        "st.release.gpu.global.u32 [%0], %1;"
                        :: "l"(weight_tile_states +
                               kNumW2WeightTileStates +
                               weight_tile_idx),
                           "r"(w13_launch_epoch)
                        : "memory");
                }
                __syncthreads();
                w13_dequant_phase ^= 1;
            }
        }
    }
    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 0 && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(4);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumDispatchWarps; ++i)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 1) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    comm::cluster_sync_with_relaxed_arrive();

    // Every role walks this deterministic schedule independently.  Pool offsets
    // are prefixes of ceil(count/BLOCK_M), matching the forward MegaMoE layout.
    const auto for_each_block = [&](const auto& func) {
        uint32_t next_assigned_block = blockIdx.x;
        uint32_t global_block = 0;
        uint32_t pool_block_offset = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0; expert_idx < kNumExperts; ++expert_idx) {
            const uint32_t num_tokens =
                static_cast<uint32_t>(__ldg(expert_counts + expert_idx));
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t expert_blocks = num_m_blocks * kNumBlockNs;
            const uint32_t expert_end = global_block + expert_blocks;

            while (next_assigned_block < global_block)
                next_assigned_block += kNumSMs;
            while (next_assigned_block < expert_end) {
                const uint32_t local_block =
                    next_assigned_block - global_block;
                const uint32_t m_block_idx = local_block / kNumBlockNs;
                const uint32_t n_block_idx =
                    local_block - m_block_idx * kNumBlockNs;
                const uint32_t valid_m = cute::min(
                    num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
                func(expert_idx, pool_block_offset, m_block_idx,
                     n_block_idx, valid_m);
                next_assigned_block += kNumSMs;
            }
            global_block = expert_end;
            pool_block_offset += num_m_blocks;
        }
    };

    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    const auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    constexpr uint32_t kNumProducerRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;

    if constexpr (!kBF16Mode) {
      if (warp_idx == 0) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();
        for_each_block([&](const uint32_t&, const uint32_t& pool_block_offset,
                           const uint32_t& m_block_idx, const uint32_t&,
                           const uint32_t& valid_m) {
            const uint32_t pool_block_idx =
                pool_block_offset + m_block_idx;
            #pragma unroll
            for (uint32_t k_block_idx = 0;
                 k_block_idx < kHidden / BLOCK_K;
                 advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                if (!is_leader_cta)
                    m_idx += math::align(valid_m, 16u) / 2;
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode,
                              a_dtype_t>(
                        &tensor_map_acts, full_barriers[stage_idx],
                        smem_a[stage_idx], k_block_idx * BLOCK_K, m_idx, 2);
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        &tensor_map_acts_sf, full_barriers[stage_idx],
                        smem_sfa[stage_idx],
                        pool_block_idx * SF_BLOCK_M, k_block_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_A_SIZE_PER_STAGE * 2 +
                            SF_BLOCK_M * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();
        for_each_block([&](const uint32_t& expert_idx, const uint32_t&,
                           const uint32_t&, const uint32_t& n_block_idx,
                           const uint32_t&) {
            #pragma unroll
            for (uint32_t k_block_idx = 0;
                 k_block_idx < kHidden / BLOCK_K;
                 advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode,
                              b_dtype_t>(
                        &tensor_map_weights, full_barriers[stage_idx],
                        smem_b[stage_idx], k_block_idx * BLOCK_K,
                        expert_idx * 2 * kIntermediateHidden +
                            n_block_idx * BLOCK_N,
                        2);
                    tma::copy<BLOCK_N, 1, 0>(
                        &tensor_map_weights_sf, full_barriers[stage_idx],
                        smem_sfb[stage_idx], n_block_idx * BLOCK_N,
                        expert_idx * (kHidden / (kGranK * 4)) +
                            k_block_idx,
                        2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            SMEM_B_SIZE_PER_STAGE +
                            BLOCK_N * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();
        if (is_leader_cta) {
            auto instr_desc =
                cute::UMMA::make_instr_desc_block_scaled<
                    b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                    UMMA_M, UMMA_N, cute::UMMA::Major::K,
                    cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);
            auto a_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K,
                kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K,
                kSwizzleBMode>(smem_b[0], 0, 0);
            const uint32_t a_desc_lo = lane_idx < kNumStages
                ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16
                : 0;
            const uint32_t b_desc_lo = lane_idx < kNumStages
                ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16
                : 0;
            uint32_t current_iter = 0;

            for_each_block([&](const uint32_t&, const uint32_t&,
                               const uint32_t&, const uint32_t&,
                               const uint32_t& valid_m) {
                mma::sm100::update_instr_desc_with_umma_n(
                    instr_desc, math::align(valid_m, 16u));
                const uint32_t accum_stage =
                    current_iter % kNumEpilogueStages;
                const uint32_t accum_phase =
                    (current_iter++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage]->wait(
                    accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                #pragma unroll
                for (uint32_t k_block_idx = 0;
                     k_block_idx < kHidden / BLOCK_K;
                     advance_pipeline(k_block_idx)) {
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    const uint32_t a_desc_base =
                        ptx::exchange(a_desc_lo, stage_idx);
                    const uint32_t b_desc_base =
                        ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        using utccp_t =
                            cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t i = 0;
                             i < SF_BLOCK_M /
                                     kNumUTCCPAlignedElems;
                             ++i) {
                            mma::sm100::replace_smem_desc_addr(
                                sf_desc,
                                smem_sfa[stage_idx] +
                                    i * kNumUTCCPAlignedElems);
                            utccp_t::copy(
                                sf_desc,
                                kTmemStartColOfSFA + i * 4);
                        }
                        mma::sm100::replace_smem_desc_addr(
                            sf_desc, smem_sfb[stage_idx]);
                        utccp_t::copy(sf_desc, kTmemStartColOfSFB);

                        #pragma unroll
                        for (uint32_t k = 0;
                             k < BLOCK_K / UMMA_K; ++k) {
                            const auto runtime_instr_desc =
                                mma::sm100::
                                    make_runtime_instr_desc_with_sf_id(
                                        instr_desc, k, k);
                            a_desc.lo =
                                mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K,
                                    LOAD_BLOCK_M, kSwizzleAMode,
                                    a_dtype_t>(
                                    a_desc_base, 0, k * UMMA_K);
                            b_desc.lo =
                                mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::K,
                                    LOAD_BLOCK_N, kSwizzleBMode,
                                    b_dtype_t>(
                                    b_desc_base, 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc,
                                accum_stage * UMMA_N,
                                k_block_idx > 0 || k > 0,
                                runtime_instr_desc,
                                kTmemStartColOfSFB,
                                kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    constexpr uint16_t kCTAMask = 0x3;
                    cutlass::arch::umma_arrive_multicast_2x1SM(
                        reinterpret_cast<uint64_t*>(
                            empty_barriers[stage_idx]),
                        kCTAMask);
                    if (k_block_idx ==
                        kHidden / BLOCK_K - 1) {
                        cutlass::arch::
                            umma_arrive_multicast_2x1SM(
                                reinterpret_cast<uint64_t*>(
                                    tmem_full_barriers[
                                        accum_stage]),
                                kCTAMask);
                    }
                    __syncwarp();
                }
            });
            if (current_iter > 0) {
                const uint32_t last = current_iter - 1;
                tmem_empty_barriers[
                    last % kNumEpilogueStages]
                    ->wait((last / kNumEpilogueStages) & 1);
            }
        }
    } else if (warp_idx == 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();
    } else if (
        warp_idx <
        (kNumNonEpilogueThreads +
         kNumEpilogueThreads) /
            32) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(
            ptx::ld_shared(tmem_ptr_in_smem) == 0);
        const uint32_t epilogue_warp_idx = warp_idx - 4;
        uint32_t current_iter = 0;
        uint32_t tma_stage_idx = 0;

        for_each_block([&](const uint32_t&, const uint32_t& pool_block_offset,
                           const uint32_t& m_block_idx,
                           const uint32_t& n_block_idx,
                           const uint32_t& valid_m) {
            const uint32_t accum_stage =
                current_iter % kNumEpilogueStages;
            const uint32_t accum_phase =
                (current_iter++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            epilogue::sm100_store_cd_swap_ab<
                BLOCK_M, BLOCK_N, STORE_BLOCK_M,
                STORE_BLOCK_N, kSwizzleCDMode,
                kNumTMAStoreStages, kNumEpilogueThreads,
                GemmType::Normal, false, cd_dtype_t,
                epilogue::transform::EpilogueIdentity>(
                smem_cd, tma_stage_idx,
                accum_stage * UMMA_N,
                (pool_block_offset + m_block_idx) * BLOCK_M,
                n_block_idx * BLOCK_N, 0,
                math::align(valid_m, 16u),
                epilogue_warp_idx, lane_idx,
                tmem_empty_barriers[accum_stage],
                tensor_map_output);
        });

        // The dgrad phase consumes gate/up from global memory. Drain the final
        // two TMA-store stages before publishing phase completion.
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
        __syncwarp();
      }
    }
    if (
        warp_idx >= kDispatchWarpStart &&
        warp_idx < kDispatchWarpStart + kNumDispatchWarps) {
        // The 1024-thread dgrad launch already reserves extra warps. Reuse one
        // warpgroup as the third role instead of increasing the launch size.
        constexpr uint32_t kNumDispatchRegisters =
            kBF16Mode ? 40 : 48;
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        if constexpr (kNumRanks > 1) {
            const uint32_t dispatch_warp_idx =
                warp_idx - kDispatchWarpStart;
            const uint32_t dispatch_thread_idx =
                dispatch_warp_idx * 32 + lane_idx;
            constexpr uint32_t kDispatchGridSyncIndex = 0;
            constexpr uint32_t kDispatchDoneGridSyncIndex = 1;
            constexpr uint32_t kBeforeBackwardPullBarrierTag = 4;
            constexpr uint32_t kDispatchNamedBarrierIdx = 15;

            // All ranks stage their local BF16 grad-y before launch. This
            // system-scope barrier publishes those stores before remote TMA.
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumDispatchThreads,
                kDispatchGridSyncIndex,
                kBeforeBackwardPullBarrierTag>(
                backward_workspace, backward_sym_buffer,
                blockIdx.x, dispatch_thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads,
                        kDispatchNamedBarrierIdx);
                },
                true, true);

            auto* pull_buffer =
                reinterpret_cast<cd_dtype_t*>(smem_buffer) +
                dispatch_warp_idx * kHidden;
            auto* pull_mbarrier =
                dispatch_barriers[dispatch_warp_idx];
            uint32_t pull_mbarrier_phase = 0;
            uint32_t pool_block_offset = 0;

            #pragma unroll
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens =
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                for (uint32_t token_idx =
                         blockIdx.x * kNumDispatchWarps +
                         dispatch_warp_idx;
                     token_idx < num_tokens;
                     token_idx +=
                         kNumSMs * kNumDispatchWarps) {
                    const uint32_t pool_row =
                        pool_block_offset * BLOCK_M +
                        token_idx;
                    const auto metadata =
                        token_src_metadata[pool_row];
                    const auto* remote_grad_y =
                        backward_sym_buffer.map(
                            backward_grad_y +
                                static_cast<uint64_t>(
                                    metadata.token_idx) *
                                    kHidden,
                            metadata.rank_idx);

                    if (cute::elect_one_sync()) {
                        ptx::tma_load_1d(
                            pull_buffer, remote_grad_y,
                            pull_mbarrier,
                            kHidden * sizeof(cd_dtype_t));
                    }
                    __syncwarp();

                    if (cute::elect_one_sync()) {
                        const auto* remote_weight =
                            backward_sym_buffer.map(
                                backward_topk_weights +
                                    static_cast<uint64_t>(
                                        metadata.token_idx) *
                                        num_topk +
                                    metadata.topk_idx,
                                metadata.rank_idx);
                        if constexpr (kBF16Mode) {
                            route_weights_fp32[pool_row] =
                                *remote_weight;
                        } else {
                            route_weights[pool_row] =
                                cd_dtype_t(*remote_weight);
                        }

                        ptx::mbarrier_arrive_and_set_tx(
                            pull_mbarrier,
                            kHidden * sizeof(cd_dtype_t));
                        ptx::mbarrier_wait_and_flip_phase(
                            pull_mbarrier,
                            pull_mbarrier_phase);
                        ptx::tma_store_1d(
                            grad_y_unweighted_output +
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kHidden,
                            pull_buffer,
                            kHidden * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                    }
                    __syncwarp();
                }
                pool_block_offset +=
                    math::ceil_div(num_tokens, BLOCK_M);
            }

            // Stronger than the eventual per-expert handshake: every L2 tile
            // sees every dispatched row. This barrier runs concurrently with
            // recompute and joins only at the phase boundary below.
            comm::grid_sync<
                kNumSMs, kDispatchDoneGridSyncIndex>(
                backward_workspace, blockIdx.x,
                dispatch_thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads,
                        kDispatchNamedBarrierIdx);
                });
        }
    } else if (warp_idx >= 12) {
        // W13 wgrad needs the exact BF16 value represented by the forward
        // FP8+UE8M0 pool.  Produce it while the recompute MMA is running, using
        // otherwise-idle warps.  Padding rows are explicitly zeroed so the
        // k-grouped wgrad mainloop can round K up to 64 without reading the
        // following expert.
        cutlass::arch::warpgroup_reg_dealloc<40>();
        constexpr uint32_t kFirstXPoolWarp = 12;
        constexpr uint32_t kNumXPoolThreads =
            kNumThreads - kFirstXPoolWarp * 32;
        const uint32_t x_thread_idx =
            (warp_idx - kFirstXPoolWarp) * 32 + lane_idx;
        uint32_t pool_block_offset = 0;
        uint32_t global_pool_block = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0; expert_idx < kNumExperts;
             ++expert_idx) {
            const uint32_t num_tokens =
                static_cast<uint32_t>(__ldg(expert_counts + expert_idx));
            const uint32_t num_blocks =
                math::ceil_div(num_tokens, BLOCK_M);
            for (uint32_t m_block_idx = 0; m_block_idx < num_blocks;
                 ++m_block_idx, ++global_pool_block) {
                if (global_pool_block % kNumSMs != blockIdx.x)
                    continue;
                const uint32_t valid_m = cute::min(
                    num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
                const uint32_t pool_block =
                    pool_block_offset + m_block_idx;
                for (uint32_t linear = x_thread_idx;
                     linear < BLOCK_M * kHidden;
                     linear += kNumXPoolThreads) {
                    const uint32_t row = linear / kHidden;
                    const uint32_t col = linear - row * kHidden;
                    const uint32_t pool_row =
                        pool_block * BLOCK_M + row;
                    cd_dtype_t value = cd_dtype_t(0.0f);
                    if (row < valid_m) {
                        if constexpr (kBF16Mode) {
                            const auto metadata =
                                token_src_metadata[pool_row];
                            value = *backward_sym_buffer.map(
                                backward_x +
                                    static_cast<uint64_t>(
                                        metadata.token_idx) *
                                        kHidden +
                                    col,
                                metadata.rank_idx);
                            if constexpr (kNumRanks == 1) {
                                grad_y_unweighted_output[
                                    static_cast<uint64_t>(pool_row) *
                                        kHidden +
                                    col] =
                                    *backward_sym_buffer.map(
                                        backward_grad_y +
                                            static_cast<uint64_t>(
                                                metadata.token_idx) *
                                                kHidden +
                                            col,
                                        metadata.rank_idx);
                                if (col == 0) {
                                    const float weight =
                                        *backward_sym_buffer.map(
                                            backward_topk_weights +
                                                static_cast<uint64_t>(
                                                    metadata.token_idx) *
                                                    num_topk +
                                                metadata.topk_idx,
                                            metadata.rank_idx);
                                    route_weights_fp32[pool_row] =
                                        weight;
                                }
                            }
                        } else {
                            const uint32_t idx = row % BLOCK_M;
                            const uint32_t sf_token =
                                pool_block * SF_BLOCK_M +
                                (idx & ~127u) +
                                (idx & 31u) * 4 +
                                ((idx >> 5) & 3u);
                            const uint32_t sf_group = col / 128;
                            const uint32_t sf_byte = (col / 32) & 3u;
                            const uint32_t packed_sf =
                                acts_sf_ptr[
                                    sf_group * acts_sf_stride +
                                    sf_token];
                            const uint32_t exponent =
                                (packed_sf >> (sf_byte * 8)) &
                                0xffu;
                            const uint32_t scale_bits =
                                exponent << 23;
                            const float scale =
                                *reinterpret_cast<const float*>(
                                    &scale_bits);
                            value = cd_dtype_t(
                                static_cast<float>(
                                    acts_ptr[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            kHidden +
                                        col]) *
                                scale);
                        }
                    }
                    x_pool_output[
                        static_cast<uint64_t>(pool_row) * kHidden +
                        col] = value;
                }
            }
            pool_block_offset += num_blocks;
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<24>();
    }

    {
        __syncthreads();
        if constexpr (kBF16Mode && kNumRanks == 1) {
            // In single-rank BF16 mode the x-pool warps also stage grad-y.
            // Their pool-block assignment is independent of the dgrad tile
            // assignment, so a cluster barrier is insufficient before W2
            // dgrad starts consuming the completed expert pool.
            constexpr uint32_t kLocalDispatchDoneGridSyncIndex = 1;
            comm::grid_sync<
                kNumSMs, kLocalDispatchDoneGridSyncIndex>(
                backward_workspace, blockIdx.x, threadIdx.x,
                []() { __syncthreads(); });
        }
        if constexpr (kBF16Mode) {
            uint32_t grad_pool_block_offset = 0;
            #pragma unroll
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens =
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                for (uint64_t linear =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear <
                         static_cast<uint64_t>(num_tokens) *
                             kHidden;
                     linear +=
                         static_cast<uint64_t>(kNumSMs) *
                         kNumThreads) {
                    const uint32_t token_idx =
                        linear / kHidden;
                    const uint32_t col =
                        linear -
                        static_cast<uint64_t>(token_idx) *
                            kHidden;
                    const uint32_t pool_row =
                        grad_pool_block_offset * BLOCK_M +
                        token_idx;
                    grad_ye_output[
                        static_cast<uint64_t>(pool_row) *
                            kHidden +
                        col] =
                        kRouteWeightMode ==
                                RouteWeightMode::PostDown
                        ? cd_dtype_t(
                              static_cast<float>(
                                  grad_y_unweighted_output[
                                      static_cast<uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col]) *
                              route_weights_fp32[pool_row])
                        : grad_y_unweighted_output[
                              static_cast<uint64_t>(
                                  pool_row) *
                                  kHidden +
                              col];
                }
                grad_pool_block_offset +=
                    math::ceil_div(num_tokens, BLOCK_M);
            }
            constexpr uint32_t kW2GradInputGridSyncIndex = 0;
            comm::grid_sync<
                kNumSMs, kW2GradInputGridSyncIndex>(
                backward_workspace, blockIdx.x, threadIdx.x,
                []() { __syncthreads(); });
        }
        if constexpr (kNumRanks > 1) {
            if (direct_remote_grad_x) {
                // backward_grad_y aliases combine plane zero. All ranks must
                // finish remotely pulling it before any W13 dgrad epilogue
                // reuses the combine planes for direct grad-x writes.
                constexpr uint32_t kBeforeDirectGradXGridSyncIndex = 2;
                constexpr uint32_t kBeforeDirectGradXBarrierTag = 7;
                comm::nvlink_barrier<
                    kNumRanks, kNumSMs, kNumThreads,
                    kBeforeDirectGradXGridSyncIndex,
                    kBeforeDirectGradXBarrierTag>(
                    backward_workspace, backward_sym_buffer,
                    blockIdx.x, threadIdx.x,
                    []() { __syncthreads(); });

                // Plane zero held grad-y during reverse dispatch. Clear it
                // after every rank has completed its pulls; otherwise masked
                // top-k slot zero would contribute stale grad-y to grad-x.
                auto* combine_buffer = const_cast<cd_dtype_t*>(
                    backward_grad_y);
                const uint64_t num_plane_values =
                    static_cast<uint64_t>(
                        backward_workspace
                            .num_max_tokens_per_rank) *
                    kHidden;
                for (uint64_t linear =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear < num_plane_values;
                     linear +=
                         static_cast<uint64_t>(kNumSMs) *
                         kNumThreads) {
                    combine_buffer[linear] = cd_dtype_t(0.0f);
                }

                // Do not let a rank remotely write direct grad-x until every
                // destination rank has finished clearing its local plane.
                constexpr uint32_t kAfterGradYClearGridSyncIndex = 3;
                constexpr uint32_t kAfterGradYClearBarrierTag = 8;
                comm::nvlink_barrier<
                    kNumRanks, kNumSMs, kNumThreads,
                    kAfterGradYClearGridSyncIndex,
                    kAfterGradYClearBarrierTag>(
                    backward_workspace, backward_sym_buffer,
                    blockIdx.x, threadIdx.x,
                    []() { __syncthreads(); });
            }
        }
        if (warp_idx >= kDispatchWarpStart &&
            warp_idx <
                kDispatchWarpStart +
                    kNumDispatchWarps) {
            // Dispatch used 48 registers; transition down to the common
            // dgrad epilogue budget with dealloc, not reg_alloc (allocating a
            // lower count is an illegal instruction on SM100).
            cutlass::arch::warpgroup_reg_dealloc<40>();
        } else if (warp_idx >= kDispatchWarpStart) {
            cutlass::arch::warpgroup_reg_alloc<40>();
        }
        const auto for_each_dgrad_block = [&](const auto& func) {
            uint32_t next_assigned_block = blockIdx.x;
            uint32_t global_block = 0;
            uint32_t pool_block_offset = 0;
            #pragma unroll
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens =
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                const uint32_t num_m_blocks =
                    math::ceil_div(num_tokens, BLOCK_M);
                const uint32_t expert_blocks =
                    num_m_blocks * kNumDgradBlockNs;
                const uint32_t expert_end =
                    global_block + expert_blocks;

                while (next_assigned_block < global_block)
                    next_assigned_block += kNumSMs;
                while (next_assigned_block < expert_end) {
                    const uint32_t local_block =
                        next_assigned_block - global_block;
                    const uint32_t m_block_idx =
                        local_block / kNumDgradBlockNs;
                    const uint32_t n_block_idx =
                        local_block -
                        m_block_idx * kNumDgradBlockNs;
                    const uint32_t valid_m = cute::min(
                        num_tokens - m_block_idx * BLOCK_M,
                        BLOCK_M);
                    func(
                        expert_idx, pool_block_offset,
                        m_block_idx, n_block_idx, valid_m);
                    next_assigned_block += kNumSMs;
                }
                global_block = expert_end;
                pool_block_offset += num_m_blocks;
            }
        };

        comm::cluster_sync_with_relaxed_arrive();

        comm::cluster_sync_with_relaxed_arrive();

        // Reinitialize the drained pipelines in-place.  The FP32 accumulator
        // columns are phase-aliased; dgrad does not need the SFA/SFB columns.
        if (warp_idx == 0 && cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                // A and transposed-W2 TMA warps in both CTAs.
                full_barriers[i]->init(4);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(
                    2 * kNumDgradEpilogueThreads);
            }
            cutlass::arch::fence_barrier_init();
        }
        comm::cluster_sync_with_relaxed_arrive();

        stage_idx = 0;
        phase = 0;
        if (warp_idx == 0) {
            // BF16 grad_y producer.  The 2-SM TMA instruction writes each
            // CTA's half-M operand and completes transactions on CTA0's
            // cluster barrier.
            for_each_dgrad_block(
                [&](const uint32_t&, const uint32_t& pool_block_offset,
                    const uint32_t& m_block_idx, const uint32_t&,
                    const uint32_t& valid_m) {
                    const uint32_t pool_block_idx =
                        pool_block_offset + m_block_idx;
                    #pragma unroll 1
                    for (uint32_t k_block_idx = 0;
                         k_block_idx < kHidden / DGRAD_BLOCK_K;
                         advance_pipeline(k_block_idx)) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        uint32_t m_idx = pool_block_idx * BLOCK_M;
                        if (!is_leader_cta)
                            m_idx += math::align(valid_m, 16u) / 2;
                        if (cute::elect_one_sync()) {
                            tma::copy<
                                DGRAD_BLOCK_K, LOAD_BLOCK_M,
                                DGRAD_BLOCK_K * sizeof(cd_dtype_t),
                                cd_dtype_t>(
                                &tensor_map_grad_ye,
                                full_barriers[stage_idx],
                                smem_dgrad_a[stage_idx],
                                k_block_idx * DGRAD_BLOCK_K,
                                m_idx, 2);
                            if (is_leader_cta) {
                                full_barriers[stage_idx]
                                    ->arrive_and_expect_tx(
                                        SMEM_A_SIZE_PER_STAGE * 2);
                            } else {
                                full_barriers[stage_idx]->arrive(0u);
                            }
                        }
                        __syncwarp();
                    }
                });
        } else if (warp_idx == 1) {
            // Load the in-kernel dequantized transposed W2 workspace.  It is
            // shared by all M tiles for this expert instead of reconverting
            // the same packed weights for every token block.
            for_each_dgrad_block(
                [&](const uint32_t& expert_idx, const uint32_t&,
                    const uint32_t&, const uint32_t& n_block_idx,
                    const uint32_t&) {
                    #pragma unroll 1
                    for (uint32_t k_block_idx = 0;
                         k_block_idx < kHidden / DGRAD_BLOCK_K;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t weight_tile_idx =
                            (expert_idx *
                                 (kHidden / DGRAD_BLOCK_K) +
                             k_block_idx) *
                                kNumDgradBlockNs +
                            n_block_idx;
                        if constexpr (!kBF16Mode) {
                            while (ptx::ld_acq(
                                       weight_tile_states +
                                       weight_tile_idx) !=
                                   launch_epoch) {
                            }
                        }
                        constexpr bool weight_tile_ready = true;
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        if (weight_tile_ready) {
                            if (cute::elect_one_sync()) {
                                tma::copy<
                                    LOAD_BLOCK_N,
                                    DGRAD_BLOCK_K,
                                    DGRAD_BLOCK_K *
                                        sizeof(
                                            dgrad_b_dtype_t),
                                    dgrad_b_dtype_t>(
                                    &tensor_map_w2_dequant,
                                    full_barriers[stage_idx],
                                    smem_dgrad_b[stage_idx],
                                    n_block_idx *
                                        BLOCK_N,
                                    expert_idx *
                                            kHidden +
                                        k_block_idx *
                                            DGRAD_BLOCK_K,
                                    2);
                                if (is_leader_cta) {
                                    full_barriers[stage_idx]
                                        ->arrive_and_expect_tx(
                                            SMEM_B_SIZE_PER_STAGE *
                                            2);
                                } else {
                                    full_barriers[stage_idx]
                                        ->arrive(0u);
                                }
                            }
                        } else {
                            constexpr uint32_t
                                kPairsPerTile =
                                    LOAD_BLOCK_N *
                                    DGRAD_BLOCK_K / 2;
                            auto* smem_b_bytes =
                                reinterpret_cast<uint8_t*>(
                                    smem_dgrad_b[stage_idx]);
                            for (uint32_t pair_idx = lane_idx;
                                 pair_idx < kPairsPerTile;
                                 pair_idx += 32) {
                                const uint32_t local_k =
                                    pair_idx /
                                    (LOAD_BLOCK_N / 2);
                                const uint32_t
                                    local_n_pair =
                                        pair_idx %
                                        (LOAD_BLOCK_N / 2);
                                const uint32_t global_k =
                                    k_block_idx *
                                        DGRAD_BLOCK_K +
                                    local_k;
                                const uint32_t
                                    global_n_pair =
                                        n_block_idx *
                                            (LOAD_BLOCK_N /
                                             2) +
                                        local_n_pair;
                                const uint8_t packed =
                                    static_cast<uint8_t>(
                                        __ldg(
                                            w2_weights +
                                            (static_cast<
                                                 uint64_t>(
                                                 expert_idx) *
                                                 kHidden +
                                             global_k) *
                                                (kIntermediateHidden /
                                                 2) +
                                            global_n_pair));
                                const float scale = __ldg(
                                    w2_scales +
                                    (static_cast<uint64_t>(
                                         expert_idx) *
                                         kHidden +
                                     global_k) *
                                        (kIntermediateHidden /
                                         32) +
                                    n_block_idx *
                                        (LOAD_BLOCK_N / 32) +
                                    (local_n_pair * 2) /
                                        32);
                                uint32_t fp16x2;
                                asm volatile(
                                    "{\n"
                                    ".reg .b8 fp4;\n"
                                    ".reg .b8 unused1, unused2, unused3;\n"
                                    "mov.b32 {fp4, unused1, unused2, unused3}, %1;\n"
                                    "cvt.rn.f16x2.e2m1x2 %0, fp4;\n"
                                    "}\n"
                                    : "=r"(fp16x2)
                                    : "r"(
                                          static_cast<
                                              uint32_t>(
                                              packed)));
                                auto value_pair =
                                    *reinterpret_cast<
                                        __half2*>(&fp16x2);
                                value_pair = __hmul2(
                                    value_pair,
                                    __float2half2_rn(
                                        scale));
                                const auto
                                    value_pair_bf16 =
                                        __float22bfloat162_rn(
                                            __half22float2(
                                                value_pair));
                                const uint32_t
                                    scaled_pair =
                                        *reinterpret_cast<
                                            const uint32_t*>(
                                            &value_pair_bf16);
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < 2; ++i) {
                                    const uint32_t local_n =
                                        local_n_pair * 2 + i;
                                    const uint32_t row =
                                        local_n & 7;
                                    const uint32_t col_byte =
                                        local_k *
                                        sizeof(
                                            dgrad_b_dtype_t);
                                    const uint32_t
                                        byte_offset =
                                            (local_n >> 3) *
                                                8 * 128 +
                                            row * 128 +
                                            ((col_byte >> 4) ^
                                             row) *
                                                16 +
                                            (col_byte & 15);
                                    *reinterpret_cast<
                                        uint16_t*>(
                                        smem_b_bytes +
                                        byte_offset) =
                                        static_cast<
                                            uint16_t>(
                                            scaled_pair >>
                                            (i * 16));
                                }
                            }
                            cutlass::arch::
                                fence_view_async_shared();
                            if (cute::elect_one_sync())
                                full_barriers[stage_idx]
                                    ->arrive(0u);
                        }
                        __syncwarp();
                    }
                });
        } else if (warp_idx == 2) {
            if (is_leader_cta) {
                auto instr_desc =
                    cute::UMMA::make_instr_desc<
                        dgrad_b_dtype_t, cd_dtype_t, float,
                        UMMA_M, UMMA_N,
                        cute::UMMA::Major::MN,
                        cute::UMMA::Major::K>();
                auto a_desc = mma::sm100::make_umma_desc<
                    cute::UMMA::Major::K, LOAD_BLOCK_M,
                    DGRAD_BLOCK_K,
                    DGRAD_BLOCK_K * sizeof(cd_dtype_t)>(
                    smem_dgrad_a[0], 0, 0);
                auto b_desc = mma::sm100::make_umma_desc<
                    cute::UMMA::Major::MN, LOAD_BLOCK_N,
                    DGRAD_BLOCK_K,
                    DGRAD_BLOCK_K * sizeof(dgrad_b_dtype_t)>(
                    smem_dgrad_b[0], 0, 0);
                const uint32_t a_desc_lo = lane_idx < kNumStages
                    ? a_desc.lo +
                          lane_idx * SMEM_A_SIZE_PER_STAGE / 16
                    : 0;
                const uint32_t b_desc_lo = lane_idx < kNumStages
                    ? b_desc.lo +
                          lane_idx * SMEM_B_SIZE_PER_STAGE / 16
                    : 0;
                uint32_t current_iter = 0;

                for_each_dgrad_block(
                    [&](const uint32_t&, const uint32_t&,
                        const uint32_t&, const uint32_t&,
                        const uint32_t& valid_m) {
                        mma::sm100::update_instr_desc_with_umma_n(
                            instr_desc,
                            math::align(valid_m, 16u));
                        const auto runtime_instr_desc =
                            cute::UMMA::make_runtime_instr_desc(
                                instr_desc);
                        const uint32_t accum_stage =
                            current_iter % kNumEpilogueStages;
                        const uint32_t accum_phase =
                            (current_iter++ /
                             kNumEpilogueStages) &
                            1;
                        tmem_empty_barriers[accum_stage]->wait(
                            accum_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        #pragma unroll 1
                        for (uint32_t k_block_idx = 0;
                             k_block_idx <
                                 kHidden / DGRAD_BLOCK_K;
                             advance_pipeline(k_block_idx)) {
                            full_barriers[stage_idx]->wait(phase);
                            ptx::tcgen05_after_thread_sync();
                            const uint32_t a_desc_base =
                                ptx::exchange(
                                    a_desc_lo, stage_idx);
                            const uint32_t b_desc_base =
                                ptx::exchange(
                                    b_desc_lo, stage_idx);
                            if (cute::elect_one_sync()) {
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k <
                                         DGRAD_BLOCK_K /
                                             DGRAD_UMMA_K;
                                     ++k) {
                                    a_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::K,
                                                LOAD_BLOCK_M,
                                                DGRAD_BLOCK_K *
                                                    sizeof(
                                                        cd_dtype_t),
                                                cd_dtype_t>(
                                                a_desc_base, 0,
                                                k *
                                                    DGRAD_UMMA_K);
                                    b_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::MN,
                                                LOAD_BLOCK_N,
                                                DGRAD_BLOCK_K *
                                                    sizeof(
                                                        dgrad_b_dtype_t),
                                                dgrad_b_dtype_t>(
                                                b_desc_base, 0,
                                                k *
                                                    DGRAD_UMMA_K);
                                    ptx::
                                        SM100_MMA_F16BF16_2x1SM_SS::
                                            fma(
                                                b_desc, a_desc,
                                                accum_stage *
                                                    UMMA_N,
                                                k_block_idx > 0 ||
                                                    k > 0,
                                                runtime_instr_desc);
                                }
                            }
                            __syncwarp();
                            constexpr uint16_t kCTAMask = 0x3;
                            cutlass::arch::
                                umma_arrive_multicast_2x1SM(
                                    reinterpret_cast<uint64_t*>(
                                        empty_barriers[
                                            stage_idx]),
                                    kCTAMask);
                            if (k_block_idx ==
                                kHidden / DGRAD_BLOCK_K - 1) {
                                cutlass::arch::
                                    umma_arrive_multicast_2x1SM(
                                        reinterpret_cast<uint64_t*>(
                                            tmem_full_barriers[
                                                accum_stage]),
                                        kCTAMask);
                            }
                            __syncwarp();
                        }
                    });
                if (current_iter > 0) {
                    const uint32_t last = current_iter - 1;
                    tmem_empty_barriers[
                        last % kNumEpilogueStages]
                        ->wait(
                            (last / kNumEpilogueStages) & 1);
                }
            }
        } else if (warp_idx >= 4) {
            const uint32_t epilogue_warp_idx = warp_idx - 4;
            const uint32_t epilogue_thread_idx =
                epilogue_warp_idx * 32 + lane_idx;
            uint32_t current_iter = 0;

            for_each_dgrad_block(
                [&](const uint32_t&, const uint32_t& pool_block_offset,
                    const uint32_t& m_block_idx,
                    const uint32_t& n_block_idx,
                    const uint32_t& valid_m) {
                    const uint32_t accum_stage =
                        current_iter % kNumEpilogueStages;
                    const uint32_t accum_phase =
                        (current_iter++ /
                         kNumEpilogueStages) &
                        1;
                    tmem_full_barriers[accum_stage]->wait(
                        accum_phase);
                    ptx::tcgen05_after_thread_sync();
                    const uint32_t effective_m =
                        math::align(valid_m, 16u);

                    for (uint32_t s = 0;
                         s < effective_m / STORE_BLOCK_M; ++s) {
                        cutlass::arch::NamedBarrier::sync(
                            kNumDgradEpilogueThreads, 0);
                        if (epilogue_warp_idx <
                            kNumEpilogueThreads / 32) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < STORE_BLOCK_M / 8; ++i) {
                                const uint32_t tmem_addr =
                                    accum_stage * UMMA_N +
                                    s * STORE_BLOCK_M + i * 8;
                                uint32_t values[8];
                                cute::SM100_TMEM_LOAD_16dp256b1x::
                                    copy(
                                        tmem_addr, values[0],
                                        values[1], values[2],
                                        values[3]);
                                cute::SM100_TMEM_LOAD_16dp256b1x::
                                    copy(
                                        tmem_addr | 0x00100000,
                                        values[4], values[5],
                                        values[6], values[7]);
                                cutlass::arch::
                                    fence_view_async_tmem_load();

                                constexpr uint32_t kBankBytes = 16;
                                const uint32_t outer_atom =
                                    (epilogue_warp_idx / 2) *
                                    STORE_BLOCK_M * 128;
                                const uint32_t inner_atom =
                                    i * 8 * 128;
                                const uint32_t row = lane_idx % 8;
                                const uint32_t col =
                                    (epilogue_warp_idx % 2) * 4 +
                                    lane_idx / 8;
                                auto* smem_ptr =
                                    reinterpret_cast<uint8_t*>(
                                        smem_cd[0]) +
                                    outer_atom + inner_atom +
                                    row * (kBankBytes * 8) +
                                    (col ^ row) * kBankBytes;
                                ptx::SM90_U32x4_STSM_T<int>::copy(
                                    math::cast_into_bf16_and_pack(
                                        values[0], values[1]),
                                    math::cast_into_bf16_and_pack(
                                        values[2], values[3]),
                                    math::cast_into_bf16_and_pack(
                                        values[4], values[5]),
                                    math::cast_into_bf16_and_pack(
                                        values[6], values[7]),
                                    smem_ptr);
                            }
                        }
                        cutlass::arch::NamedBarrier::sync(
                            kNumDgradEpilogueThreads, 0);

                        #pragma unroll
                        for (uint32_t linear =
                                 epilogue_thread_idx;
                             linear <
                                 STORE_BLOCK_M * BLOCK_N;
                             linear +=
                                 kNumDgradEpilogueThreads) {
                            const uint32_t row =
                                linear / BLOCK_N;
                            const uint32_t n =
                                linear - row * BLOCK_N;
                            const uint32_t local_m =
                                s * STORE_BLOCK_M + row;
                            if (local_m >= valid_m)
                                continue;

                            const uint32_t n_atom = n / 64;
                            const uint32_t n_in_atom =
                                n - n_atom * 64;
                            const uint32_t row_in_atom =
                                row & 7;
                            const uint32_t smem_byte_offset =
                                n_atom *
                                    STORE_BLOCK_M * 128 +
                                (row >> 3) * 8 * 128 +
                                row_in_atom * 128 +
                                ((n_in_atom >> 3) ^
                                 row_in_atom) *
                                    16 +
                                (n_in_atom & 7) *
                                    sizeof(cd_dtype_t);
                            const cd_dtype_t grad_h_w2 =
                                *reinterpret_cast<
                                    cd_dtype_t*>(
                                    reinterpret_cast<
                                        uint8_t*>(
                                        smem_cd[0]) +
                                    smem_byte_offset);
                            const uint32_t pool_row =
                                (pool_block_offset +
                                 m_block_idx) *
                                    BLOCK_M +
                                local_m;
                            const uint32_t hidden_col =
                                n_block_idx * BLOCK_N + n;
                            const float route_weight =
                                kBF16Mode
                                ? route_weights_fp32[pool_row]
                                : static_cast<float>(
                                      route_weights[pool_row]);
                            const cd_dtype_t grad_h_bf16 =
                                kBF16Mode &&
                                        kRouteWeightMode ==
                                            RouteWeightMode::PostDown
                                ? grad_h_w2
                                : cd_dtype_t(
                                      static_cast<float>(
                                          grad_h_w2) *
                                      route_weight);
                            const float grad_h =
                                static_cast<float>(grad_h_bf16);
                            // This is the W2 dgrad output before any pre-down
                            // route multiplication. In post-down mode its GEMM
                            // input was already weighted BF16 grad-y.
                            grad_h_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kIntermediateHidden +
                                hidden_col] =
                                grad_h_w2;
                            const uint32_t chunk =
                                hidden_col / 8;
                            const uint32_t in_chunk =
                                hidden_col & 7;
                            const uint32_t gate_col =
                                kBF16Mode
                                ? hidden_col
                                : chunk * 16 + in_chunk;
                            const uint32_t up_col =
                                kBF16Mode
                                ? kIntermediateHidden +
                                      hidden_col
                                : gate_col + 8;
                            const float gate_unclamped =
                                static_cast<float>(
                                    gate_up_output[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            (2 *
                                             kIntermediateHidden) +
                                        gate_col]);
                            const float up_unclamped =
                                static_cast<float>(
                                    gate_up_output[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            (2 *
                                             kIntermediateHidden) +
                                        up_col]);

                            const bool has_activation_clamp =
                                kBF16Mode
                                ? activation_limit !=
                                      cute::numeric_limits<
                                          float>::infinity()
                                : activation_limit > 0.0f;
                            const bool gate_in_range =
                                !has_activation_clamp ||
                                gate_unclamped <=
                                    activation_limit;
                            const bool up_in_range =
                                !has_activation_clamp ||
                                (up_unclamped >=
                                     -activation_limit &&
                                 up_unclamped <=
                                     activation_limit);
                            const float gate =
                                has_activation_clamp
                                ? cute::min(
                                      gate_unclamped,
                                      activation_limit)
                                : gate_unclamped;
                            const float up =
                                has_activation_clamp
                                ? cute::min(
                                      cute::max(
                                          up_unclamped,
                                          -activation_limit),
                                      activation_limit)
                                : up_unclamped;
                            float z;
                            float dz_dgate;
                            if constexpr (
                                kActivationType ==
                                ActivationType::GeGLU) {
                                constexpr float kAlpha =
                                    1.5957691216057308f;
                                constexpr float kBeta = 0.044715f;
                                const float gate_sq = gate * gate;
                                z = kAlpha * gate *
                                    (1.0f + kBeta * gate_sq);
                                dz_dgate = kAlpha *
                                    (1.0f +
                                     3.0f * kBeta * gate_sq);
                            } else {
                                z = gate;
                                dz_dgate = 1.0f;
                            }
                            const float neg_exp =
                                !kBF16Mode || kFastMath
                                ? __expf(-z)
                                : expf(-z);
                            const float sig =
                                1.0f / (1.0f + neg_exp);
                            const float activated_gate =
                                gate * sig;
                            const float h_act =
                                activated_gate * up;
                            const float activation_grad =
                                sig +
                                gate * sig *
                                    (1.0f - sig) *
                                    dz_dgate;
                            const float grad_gate =
                                gate_in_range
                                ? grad_h * up *
                                      activation_grad
                                : 0.0f;
                            const float grad_up =
                                up_in_range
                                ? grad_h * activated_gate
                                : 0.0f;
                            const cd_dtype_t h_act_bf16 =
                                cd_dtype_t(h_act);
                            h_act_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kIntermediateHidden +
                                hidden_col] =
                                h_act_bf16;
                            h_weighted_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kIntermediateHidden +
                                hidden_col] =
                                kBF16Mode &&
                                        kRouteWeightMode ==
                                            RouteWeightMode::PostDown
                                ? h_act_bf16
                                : cd_dtype_t(
                                      static_cast<float>(
                                          h_act_bf16) *
                                      route_weight);
                            grad_gate_up_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    (2 *
                                     kIntermediateHidden) +
                                hidden_col] =
                                cd_dtype_t(grad_gate);
                            grad_gate_up_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    (2 *
                                     kIntermediateHidden) +
                                kIntermediateHidden +
                                hidden_col] =
                                cd_dtype_t(grad_up);
                        }
                    }
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage]->arrive(0u);
                });

        }

        __syncthreads();
        if constexpr (kCompileW13Dgrad) {
            // W13 dgrad consumes grad_gate_up rows produced by every CTA in
            // the preceding L2-dgrad/SwiGLU phase. Cluster synchronization is
            // insufficient here: an early cluster can otherwise read rows
            // whose owning cluster has not stored them yet.
            constexpr uint32_t kNumW13WeightTileStates =
                kNumExperts *
                ((2 * kIntermediateHidden) / DGRAD_BLOCK_K) *
                kNumW13DgradBlockNs;
            auto* phase_count =
                weight_tile_states + kNumW2WeightTileStates +
                kNumW13WeightTileStates;
            auto* phase_sense = phase_count + 1;
            if (threadIdx.x == 0) {
                const uint32_t old_sense =
                    atomicAdd(phase_sense, 0u);
                __threadfence();
                const uint32_t ticket =
                    atomicAdd(phase_count, 1u);
                if (ticket == kNumSMs - 1) {
                    atomicExch(phase_count, 0u);
                    __threadfence();
                    atomicAdd(phase_sense, 1u);
                } else {
                    while (ptx::ld_acq(phase_sense) ==
                           old_sense) {
                    }
                }
            }
            __syncthreads();

            if constexpr (kBF16Mode) {
                // The activation epilogue spans multiple N-tile CTAs. Reduce
                // each route term only after all tiles are visible so the
                // router gradient has a fixed FP32 summation order instead of
                // depending on cross-CTA atomic arrival order.
                uint32_t route_pool_block_offset = 0;
                #pragma unroll
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(expert_counts + expert_idx));
                    for (uint32_t token_idx =
                             blockIdx.x * kNumThreads +
                             threadIdx.x;
                         token_idx < num_tokens;
                         token_idx += kNumSMs * kNumThreads) {
                        const uint32_t pool_row =
                            route_pool_block_offset * BLOCK_M +
                            token_idx;
                        float grad_route = 0.0f;
                        if constexpr (
                            kRouteWeightMode ==
                            RouteWeightMode::PostDown) {
                            for (uint32_t col = 0;
                                 col < kHidden; ++col) {
                                const float grad_y =
                                    static_cast<float>(
                                        grad_y_unweighted_output[
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            col]);
                                const float down =
                                    static_cast<float>(
                                        down_unweighted_output[
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            col]);
                                grad_route = __fadd_rn(
                                    grad_route,
                                    __fmul_rn(grad_y, down));
                            }
                        } else {
                            for (uint32_t col = 0;
                                 col < kIntermediateHidden;
                                 ++col) {
                                const float grad_h =
                                    static_cast<float>(
                                        grad_h_output[
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kIntermediateHidden +
                                            col]);
                                const float h_act =
                                    static_cast<float>(
                                        h_act_output[
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kIntermediateHidden +
                                            col]);
                                grad_route = __fadd_rn(
                                    grad_route,
                                    __fmul_rn(grad_h, h_act));
                            }
                        }
                        grad_route_output[pool_row] =
                            grad_route;
                    }
                    route_pool_block_offset +=
                        math::ceil_div(num_tokens, BLOCK_M);
                }
            }

            // Phase 3: dequantize canonical [W1; W3] once per launch, then
            // consume it as the transposed BF16 operand for W13 dgrad.  This
            // phase starts only after L2 dgrad/SwiGLU has drained both TMEM
            // accumulator stages, so the same 512-column allocation is reused.

            const uint32_t w13_launch_epoch =
                launch_epoch ^ 0x80000000u;

            const auto for_each_w13_dgrad_block =
                [&](const auto& func) {
                    uint32_t next_assigned_block =
                        blockIdx.x;
                    uint32_t global_block = 0;
                    uint32_t pool_block_offset = 0;
                    #pragma unroll
                    for (uint32_t expert_idx = 0;
                         expert_idx < kNumExperts;
                         ++expert_idx) {
                        const uint32_t num_tokens =
                            static_cast<uint32_t>(
                                __ldg(
                                    expert_counts +
                                    expert_idx));
                        const uint32_t num_m_blocks =
                            math::ceil_div(
                                num_tokens, BLOCK_M);
                        const uint32_t expert_blocks =
                            num_m_blocks *
                            kNumW13DgradBlockNs;
                        const uint32_t expert_end =
                            global_block +
                            expert_blocks;

                        while (next_assigned_block <
                               global_block)
                            next_assigned_block +=
                                kNumSMs;
                        while (next_assigned_block <
                               expert_end) {
                            const uint32_t local_block =
                                next_assigned_block -
                                global_block;
                            const uint32_t
                                m_block_idx =
                                    local_block /
                                    kNumW13DgradBlockNs;
                            const uint32_t
                                n_block_idx =
                                    local_block -
                                    m_block_idx *
                                        kNumW13DgradBlockNs;
                            const uint32_t valid_m =
                                cute::min(
                                    num_tokens -
                                        m_block_idx *
                                            BLOCK_M,
                                    BLOCK_M);
                            func(
                                expert_idx,
                                pool_block_offset,
                                m_block_idx,
                                n_block_idx,
                                valid_m);
                            next_assigned_block +=
                                kNumSMs;
                        }
                        global_block = expert_end;
                        pool_block_offset +=
                            num_m_blocks;
                    }
                };

            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0 &&
                cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0;
                     i < kNumStages; ++i) {
                    full_barriers[i]->init(4);
                    empty_barriers[i]->init(1);
                }
                #pragma unroll
                for (uint32_t i = 0;
                     i < kNumEpilogueStages; ++i) {
                    tmem_full_barriers[i]->init(1);
                    tmem_empty_barriers[i]->init(
                        2 *
                        kNumDgradEpilogueThreads);
                }
                cutlass::arch::fence_barrier_init();
            }
            comm::cluster_sync_with_relaxed_arrive();

            stage_idx = 0;
            phase = 0;
            if (warp_idx == 0) {
                for_each_w13_dgrad_block(
                    [&](const uint32_t&,
                        const uint32_t&
                            pool_block_offset,
                        const uint32_t&
                            m_block_idx,
                        const uint32_t&,
                        const uint32_t& valid_m) {
                        const uint32_t pool_block_idx =
                            pool_block_offset +
                            m_block_idx;
                        #pragma unroll 1
                        for (uint32_t k_block_idx =
                                 0;
                             k_block_idx <
                                 (2 *
                                  kIntermediateHidden) /
                                     DGRAD_BLOCK_K;
                             advance_pipeline(
                                 k_block_idx)) {
                            empty_barriers[stage_idx]
                                ->wait(phase ^ 1);
                            uint32_t m_idx =
                                pool_block_idx *
                                BLOCK_M;
                            if (!is_leader_cta)
                                m_idx +=
                                    math::align(
                                        valid_m, 16u) /
                                    2;
                            if (cute::elect_one_sync()) {
                                tma::copy<
                                    DGRAD_BLOCK_K,
                                    LOAD_BLOCK_M,
                                    DGRAD_BLOCK_K *
                                        sizeof(
                                            cd_dtype_t),
                                    cd_dtype_t>(
                                    &tensor_map_grad_gate_up,
                                    full_barriers[
                                        stage_idx],
                                    smem_dgrad_a[
                                        stage_idx],
                                    k_block_idx *
                                        DGRAD_BLOCK_K,
                                    m_idx, 2);
                                if (is_leader_cta) {
                                    full_barriers[
                                        stage_idx]
                                        ->arrive_and_expect_tx(
                                            SMEM_A_SIZE_PER_STAGE *
                                            2);
                                } else {
                                    full_barriers[
                                        stage_idx]
                                        ->arrive(0u);
                                }
                            }
                            __syncwarp();
                        }
                    });
            } else if (warp_idx == 1) {
                for_each_w13_dgrad_block(
                    [&](const uint32_t& expert_idx,
                        const uint32_t&,
                        const uint32_t&,
                        const uint32_t&
                            n_block_idx,
                        const uint32_t&) {
                        #pragma unroll 1
                        for (uint32_t k_block_idx =
                                 0;
                             k_block_idx <
                                 (2 *
                                  kIntermediateHidden) /
                                     DGRAD_BLOCK_K;
                             advance_pipeline(
                                 k_block_idx)) {
                            const uint32_t
                                weight_tile_idx =
                                    (expert_idx *
                                         ((2 *
                                           kIntermediateHidden) /
                                          DGRAD_BLOCK_K) +
                                     k_block_idx) *
                                        kNumW13DgradBlockNs +
                                    n_block_idx;
                            if constexpr (!kBF16Mode) {
                                while (ptx::ld_acq(
                                           weight_tile_states +
                                           kNumW2WeightTileStates +
                                           weight_tile_idx) !=
                                       w13_launch_epoch) {
                                }
                            }
                            empty_barriers[stage_idx]
                                ->wait(phase ^ 1);
                            if (cute::elect_one_sync()) {
                                tma::copy<
                                    LOAD_BLOCK_N,
                                    DGRAD_BLOCK_K,
                                    DGRAD_BLOCK_K *
                                        sizeof(
                                            dgrad_b_dtype_t),
                                    dgrad_b_dtype_t>(
                                    &tensor_map_w13_dequant,
                                    full_barriers[
                                        stage_idx],
                                    smem_dgrad_b[
                                        stage_idx],
                                    n_block_idx *
                                        BLOCK_N,
                                    expert_idx *
                                            (2 *
                                             kIntermediateHidden) +
                                        k_block_idx *
                                            DGRAD_BLOCK_K,
                                    2);
                                if (is_leader_cta) {
                                    full_barriers[
                                        stage_idx]
                                        ->arrive_and_expect_tx(
                                            SMEM_B_SIZE_PER_STAGE *
                                            2);
                                } else {
                                    full_barriers[
                                        stage_idx]
                                        ->arrive(0u);
                                }
                            }
                            __syncwarp();
                        }
                    });
            } else if (warp_idx == 2) {
                if (is_leader_cta) {
                    auto instr_desc =
                        cute::UMMA::make_instr_desc<
                            dgrad_b_dtype_t,
                            cd_dtype_t, float,
                            UMMA_M, UMMA_N,
                            cute::UMMA::Major::MN,
                            cute::UMMA::Major::K>();
                    auto a_desc =
                        mma::sm100::make_umma_desc<
                            cute::UMMA::Major::K,
                            LOAD_BLOCK_M,
                            DGRAD_BLOCK_K,
                            DGRAD_BLOCK_K *
                                sizeof(cd_dtype_t)>(
                            smem_dgrad_a[0], 0, 0);
                    auto b_desc =
                        mma::sm100::make_umma_desc<
                            cute::UMMA::Major::MN,
                            LOAD_BLOCK_N,
                            DGRAD_BLOCK_K,
                            DGRAD_BLOCK_K *
                                sizeof(
                                    dgrad_b_dtype_t)>(
                            smem_dgrad_b[0], 0, 0);
                    const uint32_t a_desc_lo =
                        lane_idx < kNumStages
                        ? a_desc.lo +
                              lane_idx *
                                  SMEM_A_SIZE_PER_STAGE /
                                  16
                        : 0;
                    const uint32_t b_desc_lo =
                        lane_idx < kNumStages
                        ? b_desc.lo +
                              lane_idx *
                                  SMEM_B_SIZE_PER_STAGE /
                                  16
                        : 0;
                    uint32_t current_iter = 0;

                    for_each_w13_dgrad_block(
                        [&](const uint32_t&,
                            const uint32_t&,
                            const uint32_t&,
                            const uint32_t&,
                            const uint32_t&
                                valid_m) {
                            mma::sm100::
                                update_instr_desc_with_umma_n(
                                    instr_desc,
                                    math::align(
                                        valid_m, 16u));
                            const auto
                                runtime_instr_desc =
                                    cute::UMMA::
                                        make_runtime_instr_desc(
                                            instr_desc);
                            const uint32_t accum_stage =
                                current_iter %
                                kNumEpilogueStages;
                            const uint32_t accum_phase =
                                (current_iter++ /
                                 kNumEpilogueStages) &
                                1;
                            tmem_empty_barriers[
                                accum_stage]
                                ->wait(
                                    accum_phase ^ 1);
                            ptx::tcgen05_after_thread_sync();

                            #pragma unroll 1
                            for (uint32_t
                                     k_block_idx = 0;
                                 k_block_idx <
                                     (2 *
                                      kIntermediateHidden) /
                                         DGRAD_BLOCK_K;
                                 advance_pipeline(
                                     k_block_idx)) {
                                full_barriers[
                                    stage_idx]
                                    ->wait(phase);
                                ptx::tcgen05_after_thread_sync();
                                const uint32_t
                                    a_desc_base =
                                        ptx::exchange(
                                            a_desc_lo,
                                            stage_idx);
                                const uint32_t
                                    b_desc_base =
                                        ptx::exchange(
                                            b_desc_lo,
                                            stage_idx);
                                if (cute::elect_one_sync()) {
                                    #pragma unroll
                                    for (uint32_t k = 0;
                                         k <
                                             DGRAD_BLOCK_K /
                                                 DGRAD_UMMA_K;
                                         ++k) {
                                        a_desc.lo =
                                            mma::sm100::
                                                advance_umma_desc_lo<
                                                    cute::UMMA::Major::K,
                                                    LOAD_BLOCK_M,
                                                    DGRAD_BLOCK_K *
                                                        sizeof(
                                                            cd_dtype_t),
                                                    cd_dtype_t>(
                                                    a_desc_base,
                                                    0,
                                                    k *
                                                        DGRAD_UMMA_K);
                                        b_desc.lo =
                                            mma::sm100::
                                                advance_umma_desc_lo<
                                                    cute::UMMA::Major::MN,
                                                    LOAD_BLOCK_N,
                                                    DGRAD_BLOCK_K *
                                                        sizeof(
                                                            dgrad_b_dtype_t),
                                                    dgrad_b_dtype_t>(
                                                    b_desc_base,
                                                    0,
                                                    k *
                                                        DGRAD_UMMA_K);
                                        ptx::
                                            SM100_MMA_F16BF16_2x1SM_SS::
                                                fma(
                                                    b_desc,
                                                    a_desc,
                                                    accum_stage *
                                                        UMMA_N,
                                                    k_block_idx >
                                                            0 ||
                                                        k > 0,
                                                    runtime_instr_desc);
                                    }
                                }
                                __syncwarp();
                                constexpr uint16_t
                                    kCTAMask = 0x3;
                                cutlass::arch::
                                    umma_arrive_multicast_2x1SM(
                                        reinterpret_cast<
                                            uint64_t*>(
                                            empty_barriers[
                                                stage_idx]),
                                        kCTAMask);
                                if (k_block_idx ==
                                    (2 *
                                     kIntermediateHidden) /
                                            DGRAD_BLOCK_K -
                                        1) {
                                    cutlass::arch::
                                        umma_arrive_multicast_2x1SM(
                                            reinterpret_cast<
                                                uint64_t*>(
                                                tmem_full_barriers[
                                                    accum_stage]),
                                            kCTAMask);
                                }
                                __syncwarp();
                            }
                        });
                    if (current_iter > 0) {
                        const uint32_t last =
                            current_iter - 1;
                        tmem_empty_barriers[
                            last %
                            kNumEpilogueStages]
                            ->wait(
                                (last /
                                 kNumEpilogueStages) &
                                1);
                    }
                }
            } else if (warp_idx >= 4) {
                const uint32_t epilogue_warp_idx =
                    warp_idx - 4;
                const uint32_t epilogue_thread_idx =
                    epilogue_warp_idx * 32 +
                    lane_idx;
                uint32_t current_iter = 0;

                for_each_w13_dgrad_block(
                    [&](const uint32_t&,
                        const uint32_t&
                            pool_block_offset,
                        const uint32_t&
                            m_block_idx,
                        const uint32_t&
                            n_block_idx,
                        const uint32_t& valid_m) {
                        const uint32_t accum_stage =
                            current_iter %
                            kNumEpilogueStages;
                        const uint32_t accum_phase =
                            (current_iter++ /
                             kNumEpilogueStages) &
                            1;
                        tmem_full_barriers[
                            accum_stage]
                            ->wait(accum_phase);
                        ptx::tcgen05_after_thread_sync();
                        const uint32_t effective_m =
                            math::align(valid_m, 16u);

                        for (uint32_t s = 0;
                             s <
                                 effective_m /
                                     STORE_BLOCK_M;
                             ++s) {
                            cutlass::arch::
                                NamedBarrier::sync(
                                    kNumDgradEpilogueThreads,
                                    0);
                            // The four proven loader warps own the 4 KiB
                            // TMEM-to-shared mapping. Extra dgrad epilogue
                            // warps participate only in the global scatter.
                            if (epilogue_warp_idx <
                                kNumEpilogueThreads /
                                    32) {
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i <
                                         STORE_BLOCK_M /
                                             8;
                                     ++i) {
                                    const uint32_t
                                        tmem_addr =
                                            accum_stage *
                                                UMMA_N +
                                            s *
                                                STORE_BLOCK_M +
                                            i * 8;
                                    uint32_t values[8];
                                    cute::
                                        SM100_TMEM_LOAD_16dp256b1x::
                                            copy(
                                                tmem_addr,
                                                values[0],
                                                values[1],
                                                values[2],
                                                values[3]);
                                    cute::
                                        SM100_TMEM_LOAD_16dp256b1x::
                                            copy(
                                                tmem_addr |
                                                    0x00100000,
                                                values[4],
                                                values[5],
                                                values[6],
                                                values[7]);
                                    cutlass::arch::
                                        fence_view_async_tmem_load();

                                    constexpr uint32_t
                                        kBankBytes = 16;
                                    const uint32_t
                                        outer_atom =
                                            (epilogue_warp_idx /
                                             2) *
                                            STORE_BLOCK_M *
                                            128;
                                    const uint32_t
                                        inner_atom =
                                            i * 8 * 128;
                                    const uint32_t row =
                                        lane_idx % 8;
                                    const uint32_t col =
                                        (epilogue_warp_idx %
                                         2) *
                                            4 +
                                        lane_idx / 8;
                                    auto* smem_ptr =
                                        reinterpret_cast<
                                            uint8_t*>(
                                            smem_cd[0]) +
                                        outer_atom +
                                        inner_atom +
                                        row *
                                            (kBankBytes *
                                             8) +
                                        (col ^ row) *
                                            kBankBytes;
                                    ptx::
                                        SM90_U32x4_STSM_T<int>::
                                            copy(
                                                math::
                                                    cast_into_bf16_and_pack(
                                                        values[0],
                                                        values[1]),
                                                math::
                                                    cast_into_bf16_and_pack(
                                                        values[2],
                                                        values[3]),
                                                math::
                                                    cast_into_bf16_and_pack(
                                                        values[4],
                                                        values[5]),
                                                math::
                                                    cast_into_bf16_and_pack(
                                                        values[6],
                                                        values[7]),
                                                smem_ptr);
                                }
                            }
                            cutlass::arch::
                                NamedBarrier::sync(
                                    kNumDgradEpilogueThreads,
                                    0);

                            #pragma unroll
                            for (uint32_t linear =
                                     epilogue_thread_idx;
                                 linear <
                                     STORE_BLOCK_M *
                                         BLOCK_N;
                                 linear +=
                                     kNumDgradEpilogueThreads) {
                                const uint32_t row =
                                    linear / BLOCK_N;
                                const uint32_t n =
                                    linear -
                                    row * BLOCK_N;
                                const uint32_t local_m =
                                    s *
                                        STORE_BLOCK_M +
                                    row;
                                if (local_m >= valid_m)
                                    continue;
                                const uint32_t n_atom =
                                    n / 64;
                                const uint32_t
                                    n_in_atom =
                                        n -
                                        n_atom * 64;
                                const uint32_t
                                    row_in_atom =
                                        row & 7;
                                const uint32_t
                                    smem_byte_offset =
                                        n_atom *
                                            STORE_BLOCK_M *
                                            128 +
                                        (row >> 3) *
                                            8 * 128 +
                                        row_in_atom *
                                            128 +
                                        ((n_in_atom >> 3) ^
                                         row_in_atom) *
                                            16 +
                                        (n_in_atom & 7) *
                                            sizeof(
                                                cd_dtype_t);
                                const uint32_t pool_row =
                                    (pool_block_offset +
                                     m_block_idx) *
                                        BLOCK_M +
                                    local_m;
                                const uint32_t out_col =
                                    n_block_idx *
                                        BLOCK_N +
                                    n;
                                const auto value =
                                    *reinterpret_cast<
                                        cd_dtype_t*>(
                                        reinterpret_cast<
                                            uint8_t*>(
                                            smem_cd[0]) +
                                        smem_byte_offset);
                                if (write_grad_x_pool) {
                                    grad_x_pool_output[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            kHidden +
                                        out_col] = value;
                                }
                                if (direct_remote_grad_x) {
                                    const auto metadata =
                                        token_src_metadata[
                                            pool_row];
                                    auto* combine_buffer =
                                        const_cast<
                                            cd_dtype_t*>(
                                            backward_grad_y);
                                    auto* dst =
                                        combine_buffer +
                                        ((static_cast<
                                              uint64_t>(
                                              metadata
                                                  .topk_idx) *
                                              backward_workspace
                                                  .num_max_tokens_per_rank +
                                          metadata.token_idx) *
                                             kHidden +
                                         out_col);
                                    *backward_sym_buffer.map(
                                        dst,
                                        metadata.rank_idx) =
                                        value;
                                }
                            }
                        }
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[
                            accum_stage]
                            ->arrive(0u);
                    });
            }
        }

        const auto clear_wgrad_padding_rows = [&]() {
            uint32_t pad_pool_block_offset = 0;
            uint32_t pad_global_block = 0;
            #pragma unroll
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens =
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                const uint32_t num_blocks =
                    math::ceil_div(num_tokens, BLOCK_M);
                if (num_blocks != 0) {
                    const uint32_t last_valid =
                        num_tokens - (num_blocks - 1) * BLOCK_M;
                    const uint32_t pool_block =
                        pad_pool_block_offset + num_blocks - 1;
                    if (pad_global_block % kNumSMs ==
                        blockIdx.x) {
                        for (uint32_t linear = threadIdx.x;
                             linear <
                                 (BLOCK_M - last_valid) *
                                     (kHidden +
                                      3 * kIntermediateHidden);
                             linear += kNumThreads) {
                            const uint32_t row_delta =
                                linear /
                                (kHidden +
                                 3 * kIntermediateHidden);
                            const uint32_t col =
                                linear -
                                row_delta *
                                    (kHidden +
                                     3 *
                                         kIntermediateHidden);
                            const uint32_t pool_row =
                                pool_block * BLOCK_M +
                                last_valid + row_delta;
                            if (col < kHidden) {
                                grad_ye_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] = cd_dtype_t(0.0f);
                            } else if (
                                col <
                                kHidden +
                                    kIntermediateHidden) {
                                h_weighted_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kIntermediateHidden +
                                    col - kHidden] =
                                    cd_dtype_t(0.0f);
                            } else {
                                grad_gate_up_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        (2 *
                                         kIntermediateHidden) +
                                    col - kHidden -
                                        kIntermediateHidden] =
                                    cd_dtype_t(0.0f);
                            }
                        }
                    }
                    ++pad_global_block;
                }
                pad_pool_block_offset += num_blocks;
            }
        };

        // Standalone Kernel B consumes only these three padded operands. Valid
        // rows are fully overwritten above; clear only the final partial block
        // of each expert instead of memset'ing every active scratch prefix.
        if (clear_wgrad_padding)
            clear_wgrad_padding_rows();



        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();
        if (warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
    }
#endif
}

}  // namespace deep_gemm
