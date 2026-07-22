#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sm103_block128_wgrad {

// The two CTAs form the same logical 256x256x64 production BF16 tile used by
// DeepGEMM's grouped large-M kernels. Each dedicated wgrad kernel first converts
// its FP8 route operands exactly once into private BF16 backing inside this same
// persistent launch, grid-fences internally, and then runs the native BF16 TMA
// / UMMA pipeline. Dequantization is therefore part of the dedicated kernel's
// load prologue rather than repeated for every output tile or composed as a
// separate kernel.
static constexpr uint32_t kHidden = 6144;
static constexpr uint32_t kIntermediate = 2048;
static constexpr uint32_t kGlobalExperts = 256;
static constexpr uint32_t kRouteBlockM = 192;
static constexpr uint32_t kBlockM = 128;
static constexpr uint32_t kBlockN = 256;
static constexpr uint32_t kBlockK = 64;
static constexpr uint32_t kLoadBlockN = kBlockN / 2;
static constexpr uint32_t kStages = 3;
static constexpr uint32_t kTMAWarp = 0;
static constexpr uint32_t kMMAWarp = 1;
static constexpr uint32_t kReadyWarp = 2;
static constexpr uint32_t kControlWarp = 3;
static constexpr uint32_t kEpilogueFirstWarp = 4;
static constexpr uint32_t kEpilogueThreads = 128;
static constexpr uint32_t kThreads =
    kEpilogueFirstWarp * 32 + kEpilogueThreads;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumTMAStoreStages = 2;
static constexpr uint32_t kStoreBlockM = 128;
static constexpr uint32_t kStoreBlockN = 64;
static constexpr uint32_t kUMMAM = 256;
static constexpr uint32_t kUMMAN = 256;
static constexpr uint32_t kUMMAK = 16;
static constexpr uint32_t kSwizzle = 128;
static constexpr uint32_t kNumTmemAccumCols =
    kNumEpilogueStages * kUMMAN;
static constexpr uint32_t kNumTmemCols =
    utils::get_num_aligned_tmem_cols<kNumTmemAccumCols>();

using fp8_t = cutlass::float_e4m3_t;
using bf16_t = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

struct alignas(1024) SharedStorage {
    alignas(1024) bf16_t smem_cd[kNumTMAStoreStages]
                                      [kStoreBlockM * kStoreBlockN];
    alignas(1024) bf16_t smem_a[kStages][kBlockK * kBlockM];
    alignas(1024) bf16_t smem_b[kStages][kBlockK * kLoadBlockN];
    Barrier tma_full_barriers[kStages];
    Barrier tma_empty_barriers[kStages];
    Barrier mma_full_barriers[kStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

DG_STATIC_ASSERT(kThreads == 256, "SM103 wgrad role layout changed");
DG_STATIC_ASSERT(kLoadBlockN == kBlockM,
                 "wgrad A/B prologues must share one tile shape");
DG_STATIC_ASSERT(kNumTmemCols == 512, "SM103 wgrad TMEM layout changed");

CUTLASS_DEVICE uint16_t fold_power2_scale_into_bf16(
    const uint16_t half_bits,
    const uint32_t scale_exponent) {
    const uint16_t sign = half_bits & 0x8000u;
    const uint32_t half_exponent = (half_bits >> 10) & 0x1fu;
    if (half_exponent == 0u)
        return sign;
    if (half_exponent == 0x1fu)
        return sign | 0x7fc0u;

    const uint32_t mantissa = (half_bits >> 3) & 0x7fu;
    const int32_t bf16_exponent =
        static_cast<int32_t>(half_exponent) +
        static_cast<int32_t>(scale_exponent) - 15;
    if (bf16_exponent >= 0xff)
        return sign | 0x7f80u;
    if (bf16_exponent > 0)
        return sign |
               static_cast<uint16_t>(bf16_exponent << 7) |
               static_cast<uint16_t>(mantissa);

    // The E4M3 significand has only four bits, so folding a power-of-two scale
    // into BF16 is exact except when the result reaches BF16's subnormal range.
    // Reproduce round-to-nearest-even there without materializing FP32.
    const uint32_t significand = 0x80u | mantissa;
    const uint32_t shift = static_cast<uint32_t>(1 - bf16_exponent);
    if (shift > 8u)
        return sign;
    const uint32_t truncated = significand >> shift;
    const uint32_t remainder =
        significand & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    const uint32_t rounded = truncated +
        (remainder > halfway ||
         (remainder == halfway && (truncated & 1u)));
    return sign | static_cast<uint16_t>(rounded);
}

CUTLASS_DEVICE uint32_t convert_fp8x2_power2_to_bf16x2(
    const uint16_t fp8x2,
    const uint32_t scale_exponent) {
    uint32_t half2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
        : "=r"(half2) : "h"(fp8x2));
    const uint32_t lo = fold_power2_scale_into_bf16(
        static_cast<uint16_t>(half2), scale_exponent);
    const uint32_t hi = fold_power2_scale_into_bf16(
        static_cast<uint16_t>(half2 >> 16), scale_exponent);
    return lo | (hi << 16);
}

template <
    uint32_t kShape, uint32_t kLocalExperts,
    uint32_t kNumSMs, uint32_t kNumThreads,
    bool kApplyPostScale>
CUTLASS_DEVICE void dequantize_route_pool_once(
    const int* expert_counts,
    const uint32_t max_pool_tokens,
    const fp8_t* source,
    const uint32_t* scales,
    const float* scores,
    bf16_t* destination) {
    constexpr uint32_t kValuesPerVector = 8;
    constexpr uint32_t kVectorsPerRow = kShape / kValuesPerVector;
    constexpr uint32_t kScaleBlocksPerRow = kShape / 128;
    DG_STATIC_ASSERT(kShape % 128 == 0,
                     "wgrad dequant shape must be block128 aligned");

    const uint64_t global_thread =
        static_cast<uint64_t>(blockIdx.x) * kNumThreads + threadIdx.x;
    constexpr uint64_t kGridThreads =
        static_cast<uint64_t>(kNumSMs) * kNumThreads;
    uint32_t pool_row = 0;

    #pragma unroll 1
    for (uint32_t expert = 0; expert < kLocalExperts; ++ expert) {
        const uint32_t count = static_cast<uint32_t>(
            __ldg(expert_counts + expert));
        const uint32_t padded_count =
            math::ceil_div(count, kRouteBlockM) * kRouteBlockM;
        const uint64_t num_vectors =
            static_cast<uint64_t>(padded_count) * kVectorsPerRow;
        for (uint64_t linear = global_thread;
             linear < num_vectors; linear += kGridThreads) {
            const uint32_t route = static_cast<uint32_t>(
                linear / kVectorsPerRow);
            const uint32_t vector_in_row = static_cast<uint32_t>(
                linear - static_cast<uint64_t>(route) * kVectorsPerRow);
            const uint32_t feature =
                vector_in_row * kValuesPerVector;
            const uint64_t full_row =
                static_cast<uint64_t>(pool_row) + route;
            uint4 packed{};
            if (route < count) {
                const uint2 raw = *reinterpret_cast<const uint2*>(
                    source + full_row * kShape + feature);
                const uint32_t raw_words[2] = {raw.x, raw.y};
                const uint32_t scale_exponent = scales[
                    full_row * kScaleBlocksPerRow + feature / 128] & 0xffu;
                const float post_scale = kApplyPostScale
                    ? scores[full_row]
                    : 1.0f;
                auto* output_pairs = reinterpret_cast<uint32_t*>(&packed);
                #pragma unroll
                for (uint32_t pair = 0; pair < 4; ++ pair) {
                    const uint16_t fp8x2 = static_cast<uint16_t>(
                        raw_words[pair / 2] >> ((pair & 1u) * 16));
                    uint32_t bf16x2 = convert_fp8x2_power2_to_bf16x2(
                        fp8x2, scale_exponent);
                    if constexpr (kApplyPostScale) {
                        const auto dequantized = __bfloat1622float2(
                            *reinterpret_cast<nv_bfloat162*>(&bf16x2));
                        const auto scaled = __float22bfloat162_rn(
                            __fmul2_rn(
                                dequantized,
                                {post_scale, post_scale}));
                        bf16x2 =
                            *reinterpret_cast<const uint32_t*>(&scaled);
                    }
                    output_pairs[pair] = bf16x2;
                }
            }
            *reinterpret_cast<uint4*>(
                destination + full_row * kShape + feature) = packed;
        }
        pool_row += padded_count;
    }
    DG_DEVICE_ASSERT(pool_row <= max_pool_tokens);

    // Empty experts consume one permanent all-zero K tile beyond the routed
    // pool. The host-side private scratch descriptors include these rows.
    constexpr uint64_t kZeroVectors =
        static_cast<uint64_t>(kBlockK) * kVectorsPerRow;
    for (uint64_t linear = global_thread;
         linear < kZeroVectors; linear += kGridThreads) {
        *reinterpret_cast<uint4*>(
            destination +
            static_cast<uint64_t>(max_pool_tokens) * kShape +
            linear * kValuesPerVector) = {};
    }
}

// This is the production grouped-GEMM L2 swizzle specialized to the fixed GLM
// shapes.  Eight adjacent M blocks sweep all N blocks before moving to the next
// M group.  The group size and every expert's tile count are even, so adjacent
// physical CTAs always remain the two M halves of one 2-CTA output tile.
template <bool kW2, uint32_t kNumRanks, uint32_t kNumSMs>
struct WgradTileScheduler {
    static constexpr uint32_t kLocalExperts =
        kGlobalExperts / kNumRanks;
    static constexpr uint32_t kShapeM =
        kW2 ? kHidden : 2 * kIntermediate;
    static constexpr uint32_t kShapeN =
        kW2 ? kIntermediate : kHidden;
    static constexpr uint32_t kNumMBlocks = kShapeM / kBlockM;
    static constexpr uint32_t kNumNBlocks = kShapeN / kBlockN;
    static constexpr uint32_t kMBlocksPerL2Group = 8;
    static constexpr uint32_t kTilesPerL2Group =
        kMBlocksPerL2Group * kNumNBlocks;
    static constexpr uint32_t kTilesPerExpert =
        kNumMBlocks * kNumNBlocks;
    static constexpr uint32_t kTotalTiles =
        kLocalExperts * kTilesPerExpert;

    const int* expert_counts;
    uint32_t linear_tile = blockIdx.x;
    uint32_t cached_expert = 0;
    uint32_t cached_pool_row = 0;

    CUTLASS_DEVICE explicit WgradTileScheduler(const int* counts)
        : expert_counts(counts) {
        DG_STATIC_ASSERT(kNumMBlocks % kMBlocksPerL2Group == 0,
                         "fixed GLM M shape must fit the L2 swizzle");
        DG_STATIC_ASSERT(kTilesPerL2Group % 2 == 0,
                         "L2 groups must preserve 2-CTA pairing");
        DG_STATIC_ASSERT(kNumSMs % 2 == 0,
                         "2-CTA wgrad requires an even SM count");
    }

    CUTLASS_DEVICE bool get_next(
        uint32_t& expert,
        uint32_t& count,
        uint32_t& pool_row,
        uint32_t& m_block,
        uint32_t& n_block) {
        if (linear_tile >= kTotalTiles)
            return false;
        expert = linear_tile / kTilesPerExpert;
        const uint32_t expert_tile =
            linear_tile - expert * kTilesPerExpert;
        while (cached_expert < expert) {
            const uint32_t previous_count = static_cast<uint32_t>(
                __ldg(expert_counts + cached_expert));
            cached_pool_row +=
                math::ceil_div(previous_count, kRouteBlockM) *
                kRouteBlockM;
            ++cached_expert;
        }
        count = static_cast<uint32_t>(__ldg(expert_counts + expert));
        pool_row = cached_pool_row;
        const uint32_t l2_group = expert_tile / kTilesPerL2Group;
        const uint32_t tile_in_group =
            expert_tile - l2_group * kTilesPerL2Group;
        m_block = l2_group * kMBlocksPerL2Group +
                  tile_in_group % kMBlocksPerL2Group;
        n_block = tile_in_group / kMBlocksPerL2Group;
        linear_tile += kNumSMs;
        return true;
    }
};

template <bool kW2, uint32_t kNumRanks, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kThreads, 1) void
sm103_fp8_block128_mega_moe_wgrad_impl(
    const int* expert_counts,
    const uint32_t max_pool_tokens,
    const __grid_constant__ layout::Workspace workspace,
    const fp8_t* full_a,
    const fp8_t* full_b,
    const uint32_t* full_a_sf,
    const uint32_t* full_b_sf,
    const float* full_scores,
    bf16_t* cached_a,
    bf16_t* cached_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_0,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    using Scheduler = WgradTileScheduler<kW2, kNumRanks, kNumSMs>;
    constexpr uint32_t kShapeM = Scheduler::kShapeM;
    constexpr uint32_t kShapeN = Scheduler::kShapeN;
    constexpr uint32_t kLocalExperts = Scheduler::kLocalExperts;

    // One fused prologue per dedicated wgrad launch. This removes conversion
    // from the output-tile loop while preserving the exact E4M3 + FP32
    // power-of-two scale and BF16-rounding semantics.
    dequantize_route_pool_once<
        kShapeM, kLocalExperts, kNumSMs, kThreads, kW2>(
            expert_counts, max_pool_tokens,
            full_a, full_a_sf, full_scores, cached_a);
    dequantize_route_pool_once<
        kShapeN, kLocalExperts, kNumSMs, kThreads, false>(
            expert_counts, max_pool_tokens,
            full_b, full_b_sf, full_scores, cached_b);
    comm::grid_sync<kNumSMs, kW2 ? 2u : 3u>(
        workspace, blockIdx.x, threadIdx.x,
        []() { __syncthreads(); });

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const bool leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t cta_rank = cute::block_rank_in_cluster();

    if (warp_idx == kTMAWarp) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_output_0);
        cute::prefetch_tma_descriptor(&tensor_map_output_1);
    }

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == kControlWarp && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kStages; ++i) {
            storage.tma_full_barriers[i].init(1);
            storage.tma_empty_barriers[i].init(1);
            // Both CTAs publish their direct-BF16 TMA completion to CTA 0.
            storage.mma_full_barriers[i].init(2);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            storage.tmem_full_barriers[i].init(1);
            storage.tmem_empty_barriers[i].init(
                2 * kEpilogueThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();
    if (warp_idx == kControlWarp)
        cute::TMEM::Allocator2Sm().allocate(
            kNumTmemCols, &storage.tmem_ptr);
    comm::cluster_sync_with_relaxed_arrive();

    auto instr_desc = cute::UMMA::make_instr_desc<
        bf16_t, bf16_t, float,
        kUMMAM, kUMMAN,
        cute::UMMA::Major::MN,
        cute::UMMA::Major::MN>();
    auto a_desc = mma::sm100::make_umma_desc<
        cute::UMMA::Major::MN, kBlockM, kBlockK, kSwizzle>(
            storage.smem_a[0], 0, 0);
    auto b_desc = mma::sm100::make_umma_desc<
        cute::UMMA::Major::MN, kLoadBlockN, kBlockK, kSwizzle>(
            storage.smem_b[0], 0, 0);
    const uint32_t a_desc_lo = lane_idx < kStages
        ? a_desc.lo + lane_idx * sizeof(storage.smem_a[0]) / 16
        : 0u;
    const uint32_t b_desc_lo = lane_idx < kStages
        ? b_desc.lo + lane_idx * sizeof(storage.smem_b[0]) / 16
        : 0u;
    const auto runtime_instr_desc =
        cute::UMMA::make_runtime_instr_desc(instr_desc);
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return storage.smem_cd[i];
    });

    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == kTMAWarp && cute::elect_one_sync()) {
        // The production load warp now reads the once-dequantized BF16 backing
        // directly into the native UMMA swizzle.
        Scheduler scheduler(expert_counts);
        uint32_t expert, count, pool_row, m_block, n_block;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            DG_DEVICE_ASSERT(pool_row + count <= max_pool_tokens);
            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.tma_empty_barriers[stage_idx].wait(phase ^ 1u);
                const uint32_t route = count == 0
                    ? max_pool_tokens
                    : pool_row + k_block * kBlockK;
                const uint32_t a_feature = m_block * kBlockM;
                const uint32_t b_feature =
                    n_block * kBlockN + cta_rank * kLoadBlockN;
                tma::copy<kBlockM, kBlockK, kSwizzle, bf16_t>(
                    &tensor_map_a,
                    &storage.tma_full_barriers[stage_idx],
                    storage.smem_a[stage_idx],
                    a_feature, route);
                tma::copy<kLoadBlockN, kBlockK, kSwizzle, bf16_t>(
                    &tensor_map_b,
                    &storage.tma_full_barriers[stage_idx],
                    storage.smem_b[stage_idx],
                    b_feature, route);
                storage.tma_full_barriers[stage_idx]
                    .arrive_and_expect_tx(
                        sizeof(storage.smem_a[0]) +
                        sizeof(storage.smem_b[0]));
                advance_pipeline();
            }
        }
    } else if (warp_idx == kReadyWarp && cute::elect_one_sync()) {
        // Each CTA waits for its local direct-BF16 TMAs, then contributes one
        // arrival to CTA 0. The leader MMA warp therefore observes both halves
        // without multicast-copying different GLM feature tiles over each
        // other.
        Scheduler scheduler(expert_counts);
        uint32_t expert, count, pool_row, m_block, n_block;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.tma_full_barriers[stage_idx].wait(phase);
                cutlass::arch::fence_view_async_shared();
                storage.mma_full_barriers[stage_idx].arrive(0u);
                advance_pipeline();
            }
        }
    } else if (warp_idx == kMMAWarp && leader_cta) {
        Scheduler scheduler(expert_counts);
        uint32_t expert, count, pool_row, m_block, n_block;
        uint32_t output_iter = 0;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            const uint32_t accum_stage =
                output_iter % kNumEpilogueStages;
            const uint32_t accum_phase =
                (output_iter / kNumEpilogueStages) & 1u;
            ++output_iter;
            storage.tmem_empty_barriers[accum_stage].wait(
                accum_phase ^ 1u);
            ptx::tcgen05_after_thread_sync();

            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.mma_full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();
                const uint32_t a_base =
                    ptx::exchange(a_desc_lo, stage_idx);
                const uint32_t b_base =
                    ptx::exchange(b_desc_lo, stage_idx);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < kBlockK / kUMMAK; ++k) {
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<
                            cute::UMMA::Major::MN, kBlockM,
                            kSwizzle, bf16_t>(
                                a_base, 0, k * kUMMAK);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<
                            cute::UMMA::Major::MN, kLoadBlockN,
                            kSwizzle, bf16_t>(
                                b_base, 0, k * kUMMAK);
                        ptx::SM100_MMA_F16BF16_2x1SM_SS::fma(
                            a_desc, b_desc,
                            accum_stage * kUMMAN,
                            k_block > 0 || k > 0,
                            runtime_instr_desc);
                    }
                }
                __syncwarp();
                constexpr uint16_t kCTAMask = 3;
                cutlass::arch::umma_arrive_multicast_2x1SM(
                    reinterpret_cast<uint64_t*>(
                        &storage.tma_empty_barriers[stage_idx]),
                    kCTAMask);
                if (k_block == num_k_blocks - 1) {
                    cutlass::arch::umma_arrive_multicast_2x1SM(
                        reinterpret_cast<uint64_t*>(
                            &storage.tmem_full_barriers[accum_stage]),
                        kCTAMask);
                }
                __syncwarp();
                advance_pipeline();
            }
        }
    } else if (
        warp_idx >= kEpilogueFirstWarp &&
        warp_idx < kEpilogueFirstWarp + kEpilogueThreads / 32) {
        Scheduler scheduler(expert_counts);
        const uint32_t epilogue_warp_idx =
            warp_idx - kEpilogueFirstWarp;
        uint32_t expert, count, pool_row, m_block, n_block;
        uint32_t output_iter = 0;
        uint32_t tma_stage_idx = 0;
        DG_TRAP_ONLY_DEVICE_ASSERT(
            ptx::ld_shared(&storage.tmem_ptr) == 0);
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            const uint32_t accum_stage =
                output_iter % kNumEpilogueStages;
            const uint32_t accum_phase =
                (output_iter / kNumEpilogueStages) & 1u;
            ++output_iter;
            storage.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();

            const cute::TmaDescriptor* output_map =
                &tensor_map_output_0;
            uint32_t output_m =
                expert * (kW2 ? kHidden : kIntermediate);
            if constexpr (kW2) {
                output_m += m_block * kBlockM;
            } else {
                const uint32_t plane =
                    m_block / (kIntermediate / kBlockM);
                const uint32_t plane_m_block =
                    m_block % (kIntermediate / kBlockM);
                output_map = plane == 0
                    ? &tensor_map_output_0
                    : &tensor_map_output_1;
                output_m += plane_m_block * kBlockM;
            }
            epilogue::sm100_store_cd<
                kBlockM, kBlockN,
                kStoreBlockM, kStoreBlockN,
                kSwizzle, kNumTMAStoreStages, kEpilogueThreads,
                GemmType::Normal, false, bf16_t,
                epilogue::transform::EpilogueIdentity>(
                    smem_cd, tma_stage_idx,
                    accum_stage * kUMMAN,
                    output_m, n_block * kBlockN, 0,
                    epilogue_warp_idx, lane_idx,
                    &storage.tmem_empty_barriers[accum_stage],
                    *output_map);
        }
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
        __syncwarp();
    }

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == kControlWarp)
        cute::TMEM::Allocator2Sm().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "SM103 MegaMoE wgrad has no fallback");
#endif
}

}  // namespace deep_gemm::sm103_block128_wgrad

#pragma clang diagnostic pop
