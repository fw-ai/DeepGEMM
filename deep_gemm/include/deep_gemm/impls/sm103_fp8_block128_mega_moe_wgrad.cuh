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
// DeepGEMM's grouped large-M kernels.  Each CTA owns 128 M rows and 128 N
// columns.  FP8 route pools are TMA-staged, then converted in the load prologue
// into the exact MN-major BF16 shared-memory layout consumed by native UMMA.
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
static constexpr uint32_t kConvertFirstWarp = 2;
static constexpr uint32_t kConvertThreads = 128;
static constexpr uint32_t kControlWarp = 6;
static constexpr uint32_t kEpilogueFirstWarp = 7;
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
    alignas(1024) fp8_t raw_a[kStages][kBlockK * kBlockM];
    alignas(1024) fp8_t raw_b[kStages][kBlockK * kLoadBlockN];
    alignas(1024) bf16_t smem_a[kStages][kBlockK * kBlockM];
    alignas(1024) bf16_t smem_b[kStages][kBlockK * kLoadBlockN];
    Barrier tma_full_barriers[kStages];
    Barrier tma_empty_barriers[kStages];
    Barrier mma_full_barriers[kStages];
    Barrier mma_empty_barriers[kStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

DG_STATIC_ASSERT(kThreads == 352, "SM103 wgrad role layout changed");
DG_STATIC_ASSERT(kLoadBlockN == kBlockM,
                 "wgrad A/B prologues must share one tile shape");
DG_STATIC_ASSERT(kNumTmemCols == 512, "SM103 wgrad TMEM layout changed");

CUTLASS_DEVICE float unpack_power2_scale(const uint32_t packed) {
    const uint32_t exponent = packed & 0xffu;
    // UE8M0 code zero denotes 2^-127.  Construct it explicitly because
    // shifting zero into an IEEE exponent field would produce zero.
    return exponent == 0u ? 0x1p-127f : __uint_as_float(exponent << 23);
}

CUTLASS_DEVICE float round_score_to_bf16(const float score) {
    return static_cast<float>(bf16_t(score));
}

// Address one 16-byte bank group in the TMA swizzle-128 layout for an
// MN-major [inner-MN, outer-K] tile.  TMA splits an inner dimension wider than
// 64 BF16 values into consecutive 64-value atoms.
template <uint32_t kInnerMN, uint32_t kOuterK>
CUTLASS_DEVICE uint8_t* get_bf16_mn_bank_group(
    bf16_t* base,
    const uint32_t inner_mn,
    const uint32_t outer_k) {
    constexpr uint32_t kBankGroupBytes = 16;
    constexpr uint32_t kInnerPerAtom = kSwizzle / sizeof(bf16_t);
    DG_STATIC_ASSERT(kInnerMN % kInnerPerAtom == 0,
                     "MN dimension must contain whole swizzle atoms");
    DG_STATIC_ASSERT(kOuterK % 8 == 0,
                     "K dimension must contain whole swizzle rows");
    const uint32_t atom = inner_mn / kInnerPerAtom;
    const uint32_t inner_in_atom = inner_mn % kInnerPerAtom;
    const uint32_t row = outer_k & 7u;
    const uint32_t inner_byte = inner_in_atom * sizeof(bf16_t);
    const uint32_t byte_offset =
        atom * kOuterK * kSwizzle +
        (outer_k >> 3) * 8u * kSwizzle +
        row * kSwizzle +
        ((inner_byte >> 4) ^ row) * kBankGroupBytes +
        (inner_byte & (kBankGroupBytes - 1));
    return reinterpret_cast<uint8_t*>(base) + byte_offset;
}

template <uint32_t kInnerMN, uint32_t kOuterK>
CUTLASS_DEVICE void convert_and_store_eight(
    const fp8_t* source,
    bf16_t* destination,
    const uint32_t inner_mn,
    const uint32_t outer_k,
    const float dequant_scale,
    const float post_scale,
    const bool valid) {
    uint4 packed{};
    auto* values = reinterpret_cast<bf16_t*>(&packed);
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        if (valid) {
            // "BF16-semantics" means the FP8+power-of-two value first becomes
            // the BF16 operand represented by the private pool.  W2 then
            // applies the BF16-rounded route score and rounds to BF16 again.
            const float dequantized = static_cast<float>(
                bf16_t(static_cast<float>(source[i]) * dequant_scale));
            values[i] = bf16_t(dequantized * post_scale);
        } else {
            values[i] = bf16_t(0.0f);
        }
    }
    *reinterpret_cast<uint4*>(
        get_bf16_mn_bank_group<kInnerMN, kOuterK>(
            destination, inner_mn, outer_k)) = packed;
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
    const uint32_t* full_a_sf,
    const uint32_t* full_b_sf,
    const float* full_scores,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_0,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    using Scheduler = WgradTileScheduler<kW2, kNumRanks, kNumSMs>;
    constexpr uint32_t kShapeM = Scheduler::kShapeM;
    constexpr uint32_t kShapeN = Scheduler::kShapeN;
    constexpr uint32_t kFullABlocks = kShapeM / 128;
    constexpr uint32_t kFullBBlocks = kShapeN / 128;
    constexpr uint32_t kRawABytes = kBlockK * kBlockM * sizeof(fp8_t);
    constexpr uint32_t kRawBBytes =
        kBlockK * kLoadBlockN * sizeof(fp8_t);

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
            // Both CTAs publish their converted halves to CTA 0.
            storage.mma_full_barriers[i].init(2);
            storage.mma_empty_barriers[i].init(1);
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
        // The production load warp issues two rectangular TMA transactions per
        // K stage.  Raw tiles are row-major [route-K, feature-MN].
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
                const uint32_t route = pool_row + k_block * kBlockK;
                const uint32_t a_feature = m_block * kBlockM;
                const uint32_t b_feature =
                    n_block * kBlockN + cta_rank * kLoadBlockN;
                tma::copy<kBlockM, kBlockK, 0, fp8_t>(
                    &tensor_map_a,
                    &storage.tma_full_barriers[stage_idx],
                    storage.raw_a[stage_idx],
                    a_feature, route);
                tma::copy<kLoadBlockN, kBlockK, 0, fp8_t>(
                    &tensor_map_b,
                    &storage.tma_full_barriers[stage_idx],
                    storage.raw_b[stage_idx],
                    b_feature, route);
                storage.tma_full_barriers[stage_idx]
                    .arrive_and_expect_tx(kRawABytes + kRawBBytes);
                advance_pipeline();
            }
        }
    } else if (
        warp_idx >= kConvertFirstWarp &&
        warp_idx < kConvertFirstWarp + kConvertThreads / 32) {
        // Two converter threads own each route.  Each thread loads four aligned
        // FP8x16 vectors per operand, reuses one row scale, and emits eight
        // aligned BF16x8 bank groups directly into the UMMA swizzle.
        Scheduler scheduler(expert_counts);
        const uint32_t convert_thread =
            threadIdx.x - kConvertFirstWarp * 32;
        const uint32_t route_in_k = convert_thread / 2;
        const uint32_t half = convert_thread & 1u;
        uint32_t expert, count, pool_row, m_block, n_block;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            DG_DEVICE_ASSERT(pool_row + count <= max_pool_tokens);
            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.tma_full_barriers[stage_idx].wait(phase);
                storage.mma_empty_barriers[stage_idx].wait(phase ^ 1u);
                const uint32_t route = k_block * kBlockK + route_in_k;
                const bool valid = route < count;
                const uint64_t full_row = pool_row + route;
                float a_scale = 0.0f;
                float b_scale = 0.0f;
                float a_post_scale = 1.0f;
                if (valid) {
                    a_scale = unpack_power2_scale(full_a_sf[
                        full_row * kFullABlocks + m_block]);
                    b_scale = unpack_power2_scale(full_b_sf[
                        full_row * kFullBBlocks +
                        n_block * (kBlockN / 128) + cta_rank]);
                    if constexpr (kW2)
                        a_post_scale = round_score_to_bf16(
                            full_scores[full_row]);
                }
                #pragma unroll
                for (uint32_t chunk = 0; chunk < 4; ++chunk) {
                    const uint32_t inner = (half * 4 + chunk) * 16;
                    const auto* raw_a = storage.raw_a[stage_idx] +
                        route_in_k * kBlockM + inner;
                    const auto* raw_b = storage.raw_b[stage_idx] +
                        route_in_k * kLoadBlockN + inner;
                    convert_and_store_eight<kBlockM, kBlockK>(
                        raw_a, storage.smem_a[stage_idx],
                        inner, route_in_k,
                        a_scale, a_post_scale, valid);
                    convert_and_store_eight<kBlockM, kBlockK>(
                        raw_a + 8, storage.smem_a[stage_idx],
                        inner + 8, route_in_k,
                        a_scale, a_post_scale, valid);
                    convert_and_store_eight<kLoadBlockN, kBlockK>(
                        raw_b, storage.smem_b[stage_idx],
                        inner, route_in_k,
                        b_scale, 1.0f, valid);
                    convert_and_store_eight<kLoadBlockN, kBlockK>(
                        raw_b + 8, storage.smem_b[stage_idx],
                        inner + 8, route_in_k,
                        b_scale, 1.0f, valid);
                }
                ptx::sync_aligned(128, 1);
                cutlass::arch::fence_view_async_shared();
                if (convert_thread == 0) {
                    storage.tma_empty_barriers[stage_idx].arrive();
                    storage.mma_full_barriers[stage_idx].arrive(0u);
                }
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
                        &storage.mma_empty_barriers[stage_idx]),
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
