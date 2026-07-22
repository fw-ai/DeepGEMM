#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sm103_block128_wgrad {

// GLM-5.2 has enough routes per expert that one fixed 2-CTA family covers both
// target topologies.  K is the route dimension.  FP8 operands stay resident in
// the private expert-padded pool; the producer groups dequantize directly into
// pipelined BF16 shared-memory tiles consumed by native BF16 UMMA.
static constexpr uint32_t kHidden = 6144;
static constexpr uint32_t kIntermediate = 2048;
static constexpr uint32_t kGlobalExperts = 256;
static constexpr uint32_t kRouteBlockM = 192;
static constexpr uint32_t kBlockM = 128;
static constexpr uint32_t kBlockN = 128;
static constexpr uint32_t kBlockK = 64;
static constexpr uint32_t kLoadBlockN = kBlockN / 2;
static constexpr uint32_t kStages = 6;
static constexpr uint32_t kAProducerThreads = 128;
static constexpr uint32_t kBProducerThreads = 64;
static constexpr uint32_t kMMAWarp = 6;
static constexpr uint32_t kControlWarp = 7;
static constexpr uint32_t kEpilogueFirstWarp = 8;
static constexpr uint32_t kEpilogueThreads = 128;
static constexpr uint32_t kThreads =
    kEpilogueFirstWarp * 32 + kEpilogueThreads;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumTMAStoreStages = 2;
static constexpr uint32_t kStoreBlockM = 128;
static constexpr uint32_t kStoreBlockN = 64;
static constexpr uint32_t kUMMAM = 256;
static constexpr uint32_t kUMMAN = 128;
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
    alignas(1024) bf16_t smem_a[kStages][kBlockM * kBlockK];
    alignas(1024) bf16_t smem_b[kStages][kLoadBlockN * kBlockK];
    Barrier full_barriers[kStages];
    Barrier empty_barriers[kStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

DG_STATIC_ASSERT(kThreads == 384, "SM103 wgrad role layout changed");
DG_STATIC_ASSERT(kNumTmemCols <= 512, "SM103 wgrad exceeds TMEM");

CUTLASS_DEVICE float unpack_power2_scale(const uint32_t packed) {
    const uint32_t exponent = packed & 0xffu;
    // UE8M0 code zero denotes 2^-127.  Construct it explicitly because
    // shifting zero into an IEEE exponent field would produce zero.
    return exponent == 0u ? 0x1p-127f : __uint_as_float(exponent << 23);
}

template <uint32_t kRows>
CUTLASS_DEVICE void store_mn_swizzle128(
    bf16_t* base,
    const uint32_t row,
    const uint32_t k,
    const bf16_t value) {
    DG_STATIC_ASSERT(kRows == 64 || kRows == 128,
                     "invalid BF16 SMEM rows");
    const uint32_t row_in_atom = row & 7u;
    const uint32_t col_byte = k * sizeof(bf16_t);
    const uint32_t byte_offset =
        (row >> 3) * 8u * kSwizzle + row_in_atom * kSwizzle +
        ((col_byte >> 4) ^ row_in_atom) * 16u + (col_byte & 15u);
    *reinterpret_cast<bf16_t*>(
        reinterpret_cast<uint8_t*>(base) + byte_offset) = value;
}

// Each physical CTA starts at its own output tile and advances by the fixed SM
// count.  Because every expert matrix has an even number of M tiles, adjacent
// CTAs in a 2-CTA cluster always address adjacent M halves of the same expert/N
// tile.  Pool offsets advance only when the monotonic tile stream crosses an
// expert boundary; there is no global queue, host metadata, or per-expert sync.
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
    static constexpr uint32_t kTilesPerExpert =
        kNumMBlocks * kNumNBlocks;
    static constexpr uint32_t kTotalTiles =
        kLocalExperts * kTilesPerExpert;

    const int* expert_counts;
    uint32_t linear_tile = blockIdx.x;
    uint32_t cached_expert = 0;
    uint32_t cached_pool_row = 0;

    CUTLASS_DEVICE explicit WgradTileScheduler(const int* counts)
        : expert_counts(counts) {}

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
        m_block = expert_tile % kNumMBlocks;
        n_block = expert_tile / kNumMBlocks;
        linear_tile += kNumSMs;
        return true;
    }
};

template <bool kW2, uint32_t kNumRanks, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kThreads, 1) void
sm103_fp8_block128_mega_moe_wgrad_impl(
    const int* expert_counts,
    const uint32_t max_pool_tokens,
    const fp8_t* full_x,
    const uint32_t* full_x_sf,
    const fp8_t* full_grad_y,
    const uint32_t* full_grad_y_sf,
    const float* full_scores,
    const fp8_t* full_h,
    const uint32_t* full_h_sf,
    const fp8_t* full_grad_preact,
    const uint32_t* full_grad_preact_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_0,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    using Scheduler = WgradTileScheduler<kW2, kNumRanks, kNumSMs>;
    constexpr uint32_t kShapeM = Scheduler::kShapeM;
    constexpr uint32_t kShapeN = Scheduler::kShapeN;
    constexpr uint32_t kFullABlocks = kShapeM / 128;
    constexpr uint32_t kFullBBlocks = kShapeN / 128;
    DG_STATIC_ASSERT(Scheduler::kNumMBlocks % 2 == 0,
                     "2-CTA wgrad requires paired M tiles");
    DG_STATIC_ASSERT(kNumSMs % 2 == 0,
                     "2-CTA wgrad requires an even SM count");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const bool leader_cta = cute::block_rank_in_cluster() == 0;

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == kControlWarp && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kStages; ++i) {
            // A and B producer groups in both CTAs complete each stage.
            storage.full_barriers[i].init(4);
            storage.empty_barriers[i].init(1);
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

    if (warp_idx == kControlWarp) {
        cute::prefetch_tma_descriptor(&tensor_map_output_0);
        cute::prefetch_tma_descriptor(&tensor_map_output_1);
    }

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

    if (threadIdx.x < kAProducerThreads) {
        // A is [output-M, routes].  One 128-thread group consumes one
        // contiguous feature block for each route, then transposes into the
        // UMMA MN-major shared-memory tile.  W2 applies the exact FP32 score
        // before the required BF16 rounding.
        Scheduler scheduler(expert_counts);
        uint32_t expert, count, pool_row, m_block, n_block;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            DG_DEVICE_ASSERT(pool_row + count <= max_pool_tokens);
            const uint32_t feature =
                m_block * kBlockM + threadIdx.x;
            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.empty_barriers[stage_idx].wait(phase ^ 1u);
                #pragma unroll
                for (uint32_t k = 0; k < kBlockK; ++k) {
                    const uint32_t route = k_block * kBlockK + k;
                    float value = 0.0f;
                    if (route < count) {
                        const uint64_t full_row = pool_row + route;
                        if constexpr (kW2) {
                            const uint32_t packed = full_grad_y_sf[
                                full_row * kFullABlocks +
                                feature / 128];
                            value =
                                static_cast<float>(full_grad_y[
                                    full_row * kShapeM + feature]) *
                                unpack_power2_scale(packed) *
                                full_scores[full_row];
                        } else {
                            const uint32_t packed =
                                full_grad_preact_sf[
                                    full_row * kFullABlocks +
                                    feature / 128];
                            value =
                                static_cast<float>(full_grad_preact[
                                    full_row * kShapeM + feature]) *
                                unpack_power2_scale(packed);
                        }
                    }
                    store_mn_swizzle128<kBlockM>(
                        storage.smem_a[stage_idx],
                        threadIdx.x, k, bf16_t(value));
                }
                ptx::sync_aligned(128, 1);
                cutlass::arch::fence_view_async_shared();
                if (threadIdx.x == 0)
                    storage.full_barriers[stage_idx].arrive(0u);
                stage_idx = stage_idx == kStages - 1
                    ? 0
                    : stage_idx + 1;
                phase ^= stage_idx == 0;
            }
        }
    } else if (
        threadIdx.x >= kAProducerThreads &&
        threadIdx.x < kAProducerThreads + kBProducerThreads) {
        // B is [output-N, routes].  The two CTAs load adjacent 64-feature
        // halves, so all 128 N features for the cluster are coalesced.
        Scheduler scheduler(expert_counts);
        const uint32_t producer_lane =
            threadIdx.x - kAProducerThreads;
        uint32_t expert, count, pool_row, m_block, n_block;
        while (scheduler.get_next(
            expert, count, pool_row, m_block, n_block)) {
            DG_DEVICE_ASSERT(pool_row + count <= max_pool_tokens);
            const uint32_t feature =
                n_block * kBlockN +
                cute::block_rank_in_cluster() * kLoadBlockN +
                producer_lane;
            const uint32_t num_k_blocks =
                cute::max(1u, math::ceil_div(count, kBlockK));
            for (uint32_t k_block = 0; k_block < num_k_blocks;
                 ++k_block) {
                storage.empty_barriers[stage_idx].wait(phase ^ 1u);
                #pragma unroll
                for (uint32_t k = 0; k < kBlockK; ++k) {
                    const uint32_t route = k_block * kBlockK + k;
                    float value = 0.0f;
                    if (route < count) {
                        const uint64_t full_row = pool_row + route;
                        if constexpr (kW2) {
                            const uint32_t packed = full_h_sf[
                                full_row * kFullBBlocks +
                                feature / 128];
                            value =
                                static_cast<float>(full_h[
                                    full_row * kShapeN + feature]) *
                                unpack_power2_scale(packed);
                        } else {
                            const uint32_t packed = full_x_sf[
                                full_row * kFullBBlocks +
                                feature / 128];
                            value =
                                static_cast<float>(full_x[
                                    full_row * kShapeN + feature]) *
                                unpack_power2_scale(packed);
                        }
                    }
                    store_mn_swizzle128<kLoadBlockN>(
                        storage.smem_b[stage_idx],
                        producer_lane, k, bf16_t(value));
                }
                ptx::sync_aligned(64, 2);
                cutlass::arch::fence_view_async_shared();
                if (producer_lane == 0)
                    storage.full_barriers[stage_idx].arrive(0u);
                stage_idx = stage_idx == kStages - 1
                    ? 0
                    : stage_idx + 1;
                phase ^= stage_idx == 0;
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
                storage.full_barriers[stage_idx].wait(phase);
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
                        &storage.empty_barriers[stage_idx]),
                    kCTAMask);
                if (k_block == num_k_blocks - 1) {
                    cutlass::arch::umma_arrive_multicast_2x1SM(
                        reinterpret_cast<uint64_t*>(
                            &storage.tmem_full_barriers[accum_stage]),
                        kCTAMask);
                }
                __syncwarp();
                stage_idx = stage_idx == kStages - 1
                    ? 0
                    : stage_idx + 1;
                phase ^= stage_idx == 0;
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
