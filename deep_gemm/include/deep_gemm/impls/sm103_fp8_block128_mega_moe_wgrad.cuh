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
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sm103_block128_wgrad {

// GLM-5.2 has enough routes per expert that a single fixed 2-CTA family is
// preferable to a runtime configuration matrix.  K is the route dimension;
// BF16 UMMA consumes it in 64-row atoms after FP8 dequantization in the load
// prologue.  No full BF16 route pool is materialized.
static constexpr uint32_t kHidden = 6144;
static constexpr uint32_t kIntermediate = 2048;
static constexpr uint32_t kGlobalExperts = 256;
static constexpr uint32_t kTopK = 8;
static constexpr uint32_t kRouteBlockM = 192;
static constexpr uint32_t kBlockM = 128;
static constexpr uint32_t kBlockN = 128;
static constexpr uint32_t kBlockK = 64;
static constexpr uint32_t kLoadBlockN = kBlockN / 2;
static constexpr uint32_t kStages = 6;
static constexpr uint32_t kThreads = 256;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumTMAStoreStages = 2;
static constexpr uint32_t kStoreBlockM = 128;
static constexpr uint32_t kStoreBlockN = 64;
static constexpr uint32_t kEpilogueThreads = 128;
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

DG_STATIC_ASSERT(kNumTmemCols <= 512, "SM103 wgrad exceeds TMEM");

CUTLASS_DEVICE uint32_t transform_sf_row(const uint32_t row) {
    const uint32_t in_block = row % kRouteBlockM;
    return row / kRouteBlockM * 256u +
           (in_block & ~127u) + (in_block & 31u) * 4u +
           ((in_block >> 5) & 3u);
}

CUTLASS_DEVICE float unpack_power2_scale(const uint32_t packed) {
    const uint32_t exponent = packed & 0xffu;
    // UE8M0 code zero denotes 2^-127.  That value is an FP32 subnormal, so
    // constructing it by shifting an IEEE exponent field would incorrectly
    // produce zero.
    return exponent == 0u ? 0x1p-127f : __uint_as_float(exponent << 23);
}

template <uint32_t kRows>
CUTLASS_DEVICE void store_mn_swizzle128(
    bf16_t* base,
    const uint32_t row,
    const uint32_t k,
    const bf16_t value) {
    DG_STATIC_ASSERT(kRows == 64 || kRows == 128, "invalid BF16 SMEM rows");
    const uint32_t row_in_atom = row & 7u;
    const uint32_t col_byte = k * sizeof(bf16_t);
    const uint32_t byte_offset =
        (row >> 3) * 8u * kSwizzle + row_in_atom * kSwizzle +
        ((col_byte >> 4) ^ row_in_atom) * 16u + (col_byte & 15u);
    *reinterpret_cast<bf16_t*>(
        reinterpret_cast<uint8_t*>(base) + byte_offset) = value;
}

template <bool kW2, uint32_t kNumRanks, uint32_t kNumSMs>
CUTLASS_DEVICE void gather_compact_operand(
    const uint32_t count,
    const uint32_t pool_row_offset,
    const uint32_t sf_ring_tokens,
    const layout::TokenSrcMetadata* token_src_metadata,
    const layout::SymBuffer<kNumRanks>& sym_buffer,
    const fp8_t* symmetric_x,
    const uint32_t* symmetric_x_sf,
    const fp8_t* symmetric_grad_y,
    const uint32_t* symmetric_grad_y_sf,
    const float* symmetric_scores,
    fp8_t* ring_operand,
    uint32_t* ring_operand_sf,
    float* ring_scores) {
    constexpr uint32_t kHiddenBlocks = kHidden / 128;
    constexpr uint32_t kVecsPerRow = kHidden / sizeof(uint4);
    const uint32_t global_thread = blockIdx.x * kThreads + threadIdx.x;
    const uint32_t global_stride = kNumSMs * kThreads;
    const fp8_t* source = kW2 ? symmetric_grad_y : symmetric_x;
    const uint32_t* source_sf =
        kW2 ? symmetric_grad_y_sf : symmetric_x_sf;

    for (uint64_t linear = global_thread;
         linear < static_cast<uint64_t>(count) * kVecsPerRow;
         linear += global_stride) {
        const uint32_t row = linear / kVecsPerRow;
        const uint32_t vec = linear -
                             static_cast<uint64_t>(row) * kVecsPerRow;
        const auto metadata = token_src_metadata[pool_row_offset + row];
        const auto* remote = sym_buffer.map(
            reinterpret_cast<const uint4*>(source) +
                static_cast<uint64_t>(metadata.token_idx) * kVecsPerRow +
                vec,
            metadata.rank_idx);
        reinterpret_cast<uint4*>(ring_operand)[
            static_cast<uint64_t>(row) * kVecsPerRow + vec] = *remote;
    }
    for (uint64_t linear = global_thread;
         linear < static_cast<uint64_t>(count) * kHiddenBlocks;
         linear += global_stride) {
        const uint32_t row = linear / kHiddenBlocks;
        const uint32_t block = linear -
                               static_cast<uint64_t>(row) * kHiddenBlocks;
        const auto metadata = token_src_metadata[pool_row_offset + row];
        const uint64_t remote_index =
            static_cast<uint64_t>(metadata.token_idx) * kHiddenBlocks + block;
        ring_operand_sf[
            block * sf_ring_tokens + transform_sf_row(row)] =
            *sym_buffer.map(source_sf + remote_index, metadata.rank_idx);
    }
    if constexpr (kW2) {
        for (uint32_t row = global_thread; row < count;
             row += global_stride) {
            const auto metadata = token_src_metadata[pool_row_offset + row];
            ring_scores[row] = *sym_buffer.map(
                symmetric_scores +
                    static_cast<uint64_t>(metadata.token_idx) * kTopK +
                    metadata.topk_idx,
                metadata.rank_idx);
        }
    }
}

template <bool kW2, uint32_t kNumRanks, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kThreads, 1) void
sm103_fp8_block128_mega_moe_wgrad_impl(
    const int* expert_counts,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t sf_ring_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ layout::Workspace workspace,
    const fp8_t* symmetric_x,
    const uint32_t* symmetric_x_sf,
    const fp8_t* symmetric_grad_y,
    const uint32_t* symmetric_grad_y_sf,
    const float* symmetric_scores,
    fp8_t* ring_operand,
    uint32_t* ring_operand_sf,
    float* ring_scores,
    const fp8_t* full_h,
    const uint32_t* full_h_sf,
    const fp8_t* full_grad_preact,
    const uint32_t* full_grad_preact_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_0,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output_1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    constexpr uint32_t kLocalExperts = kGlobalExperts / kNumRanks;
    constexpr uint32_t kShapeM = kW2 ? kHidden : 2 * kIntermediate;
    constexpr uint32_t kShapeN = kW2 ? kIntermediate : kHidden;
    constexpr uint32_t kNumMBlocks = kShapeM / kBlockM;
    constexpr uint32_t kNumNBlocks = kShapeN / kBlockN;
    constexpr uint32_t kNumTilesPerExpert = kNumMBlocks * kNumNBlocks;
    constexpr uint32_t kFullABlocks =
        (kW2 ? kHidden : 2 * kIntermediate) / 128;
    constexpr uint32_t kFullBBlocks =
        (kW2 ? kIntermediate : kHidden) / 128;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buffer);
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const bool leader_cta = cute::block_rank_in_cluster() == 0;

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 3 && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kStages; ++i) {
            // A and B producer warps in both CTAs remotely arrive at CTA 0.
            storage.full_barriers[i].init(4);
            storage.empty_barriers[i].init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            storage.tmem_full_barriers[i].init(1);
            storage.tmem_empty_barriers[i].init(2 * kEpilogueThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();
    if (warp_idx == 3)
        cute::TMEM::Allocator2Sm().allocate(kNumTmemCols, &storage.tmem_ptr);
    comm::cluster_sync_with_relaxed_arrive();

    if (warp_idx == 3) {
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
    uint32_t output_iter = 0;
    uint32_t tma_stage_idx = 0;
    uint32_t pool_block_offset = 0;

    #pragma unroll 1
    for (uint32_t expert = 0; expert < kLocalExperts; ++expert) {
        const uint32_t count =
            static_cast<uint32_t>(__ldg(expert_counts + expert));
        const uint32_t pool_row_offset =
            pool_block_offset * kRouteBlockM;
        DG_DEVICE_ASSERT(count <= workspace.num_ring_tokens);
        DG_DEVICE_ASSERT(
            pool_row_offset + count <= workspace.num_max_pool_tokens);

        // Each dedicated wgrad kernel transports its one compact operand once
        // per expert.  All output tiles then reuse the local FP8 ring.
        gather_compact_operand<kW2, kNumRanks, kNumSMs>(
            count, pool_row_offset, sf_ring_tokens,
            token_src_metadata, sym_buffer,
            symmetric_x, symmetric_x_sf,
            symmetric_grad_y, symmetric_grad_y_sf,
            symmetric_scores,
            ring_operand, ring_operand_sf, ring_scores);
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        const uint32_t num_k_blocks =
            cute::max(1u, math::ceil_div(count, kBlockK));

        if (warp_idx == 0) {
            // A: (output-M, routes).  W2 reads remotely transported dy and
            // applies score before BF16 rounding; W13 reads local dpreact.
            for (uint32_t tile = blockIdx.x; tile < kNumTilesPerExpert;
                 tile += kNumSMs) {
                const uint32_t m_block = tile % kNumMBlocks;
                #pragma unroll 1
                for (uint32_t k_block = 0; k_block < num_k_blocks;
                     ++k_block) {
                    storage.empty_barriers[stage_idx].wait(phase ^ 1);
                    for (uint32_t linear = lane_idx;
                         linear < kBlockM * kBlockK; linear += 32) {
                        const uint32_t row = linear / kBlockK;
                        const uint32_t k = linear - row * kBlockK;
                        const uint32_t route = k_block * kBlockK + k;
                        const uint32_t m = m_block * kBlockM + row;
                        float value = 0.0f;
                        if (route < count) {
                            if constexpr (kW2) {
                                const uint32_t packed = ring_operand_sf[
                                    (m / 128) * sf_ring_tokens +
                                    transform_sf_row(route)];
                                value = static_cast<float>(
                                            ring_operand[
                                                static_cast<uint64_t>(route) *
                                                    kHidden +
                                                m]) *
                                        unpack_power2_scale(packed) *
                                        ring_scores[route];
                            } else {
                                const uint64_t full_row =
                                    pool_row_offset + route;
                                const uint32_t packed =
                                    full_grad_preact_sf[
                                        full_row * kFullABlocks + m / 128];
                                value = static_cast<float>(
                                            full_grad_preact[
                                                full_row * kShapeM + m]) *
                                        unpack_power2_scale(packed);
                            }
                        }
                        store_mn_swizzle128<kBlockM>(
                            storage.smem_a[stage_idx], row, k,
                            bf16_t(value));
                    }
                    cutlass::arch::fence_view_async_shared();
                    if (cute::elect_one_sync())
                        storage.full_barriers[stage_idx].arrive(0u);
                    __syncwarp();
                    stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
                    phase ^= stage_idx == 0;
                }
            }
        } else if (warp_idx == 1) {
            // B: (output-N, routes).  W2 reads local h; W13 reads the compact
            // x operand transported into the ring above.
            for (uint32_t tile = blockIdx.x; tile < kNumTilesPerExpert;
                 tile += kNumSMs) {
                const uint32_t n_block = tile / kNumMBlocks;
                const uint32_t cta_n_base =
                    n_block * kBlockN +
                    cute::block_rank_in_cluster() * kLoadBlockN;
                #pragma unroll 1
                for (uint32_t k_block = 0; k_block < num_k_blocks;
                     ++k_block) {
                    storage.empty_barriers[stage_idx].wait(phase ^ 1);
                    for (uint32_t linear = lane_idx;
                         linear < kLoadBlockN * kBlockK; linear += 32) {
                        const uint32_t row = linear / kBlockK;
                        const uint32_t k = linear - row * kBlockK;
                        const uint32_t route = k_block * kBlockK + k;
                        const uint32_t n = cta_n_base + row;
                        float value = 0.0f;
                        if (route < count) {
                            if constexpr (kW2) {
                                const uint64_t full_row =
                                    pool_row_offset + route;
                                const uint32_t packed = full_h_sf[
                                    full_row * kFullBBlocks + n / 128];
                                value = static_cast<float>(
                                            full_h[
                                                full_row * kShapeN + n]) *
                                        unpack_power2_scale(packed);
                            } else {
                                const uint32_t packed = ring_operand_sf[
                                    (n / 128) * sf_ring_tokens +
                                    transform_sf_row(route)];
                                value = static_cast<float>(
                                            ring_operand[
                                                static_cast<uint64_t>(route) *
                                                    kHidden +
                                                n]) *
                                        unpack_power2_scale(packed);
                            }
                        }
                        store_mn_swizzle128<kLoadBlockN>(
                            storage.smem_b[stage_idx], row, k,
                            bf16_t(value));
                    }
                    cutlass::arch::fence_view_async_shared();
                    if (cute::elect_one_sync())
                        storage.full_barriers[stage_idx].arrive(0u);
                    __syncwarp();
                    stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
                    phase ^= stage_idx == 0;
                }
            }
        } else if (warp_idx == 2 && leader_cta) {
            for (uint32_t tile = blockIdx.x; tile < kNumTilesPerExpert;
                 tile += kNumSMs, ++output_iter) {
                const uint32_t accum_stage =
                    output_iter % kNumEpilogueStages;
                const uint32_t accum_phase =
                    (output_iter / kNumEpilogueStages) & 1u;
                storage.tmem_empty_barriers[accum_stage].wait(
                    accum_phase ^ 1u);
                ptx::tcgen05_after_thread_sync();

                #pragma unroll 1
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
                    stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
                    phase ^= stage_idx == 0;
                }
            }
        } else if (warp_idx >= 4) {
            const uint32_t epilogue_warp_idx = warp_idx - 4;
            DG_TRAP_ONLY_DEVICE_ASSERT(
                ptx::ld_shared(&storage.tmem_ptr) == 0);
            for (uint32_t tile = blockIdx.x; tile < kNumTilesPerExpert;
                 tile += kNumSMs, ++output_iter) {
                const uint32_t m_block = tile % kNumMBlocks;
                const uint32_t n_block = tile / kNumMBlocks;
                const uint32_t accum_stage =
                    output_iter % kNumEpilogueStages;
                const uint32_t accum_phase =
                    (output_iter / kNumEpilogueStages) & 1u;
                storage.tmem_full_barriers[accum_stage].wait(accum_phase);
                ptx::tcgen05_after_thread_sync();

                const cute::TmaDescriptor* output_map =
                    &tensor_map_output_0;
                uint32_t output_m = expert * (kW2 ? kHidden : kIntermediate);
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

        // Do not overwrite the expert ring or start a new output wave until
        // every CTA has completed all stores for this expert.
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });
        pool_block_offset += math::ceil_div(count, kRouteBlockM);
    }

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 3)
        cute::TMEM::Allocator2Sm().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "SM103 MegaMoE wgrad has no fallback");
#endif
}

}  // namespace deep_gemm::sm103_block128_wgrad

#pragma clang diagnostic pop
