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
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kHidden, uint32_t kNumExperts, uint32_t BLOCK_M,
    uint32_t kNumSMs, uint32_t kNumThreads,
    CombineOrderMode kCombineOrderMode>
__device__ __forceinline__ void
bf16_mega_moe_reduce_post_down_route(
    const int* expert_counts,
    const cutlass::bfloat16_t* grad_y_unweighted,
    const cutlass::bfloat16_t* down_unweighted,
    float* grad_route_output,
    uint8_t* scratch) {
    constexpr uint32_t kTritonRouteBlockH = [] {
        uint32_t value = 1;
        while (value < kHidden && value < 8192)
            value <<= 1;
        return value;
    }();
    constexpr uint32_t kTritonRouteNumWarps = [] {
        uint32_t value = kTritonRouteBlockH / 256;
        value = value < 4 ? 4 : value;
        return value > 32 ? 32 : value;
    }();
    constexpr uint32_t kTritonRouteThreads =
        kTritonRouteNumWarps * 32;
    constexpr uint32_t kTritonRouteValuesPerThread =
        kTritonRouteBlockH / kTritonRouteThreads;
    DG_STATIC_ASSERT(
        kTritonRouteValuesPerThread == 2 ||
            kTritonRouteValuesPerThread == 4 ||
            kTritonRouteValuesPerThread == 8,
        "Unsupported Triton route reduction width");
    constexpr uint32_t kRouteInputPow2 = [] {
        uint32_t value = 1;
        constexpr uint32_t vectorized_columns = kHidden / 4;
        while (value < 512 &&
               (value << 1) <= vectorized_columns)
            value <<= 1;
        return value;
    }();

    auto* route_lane_sums = reinterpret_cast<float*>(scratch);
    auto* route_control = reinterpret_cast<uint32_t*>(scratch);
    if constexpr (
        kCombineOrderMode != CombineOrderMode::FixedTopK) {
        // A sub-CTA named barrier is not safe while the persistent kernel's
        // earlier role-specific register/barrier phases are still live.
        // Assign one route to the CTA instead. The first Triton-sized thread
        // group keeps exactly the same lane-to-column map and butterfly tree;
        // the remaining threads only participate in CTA phase barriers.
        auto* route_warp_arrivals =
            reinterpret_cast<uint32_t*>(
                scratch +
                kNumThreads * sizeof(float));
        if (threadIdx.x == 0)
            *route_warp_arrivals = 0;
        __syncthreads();
        uint32_t route_pool_block_offset = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0;
             expert_idx < kNumExperts; ++expert_idx) {
            const uint32_t num_tokens =
                static_cast<uint32_t>(
                    __ldg(expert_counts + expert_idx));
            for (uint32_t token_idx = blockIdx.x;
                 token_idx < num_tokens;
                 token_idx += kNumSMs) {
                const uint32_t pool_row =
                    route_pool_block_offset * BLOCK_M +
                    token_idx;
                const uint32_t route_lane = threadIdx.x;
                float grad_route = 0.0f;
                if (route_lane < kTritonRouteThreads) {
                    float grad_y[
                        kTritonRouteValuesPerThread];
                    float down[
                        kTritonRouteValuesPerThread];
                    #pragma unroll
                    for (uint32_t i = 0;
                         i <
                             kTritonRouteValuesPerThread;
                         ++i) {
                        const uint32_t col =
                            route_lane +
                            i * kTritonRouteThreads;
                        grad_y[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  grad_y_unweighted[
                                      static_cast<
                                          uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col])
                            : 0.0f;
                        down[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  down_unweighted[
                                      static_cast<
                                          uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col])
                            : 0.0f;
                    }
                    if constexpr (
                        kTritonRouteValuesPerThread == 2) {
                        grad_route = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[1], down[1]));
                    } else if constexpr (
                        kTritonRouteValuesPerThread == 4) {
                        const float even = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float odd = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        grad_route =
                            __fadd_rn(even, odd);
                    } else {
                        const float pair_02 = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float pair_13 = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        const float pair_46 = __fmaf_rn(
                            grad_y[4], down[4],
                            __fmul_rn(
                                grad_y[6], down[6]));
                        const float pair_57 = __fmaf_rn(
                            grad_y[5], down[5],
                            __fmul_rn(
                                grad_y[7], down[7]));
                        grad_route = __fadd_rn(
                            __fadd_rn(
                                pair_02, pair_46),
                            __fadd_rn(
                                pair_13, pair_57));
                    }
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                    const uint32_t lane_in_warp =
                        route_lane & 31;
                    if (lane_in_warp == 0) {
                        route_lane_sums[
                            route_lane / 32] =
                            grad_route;
                        __threadfence_block();
                        atomicAdd(
                            route_warp_arrivals, 1u);
                    }
                }
                if (threadIdx.x < 32) {
                    if (threadIdx.x == 0) {
                        while (atomicAdd(
                                   route_warp_arrivals,
                                   0u) !=
                               kTritonRouteNumWarps) {
                        }
                    }
                    __syncwarp();
                    grad_route = route_lane_sums[
                        threadIdx.x &
                        (kTritonRouteNumWarps - 1)];
                    #pragma unroll
                    for (uint32_t offset =
                             kTritonRouteNumWarps / 2;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                }
                if (threadIdx.x == 0)
                    grad_route_output[pool_row] =
                        grad_route;
                __syncthreads();
                if (threadIdx.x == 0)
                    *route_warp_arrivals = 0;
                __syncthreads();
            }
            route_pool_block_offset +=
                math::ceil_div(num_tokens, BLOCK_M);
        }
        return;
    }

    if (threadIdx.x == 0) {
        uint32_t total_route_rows = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0;
             expert_idx < kNumExperts; ++expert_idx) {
            total_route_rows += static_cast<uint32_t>(
                __ldg(expert_counts + expert_idx));
        }
        route_control[0] = total_route_rows;
    }
    __syncthreads();
    const uint32_t total_route_rows = route_control[0];
    const uint32_t route_output_pow2 =
        total_route_rows > 0
        ? 1u << (31 - __clz(total_route_rows))
        : 1u;
    constexpr uint32_t kInitialRouteGroupThreads =
        cute::min(kRouteInputPow2, 32u);
    const uint32_t route_block_height =
        cute::min(
            route_output_pow2,
            512u / kInitialRouteGroupThreads);
    const uint32_t route_group_threads =
        kCombineOrderMode != CombineOrderMode::FixedTopK
        ? kTritonRouteThreads
        : cute::min(
              kRouteInputPow2,
              512u / route_block_height);
    const uint32_t num_route_groups_per_cta =
        kNumThreads / route_group_threads;
    const uint32_t route_group_idx =
        threadIdx.x / route_group_threads;
    const uint32_t route_group_lane_idx =
        threadIdx.x & (route_group_threads - 1);
    const uint32_t global_route_group =
        blockIdx.x * num_route_groups_per_cta +
        route_group_idx;
    const uint32_t num_route_groups =
        kNumSMs * num_route_groups_per_cta;
    uint32_t route_pool_block_offset = 0;

    #pragma unroll
    for (uint32_t expert_idx = 0;
         expert_idx < kNumExperts; ++expert_idx) {
        const uint32_t num_tokens = static_cast<uint32_t>(
            __ldg(expert_counts + expert_idx));
        for (uint32_t token_idx = global_route_group;
             token_idx < num_tokens;
             token_idx += num_route_groups) {
            const uint32_t pool_row =
                route_pool_block_offset * BLOCK_M + token_idx;
            float grad_route = 0.0f;
            if constexpr (
                kCombineOrderMode != CombineOrderMode::FixedTopK) {
                float grad_y[kTritonRouteValuesPerThread];
                float down[kTritonRouteValuesPerThread];
                #pragma unroll
                for (uint32_t i = 0;
                     i < kTritonRouteValuesPerThread; ++i) {
                    const uint32_t col =
                        route_group_lane_idx +
                        i * kTritonRouteThreads;
                    grad_y[i] =
                        col < kHidden
                        ? static_cast<float>(
                              grad_y_unweighted[
                                  static_cast<uint64_t>(pool_row) *
                                      kHidden +
                                  col])
                        : 0.0f;
                    down[i] =
                        col < kHidden
                        ? static_cast<float>(
                              down_unweighted[
                                  static_cast<uint64_t>(pool_row) *
                                      kHidden +
                                  col])
                        : 0.0f;
                }

                if constexpr (kTritonRouteValuesPerThread == 2) {
                    grad_route = __fmaf_rn(
                        grad_y[0], down[0],
                        __fmul_rn(grad_y[1], down[1]));
                } else if constexpr (
                    kTritonRouteValuesPerThread == 4) {
                    const float even = __fmaf_rn(
                        grad_y[0], down[0],
                        __fmul_rn(grad_y[2], down[2]));
                    const float odd = __fmaf_rn(
                        grad_y[1], down[1],
                        __fmul_rn(grad_y[3], down[3]));
                    grad_route = __fadd_rn(even, odd);
                } else {
                    const float pair_02 = __fmaf_rn(
                        grad_y[0], down[0],
                        __fmul_rn(grad_y[2], down[2]));
                    const float pair_13 = __fmaf_rn(
                        grad_y[1], down[1],
                        __fmul_rn(grad_y[3], down[3]));
                    const float pair_46 = __fmaf_rn(
                        grad_y[4], down[4],
                        __fmul_rn(grad_y[6], down[6]));
                    const float pair_57 = __fmaf_rn(
                        grad_y[5], down[5],
                        __fmul_rn(grad_y[7], down[7]));
                    grad_route = __fadd_rn(
                        __fadd_rn(pair_02, pair_46),
                        __fadd_rn(pair_13, pair_57));
                }

                #pragma unroll
                for (uint32_t offset = 16;
                     offset > 0; offset >>= 1) {
                    grad_route = __fadd_rn(
                        grad_route,
                        __shfl_xor_sync(
                            0xffffffff, grad_route, offset));
                }
                const uint32_t warp_in_group =
                    route_group_lane_idx / 32;
                const uint32_t lane_in_warp =
                    route_group_lane_idx & 31;
                if (lane_in_warp == 0) {
                    route_lane_sums[
                        route_group_idx *
                            kTritonRouteNumWarps +
                        warp_in_group] = grad_route;
                }
                ptx::sync_aligned(
                    kTritonRouteThreads, route_group_idx);
                if (warp_in_group == 0) {
                    grad_route = route_lane_sums[
                        route_group_idx *
                            kTritonRouteNumWarps +
                        (lane_in_warp &
                         (kTritonRouteNumWarps - 1))];
                    #pragma unroll
                    for (uint32_t offset =
                             kTritonRouteNumWarps / 2;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                }
            } else {
                float lane_sums[4] = {
                    0.0f, 0.0f, 0.0f, 0.0f};
                for (uint32_t col_base =
                         route_group_lane_idx * 4;
                     col_base < kHidden;
                     col_base += route_group_threads * 4) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t col = col_base + i;
                        const float grad_y = static_cast<float>(
                            grad_y_unweighted[
                                static_cast<uint64_t>(pool_row) *
                                    kHidden +
                                col]);
                        const float down = static_cast<float>(
                            down_unweighted[
                                static_cast<uint64_t>(pool_row) *
                                    kHidden +
                                col]);
                        lane_sums[i] = __fadd_rn(
                            lane_sums[i],
                            __fmul_rn(grad_y, down));
                    }
                }
                grad_route = __fadd_rn(
                    __fadd_rn(lane_sums[0], lane_sums[1]),
                    lane_sums[2]);
                grad_route =
                    __fadd_rn(grad_route, lane_sums[3]);
                route_lane_sums[threadIdx.x] = grad_route;
                if (route_group_threads > 32) {
                    for (uint32_t offset =
                             route_group_threads / 2;
                         offset >= 32; offset >>= 1) {
                        ptx::sync_aligned(
                            route_group_threads,
                            route_group_idx);
                        if (route_group_lane_idx < offset) {
                            grad_route = __fadd_rn(
                                grad_route,
                                route_lane_sums[
                                    threadIdx.x + offset]);
                            route_lane_sums[threadIdx.x] =
                                grad_route;
                        }
                    }
                }
                if (route_group_lane_idx < 32) {
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_down_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                }
            }
            if (route_group_lane_idx == 0)
                grad_route_output[pool_row] = grad_route;
            if (route_group_threads > 32) {
                ptx::sync_aligned(
                    route_group_threads, route_group_idx);
            } else {
                __syncwarp();
            }
        }
        route_pool_block_offset +=
            math::ceil_div(num_tokens, BLOCK_M);
    }
}

template <
    uint32_t kHidden, uint32_t kNumExperts, uint32_t BLOCK_M,
    uint32_t kNumSMs, uint32_t kNumRanks,
    CombineOrderMode kCombineOrderMode,
    bool kDoReverseDispatch = true,
    bool kComputeRouteDot = true,
    bool kWriteWeighted = true,
    bool kWeightedSourceIsRhs = false,
    bool kSynchronizeRanks = true,
    bool kSynchronizeAfterDispatch = true,
    bool kBarrierOnly = false,
    bool kXPrepared = false,
    uint32_t kRoutePreludeThreads = 256>
CUTLASS_GLOBAL __launch_bounds__(1024, 1) void
sm100_bf16_mega_moe_backward_post_down_prelude(
    const int* expert_counts,
    const __grid_constant__ layout::Workspace
        backward_workspace,
    const __grid_constant__ layout::SymBuffer<kNumRanks>
        backward_sym_buffer,
    const cutlass::bfloat16_t* backward_grad_y,
    const cutlass::bfloat16_t* backward_x,
    const float* backward_topk_weights,
    float* backward_grad_route,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_topk,
    const uint32_t num_pool_rows,
    cutlass::bfloat16_t* grad_y_unweighted_output,
    cutlass::bfloat16_t* grad_y_weighted_output,
    cutlass::bfloat16_t* x_pool_output,
    float* route_weights_output,
    const cutlass::bfloat16_t* down_unweighted,
    float* grad_route_output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    constexpr uint32_t kNumThreads = 1024;
    if constexpr (kSynchronizeRanks) {
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumThreads, 0, 71>(
            backward_workspace,
            backward_sym_buffer,
            blockIdx.x,
            threadIdx.x,
            []() { __syncthreads(); });
    }
    if constexpr (kBarrierOnly) {
        if constexpr (kSynchronizeAfterDispatch) {
            // Profiling-only completion barrier. Production folds this into
            // the dispatch launch below to avoid an extra kernel launch.
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumThreads, 1, 72>(
                backward_workspace,
                backward_sym_buffer,
                blockIdx.x,
                threadIdx.x,
                []() { __syncthreads(); });
        }
        return;
    }
    constexpr uint32_t kTritonRouteBlockH = [] {
        uint32_t value = 1;
        while (value < kHidden && value < 8192)
            value <<= 1;
        return value;
    }();
    constexpr uint32_t kTritonRouteNumWarps = [] {
        uint32_t value = kTritonRouteBlockH / 256;
        value = value < 4 ? 4 : value;
        return value > 32 ? 32 : value;
    }();
    constexpr uint32_t kTritonRouteThreads =
        kTritonRouteNumWarps * 32;
    constexpr uint32_t kTritonRouteValuesPerThread =
        kTritonRouteBlockH / kTritonRouteThreads;
    constexpr bool kVirtualizeRouteLanes =
        kRoutePreludeThreads == 128;
    DG_STATIC_ASSERT(
        kRoutePreludeThreads == 128 ||
            kRoutePreludeThreads == 256,
        "POST_DOWN route prelude requires 128 or 256 physical threads");
    DG_STATIC_ASSERT(
        !kVirtualizeRouteLanes ||
            (kHidden == 2048 &&
             kCombineOrderMode !=
                 CombineOrderMode::FixedTopK &&
             kComputeRouteDot),
        "128-thread route prelude is only supported for the exact "
        "non-fixed H=2048 route-dot path");
    constexpr uint32_t kExactRouteGroupThreads =
        kVirtualizeRouteLanes
        ? kRoutePreludeThreads
        : kTritonRouteThreads;
    constexpr uint32_t kRouteVirtualLanes =
        kTritonRouteThreads /
        kExactRouteGroupThreads;
    DG_STATIC_ASSERT(
        kRouteVirtualLanes == 1 ||
            kRouteVirtualLanes == 2,
        "Unsupported POST_DOWN route lane virtualization");
    DG_STATIC_ASSERT(
        kTritonRouteValuesPerThread == 2 ||
            kTritonRouteValuesPerThread == 4 ||
            kTritonRouteValuesPerThread == 8,
        "Unsupported Triton route reduction width");
    constexpr uint32_t kRouteInputPow2 = [] {
        uint32_t value = 1;
        constexpr uint32_t vectorized_columns =
            kHidden / 4;
        while (value < 512 &&
               (value << 1) <= vectorized_columns)
            value <<= 1;
        return value;
    }();
    constexpr uint32_t kInitialRouteGroupThreads =
        cute::min(kRouteInputPow2, 32u);

    extern __shared__ __align__(1024) uint8_t scratch[];
    auto* route_lane_sums =
        reinterpret_cast<float*>(scratch);
    auto* route_control =
        reinterpret_cast<uint32_t*>(scratch);
    if (threadIdx.x == 0) {
        uint32_t total_route_rows = 0;
        #pragma unroll
        for (uint32_t expert_idx = 0;
             expert_idx < kNumExperts; ++expert_idx) {
            total_route_rows += static_cast<uint32_t>(
                __ldg(expert_counts + expert_idx));
        }
        route_control[0] = total_route_rows;
    }
    __syncthreads();
    const uint32_t total_route_rows = route_control[0];
    const uint32_t route_output_pow2 =
        total_route_rows > 0
        ? 1u << (31 - __clz(total_route_rows))
        : 1u;
    const uint32_t route_block_height =
        cute::min(
            route_output_pow2,
            512u / kInitialRouteGroupThreads);
    // The dispatch-only launch copies vectorized BF16 payloads and does not
    // need Triton's exact reduction lane map. Use more route groups per CTA
    // so remote reads have enough independent rows to cover NVLink latency.
    const uint32_t route_group_threads =
        !kComputeRouteDot && !kWriteWeighted
        ? cute::min(kRouteInputPow2, 128u)
        : kCombineOrderMode != CombineOrderMode::FixedTopK
        ? kExactRouteGroupThreads
        : cute::min(
              kRouteInputPow2,
              512u / route_block_height);
    const uint32_t num_route_groups_per_cta =
        kNumThreads / route_group_threads;
    const uint32_t route_group_idx =
        threadIdx.x / route_group_threads;
    const uint32_t route_group_lane_idx =
        threadIdx.x & (route_group_threads - 1);
    const uint32_t global_route_group =
        blockIdx.x * num_route_groups_per_cta +
        route_group_idx;
    const uint32_t num_route_groups =
        kNumSMs * num_route_groups_per_cta;
    uint32_t route_pool_block_offset = 0;

    #pragma unroll
    for (uint32_t expert_idx = 0;
         expert_idx < kNumExperts; ++expert_idx) {
        const uint32_t num_tokens = static_cast<uint32_t>(
            __ldg(expert_counts + expert_idx));
        for (uint32_t token_idx = global_route_group;
             token_idx < num_tokens;
             token_idx += num_route_groups) {
            const uint32_t pool_row =
                route_pool_block_offset * BLOCK_M +
                token_idx;
            const cutlass::bfloat16_t* remote_grad_y;
            const cutlass::bfloat16_t* remote_x;
            if constexpr (kDoReverseDispatch) {
                const auto metadata =
                    token_src_metadata[pool_row];
                remote_grad_y =
                    backward_sym_buffer.map(
                        backward_grad_y +
                            static_cast<uint64_t>(
                                metadata.token_idx) *
                                kHidden,
                        metadata.rank_idx);
                if constexpr (kXPrepared) {
                    remote_x =
                        x_pool_output +
                        static_cast<uint64_t>(pool_row) *
                            kHidden;
                } else {
                    remote_x =
                        backward_sym_buffer.map(
                            backward_x +
                                static_cast<uint64_t>(
                                    metadata.token_idx) *
                                    kHidden,
                            metadata.rank_idx);
                }
                if (route_group_lane_idx == 0) {
                    const auto* remote_weight =
                        backward_sym_buffer.map(
                            backward_topk_weights +
                                static_cast<uint64_t>(
                                    metadata.token_idx) *
                                    num_topk +
                                metadata.topk_idx,
                            metadata.rank_idx);
                    route_weights_output[pool_row] =
                        *remote_weight;
                }
            } else {
                remote_grad_y =
                    grad_y_unweighted_output +
                    static_cast<uint64_t>(pool_row) *
                        kHidden;
                remote_x =
                    x_pool_output +
                    static_cast<uint64_t>(pool_row) *
                        kHidden;
            }

            float grad_route = 0.0f;
            if constexpr (!kComputeRouteDot && !kWriteWeighted) {
                constexpr uint32_t kBF16ValuesPerVector =
                    sizeof(uint4) /
                    sizeof(cutlass::bfloat16_t);
                DG_STATIC_ASSERT(
                    kHidden % kBF16ValuesPerVector == 0,
                    "BF16 dispatch requires vector-aligned hidden");
                for (uint32_t col =
                         route_group_lane_idx *
                         kBF16ValuesPerVector;
                     col < kHidden;
                     col += route_group_threads *
                            kBF16ValuesPerVector) {
                    const uint64_t offset =
                        static_cast<uint64_t>(
                            pool_row) *
                            kHidden +
                        col;
                    reinterpret_cast<uint4*>(
                        grad_y_unweighted_output)[
                        offset /
                        kBF16ValuesPerVector] =
                        reinterpret_cast<
                            const uint4*>(
                            remote_grad_y)[
                            col /
                            kBF16ValuesPerVector];
                    if constexpr (!kXPrepared) {
                        reinterpret_cast<uint4*>(
                            x_pool_output)[
                            offset /
                            kBF16ValuesPerVector] =
                            reinterpret_cast<
                                const uint4*>(
                                remote_x)[
                                col /
                                kBF16ValuesPerVector];
                    }
                }
            } else if constexpr (
                !kComputeRouteDot && kWriteWeighted) {
                const float route_weight =
                    route_weights_output[pool_row];
                for (uint32_t col =
                         route_group_lane_idx;
                     col < kHidden;
                     col += route_group_threads) {
                    const uint64_t offset =
                        static_cast<uint64_t>(
                            pool_row) *
                            kHidden +
                        col;
                    grad_y_weighted_output[offset] =
                        cutlass::bfloat16_t(
                            static_cast<float>(
                                (kWeightedSourceIsRhs
                                     ? down_unweighted[
                                           static_cast<uint64_t>(
                                               pool_row) *
                                               kHidden +
                                           col]
                                     : remote_grad_y[col])) *
                            route_weight);
                }
            } else if constexpr (
                kCombineOrderMode !=
                CombineOrderMode::FixedTopK) {
                if constexpr (kVirtualizeRouteLanes) {
                    // Each physical lane evaluates logical lanes p and
                    // p + 128. Their FMA and warp-XOR trees remain separate,
                    // then the four physical warps publish all eight logical
                    // Triton warp partials for the unchanged second level.
                    float weighted_values
                        [kRouteVirtualLanes]
                        [kTritonRouteValuesPerThread];
                    #pragma unroll
                    for (uint32_t virtual_lane = 0;
                     virtual_lane < kRouteVirtualLanes;
                     ++virtual_lane) {
                    const uint32_t logical_route_lane =
                        route_group_lane_idx +
                        virtual_lane *
                            kExactRouteGroupThreads;
                    float grad_y[
                        kTritonRouteValuesPerThread];
                    float down[
                        kTritonRouteValuesPerThread];
                    #pragma unroll
                    for (uint32_t i = 0;
                         i <
                             kTritonRouteValuesPerThread;
                         ++i) {
                        const uint32_t col =
                            logical_route_lane +
                            i * kTritonRouteThreads;
                        grad_y[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  remote_grad_y[col])
                            : 0.0f;
                        down[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  down_unweighted[
                                      static_cast<uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col])
                            : 0.0f;
                        if constexpr (
                            kDoReverseDispatch &&
                            !kXPrepared) {
                            if (col < kHidden) {
                                x_pool_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] =
                                    remote_x[col];
                            }
                        }
                        if constexpr (kWriteWeighted) {
                            weighted_values[virtual_lane][i] =
                                kWeightedSourceIsRhs
                                ? down[i]
                                : grad_y[i];
                        }
                    }
                    float logical_grad_route;
                    if constexpr (
                        kTritonRouteValuesPerThread ==
                        2) {
                        logical_grad_route = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[1], down[1]));
                    } else if constexpr (
                        kTritonRouteValuesPerThread ==
                        4) {
                        const float even = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float odd = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        logical_grad_route =
                            __fadd_rn(even, odd);
                    } else {
                        const float pair_02 = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float pair_13 = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        const float pair_46 = __fmaf_rn(
                            grad_y[4], down[4],
                            __fmul_rn(
                                grad_y[6], down[6]));
                        const float pair_57 = __fmaf_rn(
                            grad_y[5], down[5],
                            __fmul_rn(
                                grad_y[7], down[7]));
                        logical_grad_route = __fadd_rn(
                            __fadd_rn(
                                pair_02, pair_46),
                            __fadd_rn(
                                pair_13, pair_57));
                    }
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        logical_grad_route = __fadd_rn(
                            logical_grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                logical_grad_route,
                                offset));
                    }
                    const uint32_t lane_in_warp =
                        route_group_lane_idx & 31;
                    const uint32_t logical_warp =
                        logical_route_lane / 32;
                    if (lane_in_warp == 0) {
                        route_lane_sums[
                            route_group_idx *
                                kTritonRouteNumWarps +
                            logical_warp] =
                            logical_grad_route;
                    }
                }
                ptx::sync_aligned(
                    kExactRouteGroupThreads,
                    route_group_idx);
                const uint32_t physical_warp_in_group =
                    route_group_lane_idx / 32;
                const uint32_t lane_in_warp =
                    route_group_lane_idx & 31;
                if (physical_warp_in_group == 0) {
                    grad_route = route_lane_sums[
                        route_group_idx *
                            kTritonRouteNumWarps +
                        (lane_in_warp &
                         (kTritonRouteNumWarps - 1))];
                    #pragma unroll
                    for (uint32_t offset =
                             kTritonRouteNumWarps / 2;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                }
                if constexpr (kWriteWeighted) {
                    const float route_weight =
                        route_weights_output[pool_row];
                    #pragma unroll
                    for (uint32_t virtual_lane = 0;
                         virtual_lane <
                             kRouteVirtualLanes;
                         ++virtual_lane) {
                        #pragma unroll
                        for (uint32_t i = 0;
                             i <
                                 kTritonRouteValuesPerThread;
                             ++i) {
                            const uint32_t col =
                                route_group_lane_idx +
                                virtual_lane *
                                    kExactRouteGroupThreads +
                                i *
                                    kTritonRouteThreads;
                            if (col < kHidden) {
                                grad_y_weighted_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] =
                                    cutlass::bfloat16_t(
                                        weighted_values[
                                            virtual_lane]
                                            [i] *
                                        route_weight);
                            }
                        }
                    }
                }
                } else {
                    float grad_y[
                        kTritonRouteValuesPerThread];
                    float down[
                        kTritonRouteValuesPerThread];
                    #pragma unroll
                    for (uint32_t i = 0;
                         i <
                             kTritonRouteValuesPerThread;
                         ++i) {
                        const uint32_t col =
                            route_group_lane_idx +
                            i * kTritonRouteThreads;
                        grad_y[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  remote_grad_y[col])
                            : 0.0f;
                        down[i] =
                            col < kHidden
                            ? static_cast<float>(
                                  down_unweighted[
                                      static_cast<uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col])
                            : 0.0f;
                        if constexpr (
                            kDoReverseDispatch &&
                            !kXPrepared) {
                            if (col < kHidden) {
                                x_pool_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] =
                                    remote_x[col];
                            }
                        }
                    }
                    if constexpr (
                        kTritonRouteValuesPerThread ==
                        2) {
                        grad_route = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[1], down[1]));
                    } else if constexpr (
                        kTritonRouteValuesPerThread ==
                        4) {
                        const float even = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float odd = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        grad_route =
                            __fadd_rn(even, odd);
                    } else {
                        const float pair_02 = __fmaf_rn(
                            grad_y[0], down[0],
                            __fmul_rn(
                                grad_y[2], down[2]));
                        const float pair_13 = __fmaf_rn(
                            grad_y[1], down[1],
                            __fmul_rn(
                                grad_y[3], down[3]));
                        const float pair_46 = __fmaf_rn(
                            grad_y[4], down[4],
                            __fmul_rn(
                                grad_y[6], down[6]));
                        const float pair_57 = __fmaf_rn(
                            grad_y[5], down[5],
                            __fmul_rn(
                                grad_y[7], down[7]));
                        grad_route = __fadd_rn(
                            __fadd_rn(
                                pair_02, pair_46),
                            __fadd_rn(
                                pair_13, pair_57));
                    }
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_xor_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                    const uint32_t warp_in_group =
                        route_group_lane_idx / 32;
                    const uint32_t lane_in_warp =
                        route_group_lane_idx & 31;
                    if (lane_in_warp == 0) {
                        route_lane_sums[
                            route_group_idx *
                                kTritonRouteNumWarps +
                            warp_in_group] =
                            grad_route;
                    }
                    ptx::sync_aligned(
                        kTritonRouteThreads,
                        route_group_idx);
                    if (warp_in_group == 0) {
                        grad_route = route_lane_sums[
                            route_group_idx *
                                kTritonRouteNumWarps +
                            (lane_in_warp &
                             (kTritonRouteNumWarps -
                              1))];
                        #pragma unroll
                        for (uint32_t offset =
                                 kTritonRouteNumWarps /
                                 2;
                             offset > 0;
                             offset >>= 1) {
                            grad_route = __fadd_rn(
                                grad_route,
                                __shfl_xor_sync(
                                    0xffffffff,
                                    grad_route,
                                    offset));
                        }
                    }
                    if constexpr (kWriteWeighted) {
                        const float route_weight =
                            route_weights_output[
                                pool_row];
                        #pragma unroll
                        for (uint32_t i = 0;
                             i <
                                 kTritonRouteValuesPerThread;
                             ++i) {
                            const uint32_t col =
                                route_group_lane_idx +
                                i *
                                    kTritonRouteThreads;
                            if (col < kHidden) {
                                grad_y_weighted_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] =
                                    cutlass::bfloat16_t(
                                        (kWeightedSourceIsRhs
                                             ? down[i]
                                             : grad_y[i]) *
                                        route_weight);
                            }
                        }
                    }
                }
            } else {
                float lane_sums[4] = {
                    0.0f, 0.0f, 0.0f, 0.0f};
                const float route_weight =
                    kWriteWeighted
                    ? route_weights_output[pool_row]
                    : 0.0f;
                for (uint32_t col_base =
                         route_group_lane_idx * 4;
                     col_base < kHidden;
                     col_base +=
                         route_group_threads * 4) {
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint32_t col =
                            col_base + i;
                        const float grad_y =
                            static_cast<float>(
                                remote_grad_y[col]);
                        const float down =
                            static_cast<float>(
                                down_unweighted[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col]);
                        if constexpr (kWriteWeighted) {
                            grad_y_weighted_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kHidden +
                                col] =
                                cutlass::bfloat16_t(
                                    (kWeightedSourceIsRhs
                                         ? down
                                         : grad_y) *
                                    route_weight);
                        }
                        if constexpr (kDoReverseDispatch) {
                            grad_y_unweighted_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kHidden +
                                col] =
                                cutlass::bfloat16_t(
                                    grad_y);
                            if constexpr (!kXPrepared) {
                                x_pool_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kHidden +
                                    col] =
                                    remote_x[col];
                            }
                        }
                        lane_sums[i] = __fadd_rn(
                            lane_sums[i],
                            __fmul_rn(
                                grad_y, down));
                    }
                }
                grad_route = __fadd_rn(
                    __fadd_rn(
                        lane_sums[0],
                        lane_sums[1]),
                    lane_sums[2]);
                grad_route = __fadd_rn(
                    grad_route, lane_sums[3]);
                route_lane_sums[threadIdx.x] =
                    grad_route;
                if (route_group_threads > 32) {
                    for (uint32_t offset =
                             route_group_threads / 2;
                         offset >= 32;
                         offset >>= 1) {
                        ptx::sync_aligned(
                            route_group_threads,
                            route_group_idx);
                        if (route_group_lane_idx <
                            offset) {
                            grad_route = __fadd_rn(
                                grad_route,
                                route_lane_sums[
                                    threadIdx.x +
                                    offset]);
                            route_lane_sums[
                                threadIdx.x] =
                                grad_route;
                        }
                    }
                }
                if (route_group_lane_idx < 32) {
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_down_sync(
                                0xffffffff,
                                grad_route, offset));
                    }
                }
            }
            if constexpr (kComputeRouteDot) {
                if (route_group_lane_idx == 0) {
                    grad_route_output[pool_row] =
                        grad_route;
                    if (backward_grad_route != nullptr) {
                        const auto metadata =
                            token_src_metadata[pool_row];
                        auto* remote_grad_route =
                            backward_sym_buffer.map(
                                backward_grad_route +
                                    static_cast<uint64_t>(
                                        metadata.token_idx) *
                                        num_topk +
                                    metadata.topk_idx,
                                metadata.rank_idx);
                        *remote_grad_route = grad_route;
                    }
                }
                if (route_group_threads > 32) {
                    ptx::sync_aligned(
                        route_group_threads,
                        route_group_idx);
                } else {
                    __syncwarp();
                }
            }
        }
        route_pool_block_offset +=
            math::ceil_div(num_tokens, BLOCK_M);
    }

    uint32_t padded_pool_blocks = 0;
    #pragma unroll
    for (uint32_t expert_idx = 0;
         expert_idx < kNumExperts; ++expert_idx) {
        const uint32_t num_tokens = static_cast<uint32_t>(
            __ldg(expert_counts + expert_idx));
        const uint32_t num_blocks =
            math::ceil_div(num_tokens, BLOCK_M);
        const uint32_t num_padded_tokens =
            num_blocks * BLOCK_M;
        const uint32_t num_padding_rows =
            num_padded_tokens - num_tokens;
        for (uint64_t linear =
                 static_cast<uint64_t>(blockIdx.x) *
                     kNumThreads +
                 threadIdx.x;
             linear <
                 static_cast<uint64_t>(
                     num_padding_rows) *
                     kHidden;
             linear +=
                 static_cast<uint64_t>(kNumSMs) *
                     kNumThreads) {
            const uint32_t padding_row =
                linear / kHidden;
            const uint32_t col =
                linear -
                static_cast<uint64_t>(padding_row) *
                    kHidden;
            const uint32_t pool_row =
                padded_pool_blocks * BLOCK_M +
                num_tokens + padding_row;
            const uint64_t offset =
                static_cast<uint64_t>(pool_row) *
                    kHidden +
                col;
            if constexpr (kDoReverseDispatch) {
                grad_y_unweighted_output[offset] =
                    cutlass::bfloat16_t(0.0f);
                if constexpr (!kXPrepared) {
                    x_pool_output[offset] =
                        cutlass::bfloat16_t(0.0f);
                }
            }
            if constexpr (kWriteWeighted) {
                grad_y_weighted_output[offset] =
                    cutlass::bfloat16_t(0.0f);
            }
        }
        for (uint32_t padding_row =
                 blockIdx.x * kNumThreads +
                 threadIdx.x;
             padding_row < num_padding_rows;
             padding_row +=
                 kNumSMs * kNumThreads) {
            const uint32_t pool_row =
                padded_pool_blocks * BLOCK_M +
                num_tokens + padding_row;
            if constexpr (kDoReverseDispatch)
                route_weights_output[pool_row] = 0.0f;
            if constexpr (kComputeRouteDot)
                grad_route_output[pool_row] = 0.0f;
        }
        padded_pool_blocks += num_blocks;
    }
    if constexpr (kSynchronizeAfterDispatch) {
        // Every rank may reuse its local symmetric grad-y plane as the
        // direct-write grad-x destination as soon as this producer returns.
        // Publish completion only after all peers have finished their remote
        // reads; an entry-only barrier permits checkpoint/replay rank skew to
        // corrupt those in-flight pulls.
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumThreads, 1, 72>(
            backward_workspace,
            backward_sym_buffer,
            blockIdx.x,
            threadIdx.x,
            []() { __syncthreads(); });
    }
    // Rows beyond the final padded expert block are capacity only. No
    // downstream kernel addresses them, so clearing that high-water tail
    // wastes bandwidth and grows with the cached pool margin.
#endif
}

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
    CombineOrderMode kCombineOrderMode = CombineOrderMode::FixedTopK,
    bool kInputsPrepared = false,
    bool kDispatchInputsPrepared = false,
    bool kDirectRemoteGradX = false,
    bool kWriteGradXPool = true,
    bool kClearWgradPadding = false,
    bool kComputeRouteGrad = false,
    bool kTraceKernel = false,
    bool kVectorizedGradXStore = false,
    bool kWideGradXStore = false,
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
    float* backward_grad_route,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_topk,
    const uint32_t num_pool_rows,
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
    uint64_t* kernel_trace) {
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
    constexpr uint32_t kNumW13DgradSplits =
        kBF16Mode ? 2 : 1;
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

    constexpr uint32_t kNumW13WeightTileStates =
        kNumExperts *
        ((2 * kIntermediateHidden) / DGRAD_BLOCK_K) *
        kNumW13DgradBlockNs;
    auto* phase_count =
        weight_tile_states + kNumW2WeightTileStates +
        kNumW13WeightTileStates;
    auto* phase_sense = phase_count + 1;
    constexpr uint32_t kTraceSiteCount = 22;
    constexpr uint32_t kTraceValueCount = 5;
    constexpr uint32_t kTraceBeginCycle = 0;
    constexpr uint32_t kTraceEndCycle = 1;
    constexpr uint32_t kTraceBeginGlobalNs = 2;
    constexpr uint32_t kTraceEndGlobalNs = 3;
    constexpr uint32_t kTraceSM = 4;
    const auto globaltimer = [] {
        uint64_t value;
        asm volatile(
            "mov.u64 %0, %%globaltimer;" : "=l"(value));
        return value;
    };
    const auto trace_begin = [&](const uint32_t site) {
        if constexpr (kTraceKernel) {
            if (threadIdx.x == 0) {
                auto* values =
                    kernel_trace +
                    (static_cast<uint64_t>(site) * kNumSMs +
                     blockIdx.x) *
                        kTraceValueCount;
                values[kTraceBeginCycle] = clock64();
                values[kTraceBeginGlobalNs] = globaltimer();
                values[kTraceSM] = ptx::get_sm_idx();
            }
        }
    };
    const auto trace_end = [&](const uint32_t site) {
        if constexpr (kTraceKernel) {
            if (threadIdx.x == 0) {
                auto* values =
                    kernel_trace +
                    (static_cast<uint64_t>(site) * kNumSMs +
                     blockIdx.x) *
                        kTraceValueCount;
                values[kTraceEndCycle] = clock64();
                values[kTraceEndGlobalNs] = globaltimer();
            }
        }
    };
    if constexpr (kTraceKernel) {
        DG_STATIC_ASSERT(
            kTraceSiteCount == 22,
            "Update the host trace-site schema with the kernel");
        trace_begin(0);
    }
    const auto full_grid_phase_barrier =
        [&](const uint32_t trace_site) {
        trace_begin(trace_site);
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
        trace_end(trace_site);
    };

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_grad_ye);
        cute::prefetch_tma_descriptor(&tensor_map_w2_dequant);
        cute::prefetch_tma_descriptor(&tensor_map_w13_dequant);
        cute::prefetch_tma_descriptor(&tensor_map_grad_gate_up);
        if constexpr (!kBF16Mode) {
            cute::prefetch_tma_descriptor(&tensor_map_acts);
            cute::prefetch_tma_descriptor(&tensor_map_acts_sf);
            cute::prefetch_tma_descriptor(&tensor_map_weights);
            cute::prefetch_tma_descriptor(&tensor_map_weights_sf);
            cute::prefetch_tma_descriptor(&tensor_map_output);
            cute::prefetch_tma_descriptor(&tensor_map_w2_weights);
            cute::prefetch_tma_descriptor(&tensor_map_w2_scales);
            cute::prefetch_tma_descriptor(&tensor_map_w13_weights);
            cute::prefetch_tma_descriptor(&tensor_map_w13_scales);
        }
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
    trace_begin(1);
    comm::cluster_sync_with_relaxed_arrive();
    trace_end(1);
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
    trace_begin(2);
    comm::cluster_sync_with_relaxed_arrive();
    trace_end(2);

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
            kBF16Mode ? 56 : 48;
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();
        if constexpr (
            kNumRanks > 1 &&
            !(kBF16Mode && kDispatchInputsPrepared)) {
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
            trace_begin(3);
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
            trace_end(3);

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
                        if (route_weights_fp32 != nullptr) {
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
            trace_begin(4);
            comm::grid_sync<
                kNumSMs, kDispatchDoneGridSyncIndex>(
                backward_workspace, blockIdx.x,
                dispatch_thread_idx,
                [=]() {
                    ptx::sync_aligned(
                        kNumDispatchThreads,
                        kDispatchNamedBarrierIdx);
                });
            trace_end(4);
        }
    } else if (warp_idx >= 12) {
        // W13 wgrad needs the exact BF16 value represented by the forward
        // FP8+UE8M0 pool.  Produce it while the recompute MMA is running, using
        // otherwise-idle warps.  Padding rows are explicitly zeroed so the
        // k-grouped wgrad mainloop can round K up to 64 without reading the
        // following expert.
        constexpr uint32_t kNumXPoolRegisters =
            kBF16Mode ? 56 : 40;
        cutlass::arch::warpgroup_reg_dealloc<
            kNumXPoolRegisters>();
        constexpr uint32_t kFirstXPoolWarp = 12;
        constexpr uint32_t kNumXPoolThreads =
            kNumThreads - kFirstXPoolWarp * 32;
        const uint32_t x_thread_idx =
            (warp_idx - kFirstXPoolWarp) * 32 + lane_idx;
        if constexpr (!(kBF16Mode && kDispatchInputsPrepared)) {
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
        }
    } else {
        constexpr uint32_t kNumIdleRegisters =
            kBF16Mode ? 56 : 24;
        cutlass::arch::warpgroup_reg_dealloc<
            kNumIdleRegisters>();
    }

    {
        __syncthreads();
        if constexpr (
            kBF16Mode && kNumRanks == 1 &&
            !kDispatchInputsPrepared) {
            // In single-rank BF16 mode the x-pool warps also stage grad-y.
            // Their pool-block assignment is independent of the dgrad tile
            // assignment, so a cluster barrier is insufficient before W2
            // dgrad starts consuming the completed expert pool.
            constexpr uint32_t kLocalDispatchDoneGridSyncIndex = 1;
            trace_begin(5);
            comm::grid_sync<
                kNumSMs, kLocalDispatchDoneGridSyncIndex>(
                backward_workspace, blockIdx.x, threadIdx.x,
                []() { __syncthreads(); });
            trace_end(5);
        }
        if constexpr (
            (kBF16Mode ||
             kRouteWeightMode ==
                 RouteWeightMode::PostDown) &&
            !kDispatchInputsPrepared) {
            uint32_t grad_pool_block_offset = 0;
            #pragma unroll
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens =
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                const uint32_t num_padded_tokens =
                    math::ceil_div(num_tokens, BLOCK_M) *
                    BLOCK_M;
                for (uint64_t linear =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear <
                         static_cast<uint64_t>(
                             num_padded_tokens) *
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
                        token_idx >= num_tokens
                        ? cd_dtype_t(0.0f)
                        : kRouteWeightMode ==
                                  RouteWeightMode::PostDown
                        ? cd_dtype_t(
                              static_cast<float>(
                                  grad_y_unweighted_output[
                                      static_cast<uint64_t>(
                                          pool_row) *
                                          kHidden +
                                      col]) *
                              (route_weights_fp32 != nullptr
                                   ? route_weights_fp32[
                                         pool_row]
                                   : static_cast<float>(
                                         route_weights[
                                             pool_row])))
                        : grad_y_unweighted_output[
                              static_cast<uint64_t>(
                                  pool_row) *
                                  kHidden +
                              col];
                }
                grad_pool_block_offset +=
                    math::ceil_div(num_tokens, BLOCK_M);
            }
            if constexpr (kBF16Mode) {
                constexpr uint32_t
                    kW2GradInputGridSyncIndex = 0;
                trace_begin(6);
                comm::grid_sync<
                    kNumSMs,
                    kW2GradInputGridSyncIndex>(
                    backward_workspace, blockIdx.x,
                    threadIdx.x,
                    []() { __syncthreads(); });
                trace_end(6);
            } else {
                // Standalone MXFP4 backward has no symmetric Workspace.
                // Reuse its launch-epoch grid state for the same publication
                // barrier before W2 dgrad consumes weighted grad-y.
                full_grid_phase_barrier(6);
            }
        }
        if constexpr (kDirectRemoteGradX) {
            if constexpr (kNumRanks > 1) {
                // backward_grad_y aliases combine plane zero. All ranks must
                // finish remotely pulling it before any W13 dgrad epilogue
                // reuses the combine planes for direct grad-x writes.
                if constexpr (
                    !(kBF16Mode && kDispatchInputsPrepared)) {
                    constexpr uint32_t
                        kBeforeDirectGradXGridSyncIndex = 2;
                    constexpr uint32_t
                        kBeforeDirectGradXBarrierTag = 7;
                    trace_begin(7);
                    comm::nvlink_barrier<
                        kNumRanks, kNumSMs, kNumThreads,
                        kBeforeDirectGradXGridSyncIndex,
                        kBeforeDirectGradXBarrierTag>(
                        backward_workspace,
                        backward_sym_buffer,
                        blockIdx.x, threadIdx.x,
                        []() { __syncthreads(); });
                    trace_end(7);
                }

            }

            if constexpr (
                kCombineOrderMode ==
                CombineOrderMode::FixedTopK) {
                // FixedTopK consumes every physical slot, including invalid
                // routes. Clear all slot planes only after all grad-y pulls
                // have completed, then publish the clear before any direct
                // remote stores. This also makes repeated and single-rank
                // calls independent of stale valid routes.
                auto* combine_buffer =
                    const_cast<cd_dtype_t*>(backward_grad_y);
                const uint64_t num_plane_values =
                    static_cast<uint64_t>(num_topk) *
                    backward_workspace.num_max_tokens_per_rank *
                    kHidden;
                for (uint64_t linear =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear < num_plane_values;
                     linear +=
                         static_cast<uint64_t>(kNumSMs) *
                         kNumThreads) {
                    combine_buffer[linear] =
                        cd_dtype_t(0.0f);
                }

                if constexpr (kNumRanks > 1) {
                    // Do not let a rank remotely write direct grad-x until
                    // every destination has finished clearing its local
                    // slot planes.
                    constexpr uint32_t
                        kAfterGradYClearGridSyncIndex = 3;
                    constexpr uint32_t
                        kAfterGradYClearBarrierTag = 8;
                    trace_begin(8);
                    comm::nvlink_barrier<
                        kNumRanks, kNumSMs, kNumThreads,
                        kAfterGradYClearGridSyncIndex,
                        kAfterGradYClearBarrierTag>(
                        backward_workspace,
                        backward_sym_buffer,
                        blockIdx.x, threadIdx.x,
                        []() { __syncthreads(); });
                    trace_end(8);
                }
            }
        }
        if constexpr (!kBF16Mode) {
            if (warp_idx >= kDispatchWarpStart &&
                warp_idx <
                    kDispatchWarpStart +
                        kNumDispatchWarps) {
                // Dispatch used 48 registers; transition down to the common
                // dgrad epilogue budget with dealloc, not reg_alloc
                // (allocating a lower count is illegal on SM100).
                cutlass::arch::warpgroup_reg_dealloc<40>();
            } else if (warp_idx >= kDispatchWarpStart) {
                cutlass::arch::warpgroup_reg_alloc<40>();
            }
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

        trace_begin(9);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(9);

        trace_begin(10);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(10);

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
        trace_begin(11);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(11);

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
                                route_weights_fp32 != nullptr
                                ? route_weights_fp32[pool_row]
                                : static_cast<float>(
                                      route_weights[pool_row]);
                            const cd_dtype_t grad_h_bf16 =
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
                                // Python evaluates 3.0 * beta in FP64 before
                                // converting the scalar to FP32. Multiplying
                                // the already-rounded kBeta by 3.0f is one ULP
                                // lower and changes BF16 ties in GeGLU dgate.
                                constexpr float kThreeBeta = 0.134145f;
                                const float gate_sq =
                                    __fmul_rn(gate, gate);
                                z = __fmul_rn(
                                    __fmul_rn(kAlpha, gate),
                                    __fadd_rn(
                                        1.0f,
                                        __fmul_rn(
                                            kBeta, gate_sq)));
                                dz_dgate = __fmul_rn(
                                    kAlpha,
                                    __fadd_rn(
                                        1.0f,
                                        __fmul_rn(
                                            kThreeBeta,
                                            gate_sq)));
                            } else {
                                z = gate;
                                dz_dgate = 1.0f;
                            }
                            const float neg_exp =
                                !kBF16Mode || kFastMath
                                ? __expf(-z)
                                : expf(-z);
                            const float denom =
                                __fadd_rn(1.0f, neg_exp);
                            const float sig =
                                1.0f / denom;
                            cd_dtype_t h_act_bf16;
                            cd_dtype_t grad_gate_bf16;
                            cd_dtype_t grad_up_bf16;
                            if constexpr (
                                kBF16Mode &&
                                kActivationType ==
                                    ActivationType::SwiGLU) {
                                if (!has_activation_clamp) {
                                    // Native grouped experts materialize
                                    // BF16 SiLU, BF16 SiLU*up, and BF16
                                    // grad_h*up before aten::silu_backward.
                                    const cd_dtype_t silu_bf16 =
                                        cd_dtype_t(
                                            gate / denom);
                                    h_act_bf16 =
                                        cd_dtype_t(
                                            __fmul_rn(
                                                static_cast<
                                                    float>(
                                                    silu_bf16),
                                                up));
                                    const cd_dtype_t
                                        grad_silu_bf16 =
                                            cd_dtype_t(
                                                __fmul_rn(
                                                    grad_h,
                                                    up));
                                    const float
                                        one_minus_sig =
                                            __fsub_rn(
                                                1.0f, sig);
                                    const float
                                        silu_inner =
                                            __fadd_rn(
                                                1.0f,
                                                __fmul_rn(
                                                    gate,
                                                    one_minus_sig));
                                    const float
                                        grad_silu_sig =
                                            __fmul_rn(
                                                static_cast<
                                                    float>(
                                                    grad_silu_bf16),
                                                sig);
                                    grad_gate_bf16 =
                                        cd_dtype_t(
                                            __fmul_rn(
                                                grad_silu_sig,
                                                silu_inner));
                                    grad_up_bf16 =
                                        cd_dtype_t(
                                            __fmul_rn(
                                                grad_h,
                                                static_cast<
                                                    float>(
                                                    silu_bf16)));
                                } else {
                                    const float
                                        activated_gate =
                                            __fmul_rn(
                                                gate, sig);
                                    h_act_bf16 =
                                        cd_dtype_t(
                                            __fmul_rn(
                                                activated_gate,
                                                up));
                                    const float
                                        one_minus_sig =
                                            __fsub_rn(
                                                1.0f, sig);
                                    const float gate_sig =
                                        __fmul_rn(
                                            gate, sig);
                                    const float
                                        activation_grad =
                                            __fadd_rn(
                                                sig,
                                                __fmul_rn(
                                                    __fmul_rn(
                                                        gate_sig,
                                                        one_minus_sig),
                                                    dz_dgate));
                                    grad_gate_bf16 =
                                        cd_dtype_t(
                                            gate_in_range
                                            ? __fmul_rn(
                                                  __fmul_rn(
                                                      grad_h,
                                                      up),
                                                  activation_grad)
                                            : 0.0f);
                                    grad_up_bf16 =
                                        cd_dtype_t(
                                            up_in_range
                                            ? __fmul_rn(
                                                  grad_h,
                                                  activated_gate)
                                            : 0.0f);
                                }
                            } else {
                                const float activated_gate =
                                    __fmul_rn(gate, sig);
                                h_act_bf16 =
                                    cd_dtype_t(
                                        __fmul_rn(
                                            activated_gate, up));
                                const float one_minus_sig =
                                    __fsub_rn(1.0f, sig);
                                const float gate_sig =
                                    __fmul_rn(gate, sig);
                                const float activation_grad =
                                    __fadd_rn(
                                        sig,
                                        __fmul_rn(
                                            __fmul_rn(
                                                gate_sig,
                                                one_minus_sig),
                                            dz_dgate));
                                grad_gate_bf16 =
                                    cd_dtype_t(
                                        gate_in_range
                                        ? __fmul_rn(
                                              __fmul_rn(
                                                  grad_h,
                                                  up),
                                              activation_grad)
                                        : 0.0f);
                                grad_up_bf16 =
                                    cd_dtype_t(
                                        up_in_range
                                        ? __fmul_rn(
                                              grad_h,
                                              activated_gate)
                                        : 0.0f);
                            }
                            h_act_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kIntermediateHidden +
                                hidden_col] =
                                h_act_bf16;
                            // PRE_DOWN may phase-alias h_act and h_weighted.
                            // Preserve unweighted h until its route-gradient
                            // reduction, then overwrite it in a later phase.
                            if (!(
                                    kBF16Mode &&
                                    kRouteWeightMode ==
                                        RouteWeightMode::PreDown &&
                                    h_act_output ==
                                        h_weighted_output)) {
                                h_weighted_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kIntermediateHidden +
                                    hidden_col] =
                                    kRouteWeightMode ==
                                            RouteWeightMode::PostDown
                                    ? h_act_bf16
                                    : cd_dtype_t(
                                          static_cast<float>(
                                              h_act_bf16) *
                                          route_weight);
                            }
                            grad_gate_up_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    (2 *
                                     kIntermediateHidden) +
                                hidden_col] =
                                grad_gate_bf16;
                            grad_gate_up_output[
                                static_cast<uint64_t>(
                                    pool_row) *
                                    (2 *
                                     kIntermediateHidden) +
                                kIntermediateHidden +
                                hidden_col] =
                                grad_up_bf16;
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
            full_grid_phase_barrier(12);

            if constexpr (kBF16Mode) {
                // In phase-ordered mode these outputs may still contain the
                // forward gate values or reverse-dispatched grad-y in every
                // row that the active activation tiles did not visit. Clear
                // per-expert block padding and the unused capacity tail only
                // after all active gate reads have completed.
                uint32_t padding_pool_block_offset = 0;
                #pragma unroll
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(
                                expert_counts +
                                expert_idx));
                    const uint32_t num_blocks =
                        math::ceil_div(
                            num_tokens, BLOCK_M);
                    const uint32_t num_padded_tokens =
                        num_blocks * BLOCK_M;
                    const uint32_t padding_rows =
                        num_padded_tokens - num_tokens;
                    for (uint64_t linear =
                             static_cast<uint64_t>(
                                 blockIdx.x) *
                                 kNumThreads +
                             threadIdx.x;
                         linear <
                             static_cast<uint64_t>(
                                 padding_rows) *
                                 (2 *
                                  kIntermediateHidden);
                         linear +=
                             static_cast<uint64_t>(
                                 kNumSMs) *
                                 kNumThreads) {
                        const uint32_t row =
                            linear /
                            (2 * kIntermediateHidden);
                        const uint32_t col =
                            linear -
                            static_cast<uint64_t>(row) *
                                (2 *
                                 kIntermediateHidden);
                        const uint32_t pool_row =
                            padding_pool_block_offset *
                                BLOCK_M +
                            num_tokens + row;
                        grad_gate_up_output[
                            static_cast<uint64_t>(
                                pool_row) *
                                (2 *
                                 kIntermediateHidden) +
                            col] =
                            cd_dtype_t(0.0f);
                    }
                    for (uint64_t linear =
                             static_cast<uint64_t>(
                                 blockIdx.x) *
                                 kNumThreads +
                             threadIdx.x;
                         linear <
                             static_cast<uint64_t>(
                                 padding_rows) *
                                 kIntermediateHidden;
                         linear +=
                             static_cast<uint64_t>(
                                 kNumSMs) *
                                 kNumThreads) {
                        const uint32_t row =
                            linear /
                            kIntermediateHidden;
                        const uint32_t col =
                            linear -
                            static_cast<uint64_t>(row) *
                                kIntermediateHidden;
                        const uint32_t pool_row =
                            padding_pool_block_offset *
                                BLOCK_M +
                            num_tokens + row;
                        h_weighted_output[
                            static_cast<uint64_t>(
                                pool_row) *
                                kIntermediateHidden +
                            col] =
                            cd_dtype_t(0.0f);
                    }
                    padding_pool_block_offset +=
                        num_blocks;
                }
                const uint32_t capacity_tail_start =
                    padding_pool_block_offset * BLOCK_M;
                const uint32_t capacity_tail_rows =
                    num_pool_rows - capacity_tail_start;
                for (uint64_t linear =
                         static_cast<uint64_t>(
                             blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear <
                         static_cast<uint64_t>(
                             capacity_tail_rows) *
                             (2 *
                              kIntermediateHidden);
                     linear +=
                         static_cast<uint64_t>(
                             kNumSMs) *
                             kNumThreads) {
                    grad_gate_up_output[
                        static_cast<uint64_t>(
                            capacity_tail_start) *
                            (2 *
                             kIntermediateHidden) +
                        linear] =
                        cd_dtype_t(0.0f);
                }
                for (uint64_t linear =
                         static_cast<uint64_t>(
                             blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     linear <
                         static_cast<uint64_t>(
                             capacity_tail_rows) *
                             kIntermediateHidden;
                     linear +=
                         static_cast<uint64_t>(
                             kNumSMs) *
                             kNumThreads) {
                    h_weighted_output[
                        static_cast<uint64_t>(
                            capacity_tail_start) *
                            kIntermediateHidden +
                        linear] =
                        cd_dtype_t(0.0f);
                }
                full_grid_phase_barrier(13);
            }

            if constexpr (
                kComputeRouteGrad && !kInputsPrepared) {
                // The activation epilogue spans multiple N-tile CTAs. Reduce
                // each route term only after all tiles are visible so the
                // router gradient has a fixed FP32 summation order instead of
                // depending on cross-CTA atomic arrival order.
                constexpr uint32_t kRouteColumns =
                    kRouteWeightMode ==
                            RouteWeightMode::PostDown
                        ? kHidden
                        : kIntermediateHidden;
                // FireTitan's POST_DOWN path uses Triton's tl.sum with a
                // power-of-two BLOCK_H and BLOCK_H / 256 warps (clamped to
                // [4, 32]). For the production hidden sizes this gives 2, 4,
                // or 8 elements per thread. Preserve that exact logical
                // layout; changing the columns assigned to a lane changes the
                // FP32 reduction result.
                constexpr uint32_t kTritonRouteBlockH = [] {
                    uint32_t value = 1;
                    while (value < kRouteColumns && value < 8192)
                        value <<= 1;
                    return value;
                }();
                constexpr uint32_t kTritonRouteNumWarps = [] {
                    uint32_t value =
                        kTritonRouteBlockH / 256;
                    value = value < 4 ? 4 : value;
                    return value > 32 ? 32 : value;
                }();
                constexpr uint32_t kTritonRouteThreads =
                    kTritonRouteNumWarps * 32;
                constexpr uint32_t
                    kTritonRouteValuesPerThread =
                        kTritonRouteBlockH /
                        kTritonRouteThreads;
                DG_STATIC_ASSERT(
                    kTritonRouteValuesPerThread == 2 ||
                        kTritonRouteValuesPerThread == 4 ||
                        kTritonRouteValuesPerThread == 8,
                    "Unsupported Triton route reduction width");
                constexpr uint32_t kRouteInputPow2 = [] {
                    uint32_t value = 1;
                    const uint32_t vectorized_columns =
                        kRouteColumns / 4;
                    while (value < 512 &&
                           (value << 1) <=
                               vectorized_columns)
                        value <<= 1;
                    return value;
                }();
                auto* route_lane_sums =
                    reinterpret_cast<float*>(smem_gemm_base);
                auto* route_control =
                    reinterpret_cast<uint32_t*>(smem_gemm_base);
                if (threadIdx.x == 0) {
                    uint32_t total_route_rows = 0;
                    #pragma unroll
                    for (uint32_t expert_idx = 0;
                         expert_idx < kNumExperts;
                         ++expert_idx) {
                        total_route_rows +=
                            static_cast<uint32_t>(
                                __ldg(
                                    expert_counts +
                                    expert_idx));
                    }
                    route_control[0] =
                        total_route_rows;
                }
                __syncthreads();
                const uint32_t total_route_rows =
                    route_control[0];
                const uint32_t route_output_pow2 =
                    total_route_rows > 0
                    ? 1u << (31 - __clz(
                                 total_route_rows))
                    : 1u;
                constexpr uint32_t
                    kInitialRouteGroupThreads =
                        cute::min(
                            kRouteInputPow2, 32u);
                const uint32_t route_block_height =
                    cute::min(
                        route_output_pow2,
                        512u /
                            kInitialRouteGroupThreads);
                const uint32_t route_group_threads =
                    cute::min(
                        kRouteInputPow2,
                        512u / route_block_height);
                const uint32_t num_route_groups_per_cta =
                    kNumThreads / route_group_threads;
                const uint32_t route_group_idx =
                    threadIdx.x / route_group_threads;
                const uint32_t route_group_lane_idx =
                    threadIdx.x &
                    (route_group_threads - 1);
                const uint32_t global_route_group =
                    blockIdx.x *
                        num_route_groups_per_cta +
                    route_group_idx;
                const uint32_t num_route_groups =
                    kNumSMs *
                    num_route_groups_per_cta;
                uint32_t route_pool_block_offset = 0;
                #pragma unroll
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(expert_counts + expert_idx));
                    for (uint32_t token_idx = global_route_group;
                         token_idx < num_tokens;
                         token_idx += num_route_groups) {
                        const uint32_t pool_row =
                            route_pool_block_offset * BLOCK_M +
                            token_idx;
                        float grad_route = 0.0f;
                        if constexpr (false) {
                            float grad_y[
                                kTritonRouteValuesPerThread];
                            float down[
                                kTritonRouteValuesPerThread];
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i <
                                     kTritonRouteValuesPerThread;
                                 ++i) {
                                const uint32_t col =
                                    route_group_lane_idx +
                                    i * kTritonRouteThreads;
                                grad_y[i] =
                                    col < kHidden
                                    ? static_cast<float>(
                                          grad_y_unweighted_output[
                                              static_cast<uint64_t>(
                                                  pool_row) *
                                                  kHidden +
                                              col])
                                    : 0.0f;
                                down[i] =
                                    col < kHidden
                                    ? static_cast<float>(
                                          down_unweighted_output[
                                              static_cast<uint64_t>(
                                                  pool_row) *
                                                  kHidden +
                                              col])
                                    : 0.0f;
                            }

                            if constexpr (
                                kTritonRouteValuesPerThread ==
                                2) {
                                grad_route = __fmaf_rn(
                                    grad_y[0], down[0],
                                    __fmul_rn(
                                        grad_y[1], down[1]));
                            } else if constexpr (
                                kTritonRouteValuesPerThread ==
                                4) {
                                const float even =
                                    __fmaf_rn(
                                        grad_y[0], down[0],
                                        __fmul_rn(
                                            grad_y[2],
                                            down[2]));
                                const float odd =
                                    __fmaf_rn(
                                        grad_y[1], down[1],
                                        __fmul_rn(
                                            grad_y[3],
                                            down[3]));
                                grad_route =
                                    __fadd_rn(even, odd);
                            } else {
                                const float pair_02 =
                                    __fmaf_rn(
                                        grad_y[0], down[0],
                                        __fmul_rn(
                                            grad_y[2],
                                            down[2]));
                                const float pair_13 =
                                    __fmaf_rn(
                                        grad_y[1], down[1],
                                        __fmul_rn(
                                            grad_y[3],
                                            down[3]));
                                const float pair_46 =
                                    __fmaf_rn(
                                        grad_y[4], down[4],
                                        __fmul_rn(
                                            grad_y[6],
                                            down[6]));
                                const float pair_57 =
                                    __fmaf_rn(
                                        grad_y[5], down[5],
                                        __fmul_rn(
                                            grad_y[7],
                                            down[7]));
                                grad_route = __fadd_rn(
                                    __fadd_rn(
                                        pair_02, pair_46),
                                    __fadd_rn(
                                        pair_13, pair_57));
                            }

                            // Triton first performs a butterfly reduction
                            // within each physical warp.
                            #pragma unroll
                            for (uint32_t offset = 16;
                                 offset > 0;
                                 offset >>= 1) {
                                grad_route = __fadd_rn(
                                    grad_route,
                                    __shfl_xor_sync(
                                        0xffffffff,
                                        grad_route,
                                        offset));
                            }

                            const uint32_t warp_in_group =
                                route_group_lane_idx / 32;
                            const uint32_t lane_in_warp =
                                route_group_lane_idx & 31;
                            if (lane_in_warp == 0) {
                                route_lane_sums[
                                    route_group_idx *
                                        kTritonRouteNumWarps +
                                    warp_in_group] =
                                    grad_route;
                            }
                            ptx::sync_aligned(
                                kTritonRouteThreads,
                                route_group_idx);

                            // Triton loads the power-of-two set of warp
                            // partials into the first warp and reduces it with
                            // the same butterfly tree.
                            if (warp_in_group == 0) {
                                grad_route =
                                    route_lane_sums[
                                        route_group_idx *
                                            kTritonRouteNumWarps +
                                        (lane_in_warp &
                                         (kTritonRouteNumWarps -
                                          1))];
                                #pragma unroll
                                for (uint32_t offset =
                                         kTritonRouteNumWarps /
                                         2;
                                     offset > 0;
                                     offset >>= 1) {
                                    grad_route = __fadd_rn(
                                        grad_route,
                                        __shfl_xor_sync(
                                            0xffffffff,
                                            grad_route,
                                            offset));
                                }
                            }
                        } else {
                            float lane_sums[4] = {
                                0.0f, 0.0f, 0.0f, 0.0f};
                            if constexpr (
                                kRouteWeightMode ==
                                RouteWeightMode::PostDown) {
                                for (uint32_t col_base =
                                         route_group_lane_idx *
                                         4;
                                     col_base < kHidden;
                                     col_base +=
                                         route_group_threads *
                                         4) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4;
                                         ++i) {
                                        const uint32_t col =
                                            col_base + i;
                                        const float grad_y =
                                            static_cast<float>(
                                                grad_y_unweighted_output[
                                                    static_cast<
                                                        uint64_t>(
                                                        pool_row) *
                                                        kHidden +
                                                    col]);
                                        const float down =
                                            static_cast<float>(
                                                down_unweighted_output[
                                                    static_cast<
                                                        uint64_t>(
                                                        pool_row) *
                                                        kHidden +
                                                    col]);
                                        lane_sums[i] =
                                            __fadd_rn(
                                                lane_sums[i],
                                                __fmul_rn(
                                                    grad_y,
                                                    down));
                                    }
                                }
                            } else {
                                for (uint32_t col_base =
                                         route_group_lane_idx *
                                         4;
                                     col_base <
                                         kIntermediateHidden;
                                     col_base +=
                                         route_group_threads *
                                         4) {
                                    #pragma unroll
                                    for (uint32_t i = 0; i < 4;
                                         ++i) {
                                        const uint32_t col =
                                            col_base + i;
                                        const float grad_h =
                                            static_cast<float>(
                                                grad_h_output[
                                                    static_cast<
                                                        uint64_t>(
                                                        pool_row) *
                                                        kIntermediateHidden +
                                                    col]);
                                        const float h_act =
                                            static_cast<float>(
                                                h_act_output[
                                                    static_cast<
                                                        uint64_t>(
                                                        pool_row) *
                                                        kIntermediateHidden +
                                                    col]);
                                        lane_sums[i] =
                                            __fadd_rn(
                                                lane_sums[i],
                                                __fmul_rn(
                                                    grad_h,
                                                    h_act));
                                    }
                                }
                            }
                            grad_route = __fadd_rn(
                                __fadd_rn(
                                    lane_sums[0],
                                    lane_sums[1]),
                                lane_sums[2]);
                            grad_route = __fadd_rn(
                                grad_route, lane_sums[3]);

                            route_lane_sums[threadIdx.x] =
                                grad_route;
                            if (route_group_threads > 32) {
                                for (uint32_t offset =
                                         route_group_threads /
                                         2;
                                     offset >= 32;
                                     offset >>= 1) {
                                    ptx::sync_aligned(
                                        route_group_threads,
                                        route_group_idx);
                                    if (route_group_lane_idx <
                                        offset) {
                                        grad_route =
                                            __fadd_rn(
                                                grad_route,
                                                route_lane_sums[
                                                    threadIdx.x +
                                                    offset]);
                                        route_lane_sums[
                                            threadIdx.x] =
                                            grad_route;
                                    }
                                }
                            }
                            if (route_group_lane_idx < 32) {
                                #pragma unroll
                                for (uint32_t offset = 16;
                                     offset > 0;
                                     offset >>= 1) {
                                    grad_route =
                                        __fadd_rn(
                                            grad_route,
                                            __shfl_down_sync(
                                                0xffffffff,
                                                grad_route,
                                                offset));
                                }
                            }
                        }
                        if (route_group_lane_idx == 0) {
                            grad_route_output[pool_row] =
                                grad_route;
                            if (backward_grad_route != nullptr) {
                                const auto metadata =
                                    token_src_metadata[pool_row];
                                auto* remote_grad_route =
                                    backward_sym_buffer.map(
                                        backward_grad_route +
                                            static_cast<uint64_t>(
                                                metadata.token_idx) *
                                                num_topk +
                                            metadata.topk_idx,
                                        metadata.rank_idx);
                                *remote_grad_route = grad_route;
                            }
                        }
                        if (route_group_threads > 32) {
                            ptx::sync_aligned(
                                route_group_threads,
                                route_group_idx);
                        } else {
                            __syncwarp();
                        }
                    }
                    route_pool_block_offset +=
                        math::ceil_div(num_tokens, BLOCK_M);
                }
            }
            if constexpr (
                kBF16Mode &&
                kRouteWeightMode == RouteWeightMode::PreDown &&
                !kInputsPrepared) {
                if (h_act_output == h_weighted_output) {
                    // Every route reduction must consume unweighted h before
                    // the shared storage becomes the W2-wgrad input.
                    full_grid_phase_barrier(14);
                    uint32_t pool_block_offset = 0;
                    #pragma unroll
                    for (uint32_t expert_idx = 0;
                         expert_idx < kNumExperts; ++expert_idx) {
                        const uint32_t num_tokens =
                            static_cast<uint32_t>(
                                __ldg(
                                    expert_counts +
                                    expert_idx));
                        for (uint64_t linear =
                                 static_cast<uint64_t>(
                                     blockIdx.x) *
                                     kNumThreads +
                                 threadIdx.x;
                             linear <
                                 static_cast<uint64_t>(
                                     num_tokens) *
                                     kIntermediateHidden;
                             linear +=
                                 static_cast<uint64_t>(
                                     kNumSMs) *
                                 kNumThreads) {
                            const uint32_t token_idx =
                                linear /
                                kIntermediateHidden;
                            const uint32_t col =
                                linear -
                                static_cast<uint64_t>(
                                    token_idx) *
                                    kIntermediateHidden;
                            const uint32_t pool_row =
                                pool_block_offset * BLOCK_M +
                                token_idx;
                            const uint64_t offset =
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kIntermediateHidden +
                                col;
                            h_weighted_output[offset] =
                                cd_dtype_t(
                                    static_cast<float>(
                                        h_act_output[offset]) *
                                    route_weights_fp32[
                                        pool_row]);
                        }
                        pool_block_offset +=
                            math::ceil_div(
                                num_tokens, BLOCK_M);
                    }
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

            trace_begin(15);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(15);
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
            trace_begin(16);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(16);
            trace_begin(21);

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
                        #pragma unroll
                        for (uint32_t split_idx = 0;
                             split_idx <
                                 kNumW13DgradSplits;
                             ++split_idx) {
                            #pragma unroll 1
                            for (uint32_t k_block_idx = 0;
                                 k_block_idx <
                                     (2 *
                                      kIntermediateHidden) /
                                         (DGRAD_BLOCK_K *
                                          kNumW13DgradSplits);
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
                                        split_idx *
                                                ((2 *
                                                  kIntermediateHidden) /
                                                 kNumW13DgradSplits) +
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
                        #pragma unroll
                        for (uint32_t split_idx = 0;
                             split_idx <
                                 kNumW13DgradSplits;
                             ++split_idx) {
                            #pragma unroll 1
                            for (uint32_t k_block_idx = 0;
                                 k_block_idx <
                                     (2 *
                                      kIntermediateHidden) /
                                         (DGRAD_BLOCK_K *
                                          kNumW13DgradSplits);
                                 advance_pipeline(
                                     k_block_idx)) {
                                const uint32_t
                                    global_k_block_idx =
                                        split_idx *
                                            ((2 *
                                              kIntermediateHidden) /
                                             (DGRAD_BLOCK_K *
                                              kNumW13DgradSplits)) +
                                        k_block_idx;
                                const uint32_t
                                    weight_tile_idx =
                                        (expert_idx *
                                             ((2 *
                                               kIntermediateHidden) /
                                              DGRAD_BLOCK_K) +
                                         global_k_block_idx) *
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
                                            global_k_block_idx *
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
                            #pragma unroll
                            for (uint32_t split_idx = 0;
                                 split_idx <
                                     kNumW13DgradSplits;
                                 ++split_idx) {
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
                                             (DGRAD_BLOCK_K *
                                              kNumW13DgradSplits);
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
                                                (DGRAD_BLOCK_K *
                                                 kNumW13DgradSplits) -
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
                            (current_iter /
                             kNumEpilogueStages) &
                            1;
                        current_iter +=
                            kNumW13DgradSplits;
                        tmem_full_barriers[accum_stage]->wait(
                            accum_phase);
                        if constexpr (kBF16Mode)
                            tmem_full_barriers[
                                accum_stage ^ 1]
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
                                    const uint32_t tmem_addr =
                                        accum_stage * UMMA_N +
                                        s * STORE_BLOCK_M +
                                        i * 8;
                                    uint32_t w1_values[8];
                                    uint32_t w3_values[8];
                                    cute::
                                        SM100_TMEM_LOAD_16dp256b1x::
                                            copy(
                                                tmem_addr,
                                                w1_values[0],
                                                w1_values[1],
                                                w1_values[2],
                                                w1_values[3]);
                                    cute::
                                        SM100_TMEM_LOAD_16dp256b1x::
                                            copy(
                                                tmem_addr |
                                                    0x00100000,
                                                w1_values[4],
                                                w1_values[5],
                                                w1_values[6],
                                                w1_values[7]);
                                    if constexpr (kBF16Mode) {
                                        const uint32_t
                                            w3_tmem_addr =
                                                (accum_stage ^
                                                 1) *
                                                    UMMA_N +
                                                s *
                                                    STORE_BLOCK_M +
                                                i * 8;
                                        cute::
                                            SM100_TMEM_LOAD_16dp256b1x::
                                                copy(
                                                    w3_tmem_addr,
                                                    w3_values[0],
                                                    w3_values[1],
                                                    w3_values[2],
                                                    w3_values[3]);
                                        cute::
                                            SM100_TMEM_LOAD_16dp256b1x::
                                                copy(
                                                    w3_tmem_addr |
                                                        0x00100000,
                                                    w3_values[4],
                                                    w3_values[5],
                                                    w3_values[6],
                                                    w3_values[7]);
                                    }
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
                                    const auto add_bf16_pair =
                                        [](uint32_t a,
                                           uint32_t b,
                                           uint32_t c,
                                           uint32_t d) {
                                            const uint32_t
                                                w1_packed =
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            a,
                                                            b);
                                            const uint32_t
                                                w3_packed =
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            c,
                                                            d);
                                            const auto w1 =
                                                *reinterpret_cast<
                                                    const nv_bfloat162*>(
                                                    &w1_packed);
                                            const auto w3 =
                                                *reinterpret_cast<
                                                    const nv_bfloat162*>(
                                                    &w3_packed);
                                            const auto sum =
                                                __hadd2_rn(
                                                    w1, w3);
                                            return *reinterpret_cast<
                                                const uint32_t*>(
                                                &sum);
                                        };
                                    if constexpr (kBF16Mode) {
                                        ptx::
                                            SM90_U32x4_STSM_T<int>::
                                                copy(
                                                    add_bf16_pair(
                                                        w1_values[0],
                                                        w1_values[1],
                                                        w3_values[0],
                                                        w3_values[1]),
                                                    add_bf16_pair(
                                                        w1_values[2],
                                                        w1_values[3],
                                                        w3_values[2],
                                                        w3_values[3]),
                                                    add_bf16_pair(
                                                        w1_values[4],
                                                        w1_values[5],
                                                        w3_values[4],
                                                        w3_values[5]),
                                                    add_bf16_pair(
                                                        w1_values[6],
                                                        w1_values[7],
                                                        w3_values[6],
                                                        w3_values[7]),
                                                    smem_ptr);
                                    } else {
                                        ptx::
                                            SM90_U32x4_STSM_T<int>::
                                                copy(
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            w1_values[0],
                                                            w1_values[1]),
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            w1_values[2],
                                                            w1_values[3]),
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            w1_values[4],
                                                            w1_values[5]),
                                                    math::
                                                        cast_into_bf16_and_pack(
                                                            w1_values[6],
                                                            w1_values[7]),
                                                    smem_ptr);
                                    }
                                }
                            }
                            cutlass::arch::
                                NamedBarrier::sync(
                                    kNumDgradEpilogueThreads,
                                    0);

                            if constexpr (
                                kWideGradXStore) {
                                DG_STATIC_ASSERT(
                                    BLOCK_N % 8 == 0,
                                    "Wide grad-x stores require eight-column alignment");
                                DG_STATIC_ASSERT(
                                    kHidden % 8 == 0,
                                    "Wide grad-x stores require aligned output rows");
                                #pragma unroll
                                for (uint32_t linear =
                                         epilogue_thread_idx;
                                     linear <
                                         STORE_BLOCK_M *
                                             (BLOCK_N / 8);
                                     linear +=
                                         kNumDgradEpilogueThreads) {
                                    const uint32_t row =
                                        linear /
                                        (BLOCK_N / 8);
                                    const uint32_t n =
                                        (linear -
                                         row *
                                             (BLOCK_N / 8)) *
                                        8;
                                    const uint32_t local_m =
                                        s * STORE_BLOCK_M +
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
                                            row_in_atom * 128 +
                                            ((n_in_atom >> 3) ^
                                             row_in_atom) *
                                                16;
                                    const auto packed =
                                        *reinterpret_cast<
                                            const uint4*>(
                                            reinterpret_cast<
                                                const uint8_t*>(
                                                smem_cd[0]) +
                                            smem_byte_offset);
                                    const uint32_t pool_row =
                                        (pool_block_offset +
                                         m_block_idx) *
                                            BLOCK_M +
                                        local_m;
                                    const uint32_t out_col =
                                        n_block_idx *
                                            BLOCK_N +
                                        n;
                                    if constexpr (
                                        kWriteGradXPool) {
                                        *reinterpret_cast<
                                            uint4*>(
                                            grad_x_pool_output +
                                            static_cast<
                                                uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            out_col) = packed;
                                    }
                                    if constexpr (
                                        kDirectRemoteGradX) {
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
                                              metadata
                                                  .token_idx) *
                                                 kHidden +
                                             out_col);
                                        *reinterpret_cast<
                                            uint4*>(
                                            backward_sym_buffer
                                                .map(
                                                    dst,
                                                    metadata
                                                        .rank_idx)) =
                                            packed;
                                    }
                                }
                            } else if constexpr (
                                kVectorizedGradXStore) {
                                #pragma unroll
                                for (uint32_t linear =
                                         epilogue_thread_idx;
                                     linear <
                                         STORE_BLOCK_M *
                                             (BLOCK_N / 2);
                                     linear +=
                                         kNumDgradEpilogueThreads) {
                                    const uint32_t row =
                                        linear /
                                        (BLOCK_N / 2);
                                    const uint32_t n =
                                        (linear -
                                         row *
                                             (BLOCK_N / 2)) *
                                        2;
                                    const uint32_t local_m =
                                        s * STORE_BLOCK_M +
                                        row;
                                    if (local_m >= valid_m)
                                        continue;
                                    const uint32_t
                                        row_in_atom =
                                            row & 7;
                                    const auto load_bf16_bits =
                                        [&](const uint32_t
                                                element_n) {
                                            const uint32_t
                                                n_atom =
                                                    element_n /
                                                    64;
                                            const uint32_t
                                                n_in_atom =
                                                    element_n -
                                                    n_atom *
                                                        64;
                                            const uint32_t
                                                smem_byte_offset =
                                                    n_atom *
                                                        STORE_BLOCK_M *
                                                        128 +
                                                    (row >> 3) *
                                                        8 *
                                                        128 +
                                                    row_in_atom *
                                                        128 +
                                                    ((n_in_atom >>
                                                      3) ^
                                                     row_in_atom) *
                                                        16 +
                                                    (n_in_atom &
                                                     7) *
                                                        sizeof(
                                                            cd_dtype_t);
                                            return *reinterpret_cast<
                                                uint16_t*>(
                                                reinterpret_cast<
                                                    uint8_t*>(
                                                    smem_cd[0]) +
                                                smem_byte_offset);
                                        };
                                    const uint32_t packed =
                                        static_cast<uint32_t>(
                                            load_bf16_bits(n)) |
                                        (static_cast<uint32_t>(
                                             load_bf16_bits(
                                                 n + 1))
                                         << 16);
                                    const uint32_t pool_row =
                                        (pool_block_offset +
                                         m_block_idx) *
                                            BLOCK_M +
                                        local_m;
                                    const uint32_t out_col =
                                        n_block_idx *
                                            BLOCK_N +
                                        n;
                                    if constexpr (
                                        kWriteGradXPool) {
                                        *reinterpret_cast<
                                            uint32_t*>(
                                            grad_x_pool_output +
                                            static_cast<
                                                uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            out_col) = packed;
                                    }
                                    if constexpr (
                                        kDirectRemoteGradX) {
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
                                              metadata
                                                  .token_idx) *
                                                 kHidden +
                                             out_col);
                                        *reinterpret_cast<
                                            uint32_t*>(
                                            backward_sym_buffer
                                                .map(
                                                    dst,
                                                    metadata
                                                        .rank_idx)) =
                                            packed;
                                    }
                                }
                            } else {
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
                                    if constexpr (
                                        kWriteGradXPool) {
                                        grad_x_pool_output[
                                            static_cast<
                                                uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            out_col] = value;
                                    }
                                    if constexpr (
                                        kDirectRemoteGradX) {
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
                                              metadata
                                                  .token_idx) *
                                                 kHidden +
                                             out_col);
                                        *backward_sym_buffer
                                             .map(
                                                 dst,
                                                 metadata
                                                     .rank_idx) =
                                            value;
                                    }
                                }
                            }
                        }
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage]
                            ->arrive(0u);
                        if constexpr (kBF16Mode)
                            tmem_empty_barriers[
                                accum_stage ^ 1]
                                ->arrive(0u);
                    });
            }
        }

        trace_end(21);
        if constexpr (kNumRanks > 1) {
            if constexpr (
                kDirectRemoteGradX || kComputeRouteGrad) {
                // Publish every direct grad-x and route-gradient NVLink store
                // before any destination rank consumes its source planes.
                constexpr uint32_t
                    kDirectGradXDoneGridSyncIndex = 1;
                constexpr uint32_t
                    kDirectGradXDoneBarrierTag = 9;
                if constexpr (kTraceKernel) {
                    // Decompose the otherwise identical NVLink barrier so the
                    // trace distinguishes local compute/grid skew from the
                    // cross-rank signal and its publication grid sync.
                    trace_begin(17);
                    comm::grid_sync<
                        kNumSMs,
                        kDirectGradXDoneGridSyncIndex>(
                        backward_workspace,
                        blockIdx.x,
                        threadIdx.x,
                        []() { __syncthreads(); });
                    trace_end(17);

                    if (blockIdx.x == 0)
                        trace_begin(18);
                    comm::nvlink_barrier<
                        kNumRanks, kNumSMs, kNumThreads,
                        kDirectGradXDoneGridSyncIndex,
                        kDirectGradXDoneBarrierTag>(
                        backward_workspace,
                        backward_sym_buffer,
                        blockIdx.x,
                        threadIdx.x,
                        []() { __syncthreads(); },
                        false, false);
                    if (blockIdx.x == 0)
                        trace_end(18);

                    trace_begin(19);
                    comm::grid_sync<
                        kNumSMs,
                        kDirectGradXDoneGridSyncIndex>(
                        backward_workspace,
                        blockIdx.x,
                        threadIdx.x,
                        []() { __syncthreads(); });
                    trace_end(19);
                } else {
                    comm::nvlink_barrier<
                        kNumRanks, kNumSMs, kNumThreads,
                        kDirectGradXDoneGridSyncIndex,
                        kDirectGradXDoneBarrierTag>(
                        backward_workspace,
                        backward_sym_buffer,
                        blockIdx.x,
                        threadIdx.x,
                        []() { __syncthreads(); });
                }
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
        if constexpr (kClearWgradPadding)
            clear_wgrad_padding_rows();

        __syncthreads();
        trace_begin(20);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(20);
        trace_end(0);
        if (warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
    }
#endif
}

// --------------------------------------------------------------------------
// Fixed SM103 E4M3-block128 training reverse used by GLM-5.2.
//
// This is deliberately AOT-only.  It keeps the upstream 2-CTA/TMEM structure
// but turns the reverse into one resident expert-wave loop.  Compact x and dy
// are power-of-two quantized into symmetric storage by this kernel, each
// expert is pulled into the private ring, and the three hardware-scaled GEMMs
// execute before that ring slot is reused.  Only the two compressed operands
// retained for the dedicated wgrad kernels span the full padded route pool.
// --------------------------------------------------------------------------

namespace sm103_block128_backward {

static constexpr uint32_t kHidden = 6144;
static constexpr uint32_t kIntermediate = 2048;
static constexpr uint32_t kGlobalExperts = 256;
static constexpr uint32_t kTopK = 8;
static constexpr uint32_t kBlockM = 192;
static constexpr uint32_t kBlockN = 128;
static constexpr uint32_t kBlockK = 128;
static constexpr uint32_t kSFBlockM = 256;
static constexpr uint32_t kSFBlockN = 128;
static constexpr uint32_t kStages = 6;
static constexpr uint32_t kThreads = 512;
static constexpr uint32_t kStoreBlockM = 32;
static constexpr uint32_t kEpilogueThreads = 256;
static constexpr uint32_t kNumEpilogueStages = 2;
static constexpr uint32_t kNumTMAStoreStages = 2;
static constexpr uint32_t kLoadBlockM = kBlockM / 2;
static constexpr uint32_t kLoadBlockN = kBlockN;
static constexpr uint32_t kUMMAM = 256;
static constexpr uint32_t kUMMAN = kBlockM;
static constexpr uint32_t kUMMAK = 32;
static constexpr uint32_t kSwizzle = 128;
static constexpr uint32_t kNumTmemAccumCols =
    kUMMAN * kNumEpilogueStages;
static constexpr uint32_t kNumTmemSFACols = kSFBlockM / 32;
static constexpr uint32_t kNumTmemSFBCols = kSFBlockN / 32;
static constexpr uint32_t kTmemSFAStart = kNumTmemAccumCols;
static constexpr uint32_t kTmemSFBStart =
    kNumTmemAccumCols + kNumTmemSFACols;
static constexpr uint32_t kNumTmemCols =
    utils::get_num_aligned_tmem_cols<
        kNumTmemAccumCols + kNumTmemSFACols + kNumTmemSFBCols>();

using fp8_t = cutlass::float_e4m3_t;
using bf16_t = cutlass::bfloat16_t;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

struct alignas(1024) SharedStorage {
    alignas(1024) bf16_t smem_cd[kNumTMAStoreStages]
                                      [kStoreBlockM * kBlockN];
    alignas(1024) fp8_t smem_a[kStages][kLoadBlockM * kBlockK];
    alignas(1024) fp8_t smem_b[kStages][kLoadBlockN * kBlockK];
    alignas(1024) uint32_t smem_sfa[kStages][kSFBlockM];
    alignas(1024) uint32_t smem_sfb[kStages][kSFBlockN];
    alignas(128) float reduce_values[4][8];
    Barrier full_barriers[kStages];
    Barrier empty_barriers[kStages];
    Barrier tmem_full_barriers[kNumEpilogueStages];
    Barrier tmem_empty_barriers[kNumEpilogueStages];
    uint32_t tmem_ptr;
};

DG_STATIC_ASSERT(kNumTmemCols <= 512, "SM103 backward exceeds TMEM");

CUTLASS_DEVICE float warp_reduce_max(float value) {
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        value = cute::max(
            value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

CUTLASS_DEVICE float warp_reduce_sum(float value) {
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(
            value, __shfl_down_sync(0xffffffff, value, offset));
    return value;
}

template <bool kMax>
CUTLASS_DEVICE float reduce_group_128(
    float value, SharedStorage& storage, const uint32_t group_idx) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_in_group = (threadIdx.x >> 5) & 3;
    value = kMax ? warp_reduce_max(value) : warp_reduce_sum(value);
    if (lane == 0)
        storage.reduce_values[group_idx][warp_in_group] = value;
    ptx::sync_aligned(128, group_idx);
    if (warp_in_group == 0) {
        value = lane < 4
                    ? storage.reduce_values[group_idx][lane]
                    : (kMax ? 0.0f : 0.0f);
        value = kMax ? warp_reduce_max(value) : warp_reduce_sum(value);
        if (lane == 0)
            storage.reduce_values[group_idx][4] = value;
    }
    ptx::sync_aligned(128, group_idx);
    return storage.reduce_values[group_idx][4];
}

CUTLASS_DEVICE uint32_t packed_power2_scale(
    const float amax, float& scale_inv) {
    const float raw = cute::max(amax * (1.0f / 448.0f), 0x1p-127f);
    const int exponent = math::fast_log2_ceil(raw);
    // UE8M0 code zero represents 2^-127.  It is an FP32 subnormal and cannot
    // be constructed by merely shifting an IEEE exponent field.
    const float scale = exponent == -127
        ? 0x1p-127f
        : math::fast_pow2(exponent);
    scale_inv = math::fast_pow2(-exponent);
    return ((*reinterpret_cast<const uint32_t*>(&scale)) >> 23) *
           0x01010101u;
}

CUTLASS_DEVICE uint32_t transform_sf_row(
    const uint32_t row) {
    const uint32_t in_block = row % kBlockM;
    return row / kBlockM * kSFBlockM +
           (in_block & ~127u) + (in_block & 31u) * 4 +
           ((in_block >> 5) & 3u);
}

template <sched::BackwardBlockPhase kPhase>
CUTLASS_DEVICE uint32_t phase_shape_n() {
    if constexpr (kPhase == sched::BackwardBlockPhase::RecomputeW13)
        return 2 * kIntermediate;
    if constexpr (kPhase == sched::BackwardBlockPhase::W2Dgrad)
        return kIntermediate;
    return kHidden;
}

template <sched::BackwardBlockPhase kPhase>
CUTLASS_DEVICE uint32_t phase_shape_k() {
    if constexpr (kPhase == sched::BackwardBlockPhase::W13Dgrad)
        return 2 * kIntermediate;
    return kHidden;
}

template <sched::BackwardBlockPhase kPhase, uint32_t kNumSMs>
CUTLASS_DEVICE void run_gemm_phase(
    SharedStorage& storage,
    const uint32_t local_expert_idx,
    const uint32_t num_tokens,
    const cute::TmaDescriptor& tensor_map_a,
    const cute::TmaDescriptor& tensor_map_sfa,
    const cute::TmaDescriptor& tensor_map_w13_recompute,
    const cute::TmaDescriptor& tensor_map_w2_dgrad,
    const cute::TmaDescriptor& tensor_map_w13_dgrad,
    const cute::TmaDescriptor& tensor_map_output,
    const float* w13_scales,
    const float* w2_scales,
    const uint32_t sf_ring_tokens) {
    constexpr uint32_t shape_n =
        kPhase == sched::BackwardBlockPhase::RecomputeW13
            ? 2 * kIntermediate
            : kPhase == sched::BackwardBlockPhase::W2Dgrad
                  ? kIntermediate
                  : kHidden;
    constexpr uint32_t shape_k =
        kPhase == sched::BackwardBlockPhase::W13Dgrad
            ? 2 * kIntermediate
            : kHidden;
    constexpr uint32_t num_block_ns = shape_n / kBlockN;
    constexpr uint32_t num_block_ks = shape_k / kBlockK;
    constexpr cute::UMMA::Major major_b =
        kPhase == sched::BackwardBlockPhase::RecomputeW13
            ? cute::UMMA::Major::K
            : cute::UMMA::Major::MN;

    const bool leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t num_m_blocks = math::ceil_div(num_tokens, kBlockM);

    comm::cluster_sync_with_relaxed_arrive();
    if (warp_idx == 4 && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kStages; ++i) {
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
    comm::cluster_sync_with_relaxed_arrive();

    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    const auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 4) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        for (uint32_t block = blockIdx.x;
             block < num_m_blocks * num_block_ns; block += kNumSMs) {
            const uint32_t m_block_idx = block / num_block_ns;
            const uint32_t valid_m = cute::min(
                num_tokens - m_block_idx * 192u, 192u);
            #pragma unroll 2
            for (uint32_t k_block_idx = 0; k_block_idx < num_block_ks;
                 advance_pipeline(k_block_idx)) {
                storage.empty_barriers[stage_idx].wait(phase ^ 1);
                uint32_t m_idx = m_block_idx * 192u;
                if (!leader_cta)
                    m_idx += math::align(valid_m, 16u) / 2;
                if (cute::elect_one_sync()) {
                    tma::copy<kBlockK, kLoadBlockM, kSwizzle, fp8_t>(
                        &tensor_map_a, &storage.full_barriers[stage_idx],
                        storage.smem_a[stage_idx],
                        k_block_idx * kBlockK, m_idx, 2);
                    tma::copy<kSFBlockM, 1, 0>(
                        &tensor_map_sfa, &storage.full_barriers[stage_idx],
                        storage.smem_sfa[stage_idx],
                        m_block_idx * kSFBlockM,
                        k_block_idx, 2);
                    if (leader_cta) {
                        storage.full_barriers[stage_idx].arrive_and_expect_tx(
                            sizeof(storage.smem_a[0]) * 2 +
                            sizeof(storage.smem_sfa[0]) * 2);
                    } else {
                        storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == 5) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        for (uint32_t block = blockIdx.x;
             block < num_m_blocks * num_block_ns; block += kNumSMs) {
            const uint32_t n_block_idx = block % num_block_ns;
            #pragma unroll 2
            for (uint32_t k_block_idx = 0; k_block_idx < num_block_ks;
                 advance_pipeline(k_block_idx)) {
                storage.empty_barriers[stage_idx].wait(phase ^ 1);
                if constexpr (
                    kPhase == sched::BackwardBlockPhase::RecomputeW13) {
                    if (cute::elect_one_sync()) {
                        constexpr uint32_t gran = 8;
                        constexpr uint32_t logical_rows = kBlockN / 2;
                        #pragma unroll
                        for (uint32_t group = 0;
                             group < logical_rows / gran; ++group) {
                            const uint32_t logical_row =
                                n_block_idx * logical_rows + group * gran;
                            const uint32_t up_row =
                                (local_expert_idx * 2 + 1) * kIntermediate +
                                logical_row;
                            const uint32_t gate_row =
                                (local_expert_idx * 2) * kIntermediate +
                                logical_row;
                            tma::copy<kBlockK, gran, kSwizzle, fp8_t>(
                                &tensor_map_w13_recompute,
                                &storage.full_barriers[stage_idx],
                                storage.smem_b[stage_idx] +
                                    (group * 2) * gran * kBlockK,
                                k_block_idx * kBlockK, up_row, 2);
                            tma::copy<kBlockK, gran, kSwizzle, fp8_t>(
                                &tensor_map_w13_recompute,
                                &storage.full_barriers[stage_idx],
                                storage.smem_b[stage_idx] +
                                    (group * 2 + 1) * gran * kBlockK,
                                k_block_idx * kBlockK, gate_row, 2);
                        }
                    }
                } else {
                    const auto* map =
                        kPhase == sched::BackwardBlockPhase::W2Dgrad
                            ? &tensor_map_w2_dgrad
                            : &tensor_map_w13_dgrad;
                    if (cute::elect_one_sync()) {
                        const uint32_t outer_k =
                            kPhase == sched::BackwardBlockPhase::W2Dgrad
                                ? local_expert_idx * kHidden +
                                      k_block_idx * kBlockK
                                : local_expert_idx * (2 * kIntermediate) +
                                      k_block_idx * kBlockK;
                        tma::copy<kBlockN, kBlockK, kSwizzle, fp8_t>(
                            map, &storage.full_barriers[stage_idx],
                            storage.smem_b[stage_idx],
                            n_block_idx * kBlockN, outer_k, 2);
                    }
                }

                #pragma unroll
                for (uint32_t row = lane_idx; row < kBlockN; row += 32) {
                    float scale;
                    if constexpr (
                        kPhase == sched::BackwardBlockPhase::RecomputeW13) {
                        constexpr uint32_t gran = 8;
                        constexpr uint32_t logical_rows = kBlockN / 2;
                        const uint32_t segment = row / gran;
                        const uint32_t logical_row =
                            n_block_idx * logical_rows +
                            (segment / 2) * gran;
                        const uint32_t canonical_expert =
                            local_expert_idx * 2 +
                            ((segment & 1u) ? 0u : 1u);
                        scale = __ldg(
                            w13_scales +
                            (canonical_expert * (kIntermediate / 128) +
                             logical_row / 128) *
                                (kHidden / 128) +
                            k_block_idx);
                    } else if constexpr (
                        kPhase == sched::BackwardBlockPhase::W2Dgrad) {
                        scale = __ldg(
                            w2_scales +
                            (local_expert_idx * (kHidden / 128) +
                             k_block_idx) *
                                (kIntermediate / 128) +
                            n_block_idx);
                    } else {
                        const uint32_t plane =
                            k_block_idx / (kIntermediate / 128);
                        const uint32_t row_block =
                            k_block_idx % (kIntermediate / 128);
                        scale = __ldg(
                            w13_scales +
                            ((local_expert_idx * 2 + plane) *
                                 (kIntermediate / 128) +
                             row_block) *
                                (kHidden / 128) +
                            n_block_idx);
                    }
                    const uint32_t exponent =
                        __float_as_uint(scale) >> 23;
                    storage.smem_sfb[stage_idx][row] =
                        exponent * 0x01010101u;
                }
                __syncwarp();
                if (cute::elect_one_sync()) {
                    if (leader_cta) {
                        storage.full_barriers[stage_idx]
                            .arrive_and_expect_tx(
                                sizeof(storage.smem_b[0]));
                    } else {
                        storage.full_barriers[stage_idx].arrive(0u);
                    }
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == 6) {
        cutlass::arch::warpgroup_reg_dealloc<40>();
        if (leader_cta) {
            auto instr_desc =
                cute::UMMA::make_instr_desc_block_scaled<
                    fp8_t, fp8_t, float, cutlass::float_ue8m0_t,
                    kUMMAM, kUMMAN, major_b,
                    cute::UMMA::Major::K>();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);
            auto a_desc = mma::sm100::make_umma_desc<
                cute::UMMA::Major::K, kLoadBlockM, kBlockK, kSwizzle>(
                storage.smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<
                major_b, kLoadBlockN, kBlockK, kSwizzle>(
                storage.smem_b[0], 0, 0);
            const uint32_t a_desc_lo =
                lane_idx < kStages
                    ? a_desc.lo +
                          lane_idx * sizeof(storage.smem_a[0]) / 16
                    : 0u;
            const uint32_t b_desc_lo =
                lane_idx < kStages
                    ? b_desc.lo +
                          lane_idx * sizeof(storage.smem_b[0]) / 16
                    : 0u;
            uint32_t current_iter = 0;
            for (uint32_t block = blockIdx.x;
                 block < num_m_blocks * num_block_ns;
                 block += kNumSMs) {
                const uint32_t m_block_idx = block / num_block_ns;
                const uint32_t valid_m = cute::min(
                    num_tokens - m_block_idx * 192u, 192u);
                mma::sm100::update_instr_desc_with_umma_n(
                    instr_desc, math::align(valid_m, 16u));
                const uint32_t accum_stage =
                    current_iter % kNumEpilogueStages;
                const uint32_t accum_phase =
                    (current_iter++ / kNumEpilogueStages) & 1;
                storage.tmem_empty_barriers[accum_stage].wait(
                    accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                #pragma unroll 2
                for (uint32_t k_block_idx = 0;
                     k_block_idx < num_block_ks;
                     advance_pipeline(k_block_idx)) {
                    storage.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    const uint32_t a_base =
                        ptx::exchange(a_desc_lo, stage_idx);
                    const uint32_t b_base =
                        ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        auto* sfa = storage.smem_sfa[stage_idx];
                        mma::sm100::replace_smem_desc_addr(sf_desc, sfa);
                        cute::SM100_UTCCP_4x32dp128bit_2cta::copy(
                            sf_desc, 384u);
                        mma::sm100::replace_smem_desc_addr(
                            sf_desc, storage.smem_sfb[stage_idx]);
                        cute::SM100_UTCCP_4x32dp128bit_2cta::copy(
                            sf_desc, 392u);
                        #pragma unroll
                        for (uint32_t k = 0; k < kBlockK / kUMMAK; ++k) {
                            const auto runtime_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(
                                    instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, kLoadBlockM,
                                kSwizzle, fp8_t>(a_base, 0, k * kUMMAK);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                major_b, kLoadBlockN, kSwizzle, fp8_t>(
                                b_base, 0, k * kUMMAK);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc,
                                accum_stage * kUMMAN,
                                k_block_idx > 0 || k > 0,
                                runtime_desc,
                                392u, 384u);
                        }
                    }
                    __syncwarp();
                    constexpr uint16_t cta_mask = 3;
                    cutlass::arch::umma_arrive_multicast_2x1SM(
                        reinterpret_cast<uint64_t*>(
                            &storage.empty_barriers[stage_idx]),
                        cta_mask);
                    if (k_block_idx == num_block_ks - 1) {
                        cutlass::arch::umma_arrive_multicast_2x1SM(
                            reinterpret_cast<uint64_t*>(
                                &storage.tmem_full_barriers[accum_stage]),
                            cta_mask);
                    }
                    __syncwarp();
                }
            }
            if (current_iter > 0) {
                const uint32_t last = current_iter - 1;
                storage.tmem_empty_barriers[
                    last % kNumEpilogueStages]
                    .wait((last / kNumEpilogueStages) & 1);
            }
        }
    } else if (warp_idx >= 8) {
        cutlass::arch::warpgroup_reg_alloc<208>();
        const uint32_t epilogue_warp_idx = warp_idx - 8;
        uint32_t current_iter = 0;
        uint32_t tma_stage_idx = 0;
        auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
            return storage.smem_cd[i];
        });
        for (uint32_t block = blockIdx.x;
             block < num_m_blocks * num_block_ns; block += kNumSMs) {
            const uint32_t m_block_idx = block / num_block_ns;
            const uint32_t n_block_idx = block % num_block_ns;
            const uint32_t valid_m = cute::min(
                num_tokens - m_block_idx * 192u, 192u);
            const uint32_t accum_stage =
                current_iter % kNumEpilogueStages;
            const uint32_t accum_phase =
                (current_iter++ / kNumEpilogueStages) & 1;
            storage.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_after_thread_sync();
            epilogue::sm100_store_cd_swap_ab<
                kBlockM, kBlockN, kStoreBlockM, kBlockN,
                kSwizzle, kNumTMAStoreStages, kEpilogueThreads,
                GemmType::Normal, false, bf16_t,
                epilogue::transform::EpilogueIdentity>(
                smem_cd, tma_stage_idx,
                accum_stage * kUMMAN,
                m_block_idx * kBlockM,
                n_block_idx * kBlockN, 0,
                math::align(valid_m, 16u),
                epilogue_warp_idx, lane_idx,
                &storage.tmem_empty_barriers[accum_stage],
                tensor_map_output);
        }
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
        __syncwarp();
    } else {
        cutlass::arch::warpgroup_reg_dealloc<40>();
    }

    __syncthreads();
    comm::cluster_sync_with_relaxed_arrive();
}

template <uint32_t kNumRanks, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kThreads, 1) void
sm103_fp8_block128_mega_moe_backward_impl(
    const int* expert_counts,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_tokens,
    const uint32_t capacity,
    const uint32_t ring_tokens,
    const uint32_t sf_ring_tokens,
    const uint32_t max_pool_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ layout::Workspace workspace,
    const bf16_t* compact_x,
    const bf16_t* compact_grad_y,
    const float* compact_scores,
    fp8_t* symmetric_x,
    uint32_t* symmetric_x_sf,
    fp8_t* symmetric_grad_y,
    uint32_t* symmetric_grad_y_sf,
    float* symmetric_scores,
    float* symmetric_grad_scores,
    bf16_t* combine_slots,
    fp8_t* ring_x,
    uint32_t* ring_x_sf,
    fp8_t* ring_grad_y,
    uint32_t* ring_grad_y_sf,
    float* ring_scores,
    fp8_t* ring_h,
    uint32_t* ring_h_sf,
    fp8_t* ring_grad_preact,
    uint32_t* ring_grad_preact_sf,
    bf16_t* ring_bf16,
    float* ring_dscore,
    fp8_t* full_h,
    uint32_t* full_h_sf,
    fp8_t* full_grad_preact,
    uint32_t* full_grad_preact_sf,
    bf16_t* grad_x,
    float* grad_scores,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_x,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_x_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_grad_y,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_grad_y_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_h,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_h_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_grad_preact,
    const __grid_constant__ cute::TmaDescriptor tensor_map_ring_grad_preact_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_recompute,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_dgrad,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_dgrad,
    const __grid_constant__ cute::TmaDescriptor tensor_map_gate_up,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_h,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_x,
    const float* w13_scales,
    const float* w2_scales) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    constexpr uint32_t kLocalExperts = kGlobalExperts / kNumRanks;
    const uint32_t global_thread =
        blockIdx.x * kThreads + threadIdx.x;
    const uint32_t global_stride = kNumSMs * kThreads;
    const uint32_t group_idx = threadIdx.x / 128;
    const uint32_t group_lane = threadIdx.x % 128;
    const uint32_t group_global = blockIdx.x * 4 + group_idx;
    const uint32_t group_stride = kNumSMs * 4;
    const uint32_t hidden_blocks = kHidden / 128;
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    SharedStorage& storage =
        *reinterpret_cast<SharedStorage*>(smem_buffer);

    // Install this call's ordinary PyTorch inputs into the symmetric compact
    // planes.  Four 128-thread groups quantize independent rows/blocks; the
    // resulting FP32 ABI is represented by four repeated UE8M0 bytes.
    for (uint64_t work = group_global;
         work < static_cast<uint64_t>(num_tokens) * hidden_blocks;
         work += group_stride) {
        const uint32_t row = work / hidden_blocks;
        const uint32_t block = work -
                               static_cast<uint64_t>(row) * hidden_blocks;
        const uint32_t col = block * 128 + group_lane;
        const uint64_t index =
            static_cast<uint64_t>(row) * kHidden + col;
        const float x = static_cast<float>(compact_x[index]);
        const float dy = static_cast<float>(compact_grad_y[index]);
        const float x_amax = reduce_group_128<true>(
            cute::abs(x), storage, group_idx);
        const float dy_amax = reduce_group_128<true>(
            cute::abs(dy), storage, group_idx);
        float x_inv, dy_inv;
        const uint32_t x_scale = packed_power2_scale(x_amax, x_inv);
        const uint32_t dy_scale = packed_power2_scale(dy_amax, dy_inv);
        if (group_lane == 0) {
            symmetric_x_sf[row * hidden_blocks + block] = x_scale;
            symmetric_grad_y_sf[row * hidden_blocks + block] = dy_scale;
        }
        symmetric_x[index] = fp8_t(x * x_inv);
        symmetric_grad_y[index] = fp8_t(dy * dy_inv);
        if (block == 0 && group_lane < kTopK) {
            const uint64_t route =
                static_cast<uint64_t>(row) * kTopK + group_lane;
            symmetric_scores[route] = compact_scores[route];
        }
    }
    __syncthreads();

    // Publish compact q/s to peers before any expert wave performs remote
    // gathers. Grid/NVLink counters are reusable across calls and layers.
    comm::nvlink_barrier<kNumRanks, kNumSMs, kThreads, 2, 81>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x,
        []() { __syncthreads(); });

    if (cutlass::canonical_warp_idx_sync() == 7 &&
        cute::elect_one_sync()) {
        cutlass::arch::fence_barrier_init();
    }
    comm::cluster_sync_with_relaxed_arrive();
    if (cutlass::canonical_warp_idx_sync() == 7)
        cute::TMEM::Allocator2Sm().allocate(kNumTmemCols, &storage.tmem_ptr);
    comm::cluster_sync_with_relaxed_arrive();

    uint32_t pool_block_offset = 0;
    #pragma unroll 1
    for (uint32_t expert = 0; expert < kLocalExperts; ++expert) {
        const uint32_t count =
            static_cast<uint32_t>(__ldg(expert_counts + expert));
        const uint32_t pool_row_offset = pool_block_offset * kBlockM;
        DG_DEVICE_ASSERT(count <= ring_tokens);
        DG_DEVICE_ASSERT(
            pool_row_offset + count <= max_pool_tokens);

        // Gather both compact operands and their exact route score into the
        // local expert ring. FP8 vectors stay FP8; no dequantized x/dy pool is
        // constructed.
        constexpr uint32_t vecs_per_row = kHidden / sizeof(uint4);
        for (uint64_t linear = global_thread;
             linear < static_cast<uint64_t>(count) * vecs_per_row;
             linear += global_stride) {
            const uint32_t row = linear / vecs_per_row;
            const uint32_t vec = linear -
                                 static_cast<uint64_t>(row) * vecs_per_row;
            const auto metadata =
                token_src_metadata[pool_row_offset + row];
            const auto* remote_x = sym_buffer.map(
                reinterpret_cast<const uint4*>(symmetric_x) +
                    static_cast<uint64_t>(metadata.token_idx) * vecs_per_row +
                    vec,
                metadata.rank_idx);
            const auto* remote_dy = sym_buffer.map(
                reinterpret_cast<const uint4*>(symmetric_grad_y) +
                    static_cast<uint64_t>(metadata.token_idx) * vecs_per_row +
                    vec,
                metadata.rank_idx);
            reinterpret_cast<uint4*>(ring_x)[
                static_cast<uint64_t>(row) * vecs_per_row + vec] =
                *remote_x;
            reinterpret_cast<uint4*>(ring_grad_y)[
                static_cast<uint64_t>(row) * vecs_per_row + vec] =
                *remote_dy;
        }
        for (uint64_t linear = global_thread;
             linear < static_cast<uint64_t>(count) * hidden_blocks;
             linear += global_stride) {
            const uint32_t row = linear / hidden_blocks;
            const uint32_t block = linear -
                                   static_cast<uint64_t>(row) * hidden_blocks;
            const auto metadata =
                token_src_metadata[pool_row_offset + row];
            const uint64_t remote_index =
                static_cast<uint64_t>(metadata.token_idx) * hidden_blocks +
                block;
            const uint32_t sf_row = transform_sf_row(row);
            ring_x_sf[block * sf_ring_tokens + sf_row] =
                *sym_buffer.map(symmetric_x_sf + remote_index,
                                metadata.rank_idx);
            ring_grad_y_sf[block * sf_ring_tokens + sf_row] =
                *sym_buffer.map(symmetric_grad_y_sf + remote_index,
                                metadata.rank_idx);
        }
        for (uint32_t row = global_thread; row < count;
             row += global_stride) {
            const auto metadata =
                token_src_metadata[pool_row_offset + row];
            ring_scores[row] = *sym_buffer.map(
                symmetric_scores +
                    static_cast<uint64_t>(metadata.token_idx) * kTopK +
                    metadata.topk_idx,
                metadata.rank_idx);
            ring_dscore[row] = 0.0f;
        }
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        if (count != 0) {
            run_gemm_phase<
                sched::BackwardBlockPhase::RecomputeW13, kNumSMs>(
                storage, expert, count,
                tensor_map_ring_x, tensor_map_ring_x_sf,
                tensor_map_w13_recompute, tensor_map_w2_dgrad,
                tensor_map_w13_dgrad, tensor_map_gate_up,
                w13_scales, w2_scales, sf_ring_tokens);
        }
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        // [up8,gate8] physical W13 output -> logical h, quantized per 128
        // columns with the same power-of-two recipe as compact x.
        constexpr uint32_t h_blocks = kIntermediate / 128;
        for (uint64_t work = group_global;
             work < static_cast<uint64_t>(count) * h_blocks;
             work += group_stride) {
            const uint32_t row = work / h_blocks;
            const uint32_t block = work -
                                   static_cast<uint64_t>(row) * h_blocks;
            const uint32_t h_col = block * 128 + group_lane;
            const uint32_t w13_block = h_col / 64;
            const uint32_t in_block = h_col % 64;
            const uint32_t physical_up =
                w13_block * 128 + (in_block / 8) * 16 + in_block % 8;
            const uint32_t physical_gate = physical_up + 8;
            const float up = static_cast<float>(
                ring_bf16[static_cast<uint64_t>(row) * kHidden +
                            physical_up]);
            const float gate = static_cast<float>(
                ring_bf16[static_cast<uint64_t>(row) * kHidden +
                            physical_gate]);
            const float sigmoid = 1.0f / (1.0f + expf(-gate));
            const float h = up * gate * sigmoid;
            const float amax = reduce_group_128<true>(
                cute::abs(h), storage, group_idx);
            float inv;
            const uint32_t packed = packed_power2_scale(amax, inv);
            const uint64_t ring_index =
                static_cast<uint64_t>(row) * kIntermediate + h_col;
            const uint64_t full_index =
                static_cast<uint64_t>(pool_row_offset + row) *
                    kIntermediate +
                h_col;
            ring_h[ring_index] = fp8_t(h * inv);
            full_h[full_index] = fp8_t(h * inv);
            if (group_lane == 0) {
                const uint32_t sf_row = transform_sf_row(row);
                ring_h_sf[block * sf_ring_tokens + sf_row] = packed;
                full_h_sf[
                    static_cast<uint64_t>(pool_row_offset + row) *
                        h_blocks +
                    block] = packed;
            }
        }
        __syncthreads();
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        if (count != 0) {
            run_gemm_phase<
                sched::BackwardBlockPhase::W2Dgrad, kNumSMs>(
                storage, expert, count,
                tensor_map_ring_grad_y,
                tensor_map_ring_grad_y_sf,
                tensor_map_w13_recompute, tensor_map_w2_dgrad,
                tensor_map_w13_dgrad, tensor_map_grad_h,
                w13_scales, w2_scales, sf_ring_tokens);
        }
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        // Exact per-route dscore = dot(W2^T dy, h) in a deterministic
        // 128-thread tree. This avoids retaining the 6144-wide down output.
        for (uint32_t row_work = group_global; row_work < count;
             row_work += group_stride) {
            float partial = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < kIntermediate / 128; ++i) {
                const uint32_t col = group_lane + i * 128;
                const uint32_t packed =
                    ring_h_sf[(col / 128) * sf_ring_tokens +
                              transform_sf_row(row_work)];
                const uint32_t exponent = packed & 0xffu;
                const float scale =
                    __uint_as_float(exponent << 23);
                const float h = static_cast<float>(
                                    ring_h[static_cast<uint64_t>(row_work) *
                                               kIntermediate +
                                           col]) *
                                scale;
                const float dh = static_cast<float>(
                    ring_bf16[static_cast<uint64_t>(row_work) * kHidden +
                              2 * kIntermediate +
                              col]);
                partial = __fmaf_rn(dh, h, partial);
            }
            partial = reduce_group_128<false>(
                partial, storage, group_idx);
            if (group_lane == 0)
                ring_dscore[row_work] = partial;
        }
        __syncthreads();

        constexpr uint32_t grad_blocks = (2 * kIntermediate) / 128;
        for (uint64_t work = group_global;
             work < static_cast<uint64_t>(count) * grad_blocks;
             work += group_stride) {
            const uint32_t row = work / grad_blocks;
            const uint32_t grad_block = work -
                                        static_cast<uint64_t>(row) * grad_blocks;
            const bool gate_plane = grad_block < h_blocks;
            const uint32_t block =
                gate_plane ? grad_block : grad_block - h_blocks;
            const uint32_t col = block * 128 + group_lane;
            const uint32_t w13_block = col / 64;
            const uint32_t in_block = col % 64;
            const uint32_t physical_up =
                w13_block * 128 + (in_block / 8) * 16 + in_block % 8;
            const float up = static_cast<float>(
                ring_bf16[static_cast<uint64_t>(row) * kHidden +
                            physical_up]);
            const float gate = static_cast<float>(
                ring_bf16[static_cast<uint64_t>(row) * kHidden +
                            physical_up + 8]);
            const float dy_h = static_cast<float>(
                                   ring_bf16[
                                       static_cast<uint64_t>(row) * kHidden +
                                       2 * kIntermediate +
                                       col]) *
                               ring_scores[row];
            const float sigmoid = 1.0f / (1.0f + expf(-gate));
            const float grad_value = gate_plane
                ? dy_h * up * sigmoid *
                      (1.0f + gate * (1.0f - sigmoid))
                : dy_h * gate * sigmoid;
            const float amax = reduce_group_128<true>(
                cute::abs(grad_value), storage, group_idx);
            float inv;
            const uint32_t packed = packed_power2_scale(amax, inv);
            const uint32_t logical_col = grad_block * 128 + group_lane;
            const uint64_t ring_index =
                static_cast<uint64_t>(row) *
                    (2 * kIntermediate) +
                logical_col;
            const uint64_t full_index =
                static_cast<uint64_t>(pool_row_offset + row) *
                    (2 * kIntermediate) +
                logical_col;
            ring_grad_preact[ring_index] = fp8_t(grad_value * inv);
            full_grad_preact[full_index] = fp8_t(grad_value * inv);
            if (group_lane == 0) {
                const uint32_t sf_row = transform_sf_row(row);
                ring_grad_preact_sf[
                    grad_block * sf_ring_tokens + sf_row] = packed;
                full_grad_preact_sf[
                    static_cast<uint64_t>(pool_row_offset + row) *
                        grad_blocks +
                    grad_block] = packed;
            }
        }
        __syncthreads();
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        if (count != 0) {
            run_gemm_phase<
                sched::BackwardBlockPhase::W13Dgrad, kNumSMs>(
                storage, expert, count,
                tensor_map_ring_grad_preact,
                tensor_map_ring_grad_preact_sf,
                tensor_map_w13_recompute, tensor_map_w2_dgrad,
                tensor_map_w13_dgrad, tensor_map_grad_x,
                w13_scales, w2_scales, sf_ring_tokens);
        }
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });

        // Publish each route directly into its immutable source top-k slot.
        for (uint64_t linear = global_thread;
             linear < static_cast<uint64_t>(count) * vecs_per_row;
             linear += global_stride) {
            const uint32_t row = linear / vecs_per_row;
            const uint32_t vec = linear -
                                 static_cast<uint64_t>(row) * vecs_per_row;
            const auto metadata =
                token_src_metadata[pool_row_offset + row];
            auto* remote_dst = sym_buffer.map(
                reinterpret_cast<uint4*>(combine_slots) +
                    (static_cast<uint64_t>(metadata.topk_idx) * capacity +
                     metadata.token_idx) *
                        vecs_per_row +
                    vec,
                metadata.rank_idx);
            *remote_dst = reinterpret_cast<const uint4*>(ring_bf16)[
                static_cast<uint64_t>(row) * vecs_per_row + vec];
        }
        for (uint32_t row = global_thread; row < count;
             row += global_stride) {
            const auto metadata =
                token_src_metadata[pool_row_offset + row];
            *sym_buffer.map(
                symmetric_grad_scores +
                    static_cast<uint64_t>(metadata.token_idx) * kTopK +
                    metadata.topk_idx,
                metadata.rank_idx) = ring_dscore[row];
        }
        comm::grid_sync<kNumSMs, 0>(
            workspace, blockIdx.x, threadIdx.x,
            []() { __syncthreads(); });
        pool_block_offset += math::ceil_div(count, kBlockM);
    }

    comm::nvlink_barrier<kNumRanks, kNumSMs, kThreads, 3, 82>(
        workspace, sym_buffer, blockIdx.x, threadIdx.x,
        []() { __syncthreads(); });

    for (uint64_t linear = global_thread;
         linear < static_cast<uint64_t>(num_tokens) * kHidden;
         linear += global_stride) {
        const uint32_t token = linear / kHidden;
        const uint32_t col = linear -
                             static_cast<uint64_t>(token) * kHidden;
        float value = 0.0f;
        #pragma unroll
        for (uint32_t slot = 0; slot < kTopK; ++slot) {
            value = __fadd_rn(
                value,
                static_cast<float>(
                    combine_slots[
                        (static_cast<uint64_t>(slot) * capacity + token) *
                            kHidden +
                        col]));
        }
        grad_x[linear] = bf16_t(value);
    }
    for (uint64_t route = global_thread;
         route < static_cast<uint64_t>(num_tokens) * kTopK;
         route += global_stride) {
        grad_scores[route] = symmetric_grad_scores[route];
    }
    __syncthreads();
    comm::grid_sync<kNumSMs, 1>(
        workspace, blockIdx.x, threadIdx.x,
        []() { __syncthreads(); });
    if (cutlass::canonical_warp_idx_sync() == 7)
        cute::TMEM::Allocator2Sm().free(0, kNumTmemCols);
#endif
}

}  // namespace sm103_block128_backward

}  // namespace deep_gemm
