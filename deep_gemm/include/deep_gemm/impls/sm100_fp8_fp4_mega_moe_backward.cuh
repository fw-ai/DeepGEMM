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
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Convert one E2M1 value multiplied by its power-of-two MX scale directly to
// BF16 bits. This is the packed-weight bridge used by MoK: it avoids a
// half->FP32 multiply for every source element while preserving the exact
// BF16 boundary consumed by the reference MXFP4 path.
CUTLASS_DEVICE uint16_t k3_mxfp4_bf16_bits(
    const uint8_t nibble, const uint32_t scale_bits) {
    const int magnitude_code = nibble & 0x7;
    const int scale_exponent = (scale_bits >> 23) & 0xff;
    if (magnitude_code == 0 || scale_exponent == 0)
        return 0;
    const int value_exponent =
        scale_exponent + (magnitude_code >> 1) - 1;
    const uint16_t sign =
        static_cast<uint16_t>(nibble & 0x8) << 12;
    const uint16_t mantissa =
        magnitude_code > 1 && (magnitude_code & 1) ? 0x40 : 0;
    return sign |
        static_cast<uint16_t>(value_exponent << 7) |
        mantissa;
}

// Quantize four BF16 values with a caller-selected UE8M0 scale byte. The two
// packed conversion instructions mirror MoK's converter and return the native
// four-byte E4M3 order expected by the DeepGEMM shared-memory swizzle.
CUTLASS_DEVICE uint32_t k3_quantize_bf16x4_e4m3(
    const uint16_t (&bf16_bits)[4], const uint32_t scale_byte) {
    const uint32_t pair01_bits =
        static_cast<uint32_t>(bf16_bits[0]) |
        (static_cast<uint32_t>(bf16_bits[1]) << 16);
    const uint32_t pair23_bits =
        static_cast<uint32_t>(bf16_bits[2]) |
        (static_cast<uint32_t>(bf16_bits[3]) << 16);
    const float2 pair01 = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&pair01_bits));
    const float2 pair23 = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&pair23_bits));
    const float scale_inv =
        __uint_as_float((254u - scale_byte) << 23);
    uint16_t fp8_pair01;
    uint16_t fp8_pair23;
    asm volatile(
        "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
        : "=h"(fp8_pair01)
        : "f"(pair01.x * scale_inv),
          "f"(pair01.y * scale_inv));
    asm volatile(
        "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
        : "=h"(fp8_pair23)
        : "f"(pair23.x * scale_inv),
          "f"(pair23.y * scale_inv));
    return static_cast<uint32_t>(fp8_pair01) |
        (static_cast<uint32_t>(fp8_pair23) << 16);
}

// Re-group one normal-layout 128x128 native MXFP4 tile into the transposed
// MXFP8 layout used by dgrad. This is the MoK row-owner mapping: four warps
// own the 128 output rows, and the 32 rows in each warp cooperatively load the
// 32 independent source scales before selecting them with warp shuffles. The
// previous one-warp-per-row mapping reloaded every source scale for every
// output row and was dominated by shared-memory traffic.
CUTLASS_DEVICE void k3_mxfp4_to_mxfp8_transposed_tile(
    const int8_t* packed,
    const float* source_scales,
    uint8_t* output,
    uint32_t* output_scales,
    uint8_t* cached_output,
    uint32_t* cached_output_scales,
    const uint32_t output_mn,
    const uint32_t output_k,
    const uint32_t n_block_idx,
    const uint32_t k_block_idx,
    const uint32_t producer_thread_idx) {
    constexpr uint32_t kTile = 128;
    constexpr uint32_t kGroup = 32;
    constexpr uint32_t kPackedStride = kTile / 2;
    constexpr uint32_t kScaleStride = kTile / kGroup;
    if (producer_thread_idx >= kTile)
        return;

    const uint32_t warp = producer_thread_idx / 32;
    const uint32_t lane = producer_thread_idx % 32;
    const uint32_t local_n = producer_thread_idx;
    const uint32_t global_n = n_block_idx * kTile + local_n;
    const uint32_t packed_col = local_n / 2;
    const uint32_t nibble_shift = (global_n & 1u) * 4;
    const uint32_t row = local_n & 7u;

    uint32_t scale_word = 0;
    #pragma unroll
    for (uint32_t group = 0; group < 4; ++group) {
        const uint32_t lane_scale_bits =
            *reinterpret_cast<const uint32_t*>(
                &source_scales[
                    (group * kGroup + lane) * kScaleStride + warp]);
        uint32_t amax_bits = 0;
        #pragma unroll
        for (uint32_t pair = 0; pair < 16; ++pair) {
            const uint32_t source_row = group * kGroup + pair * 2;
            const uint8_t packed_first = static_cast<uint8_t>(
                packed[source_row * kPackedStride + packed_col]);
            const uint8_t packed_second = static_cast<uint8_t>(
                packed[(source_row + 1) * kPackedStride + packed_col]);
            const uint32_t first_scale_bits = __shfl_sync(
                0xffffffffu, lane_scale_bits, pair * 2);
            const uint32_t second_scale_bits = __shfl_sync(
                0xffffffffu, lane_scale_bits, pair * 2 + 1);
            const uint16_t first = k3_mxfp4_bf16_bits(
                (packed_first >> nibble_shift) & 0xf,
                first_scale_bits);
            const uint16_t second = k3_mxfp4_bf16_bits(
                (packed_second >> nibble_shift) & 0xf,
                second_scale_bits);
            amax_bits = cute::max(
                amax_bits, static_cast<uint32_t>(first & 0x7fff));
            amax_bits = cute::max(
                amax_bits, static_cast<uint32_t>(second & 0x7fff));
        }

        const uint32_t scale_byte = cute::max(
            88,
            static_cast<int>((amax_bits >> 7) & 0xff) - 8);
        scale_word |= scale_byte << (group * 8);
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            uint16_t word_values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t source_row =
                    group * kGroup + word * 4 + i;
                const uint8_t source_packed = static_cast<uint8_t>(
                    packed[source_row * kPackedStride + packed_col]);
                const uint32_t source_scale_bits = __shfl_sync(
                    0xffffffffu,
                    lane_scale_bits,
                    word * 4 + i);
                word_values[i] = k3_mxfp4_bf16_bits(
                    (source_packed >> nibble_shift) & 0xf,
                    source_scale_bits);
            }
            const uint32_t quantized = k3_quantize_bf16x4_e4m3(
                word_values, scale_byte);
            const uint32_t logical_k_byte = group * kGroup + word * 4;
            const uint32_t byte_offset =
                (local_n >> 3) * 8 * kTile +
                row * kTile +
                ((logical_k_byte >> 4) ^ row) * 16 +
                (logical_k_byte & 15);
            *reinterpret_cast<uint32_t*>(output + byte_offset) = quantized;
            *reinterpret_cast<uint32_t*>(
                cached_output +
                static_cast<uint64_t>(global_n) * output_k +
                k_block_idx * kTile + logical_k_byte) = quantized;
        }
    }

    const uint32_t local_sf_n =
        (local_n & 31u) * 4 + ((local_n >> 5) & 3u);
    const uint32_t global_sf_n =
        (global_n & ~127u) +
        (global_n & 31u) * 4 +
        ((global_n >> 5) & 3u);
    output_scales[local_sf_n] = scale_word;
    cached_output_scales[
        static_cast<uint64_t>(k_block_idx) * output_mn + global_sf_n] =
        scale_word;
}

// Publish one packed MXFP8 scale word into the same shared-memory address on
// another CTA in the cluster. The dgrad producer uses this to avoid repeating
// the leader CTA's row quantization solely to populate the follower's UTCCP
// scale plane.
CUTLASS_DEVICE void store_cluster_uint32(
    uint32_t* ptr, const uint32_t cta_rank, const uint32_t value) {
    const uint32_t remote_addr = cute::set_block_rank(
        cute::cast_smem_ptr_to_uint(ptr), cta_rank);
    asm volatile(
        "st.shared::cluster.u32 [%0], %1;\n"
        :: "r"(remote_addr), "r"(value)
        : "memory");
}

template <
    uint32_t kHidden, uint32_t kNumExperts, uint32_t BLOCK_M,
    uint32_t kNumSMs, uint32_t kNumThreads,
    uint32_t kNumRanks,
    CombineOrderMode kCombineOrderMode>
__device__ __forceinline__ void
bf16_mega_moe_reduce_post_down_route(
    const int* expert_counts,
    const cutlass::bfloat16_t* grad_y_unweighted,
    const cutlass::bfloat16_t* down_unweighted,
    float* grad_route_output,
    float* backward_grad_route,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_topk,
    const layout::SymBuffer<kNumRanks>& backward_sym_buffer,
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
        #pragma unroll 1
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
                if (threadIdx.x == 0) {
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
        #pragma unroll 1
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

    #pragma unroll 1
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
            if (route_group_lane_idx == 0) {
                grad_route_output[pool_row] = grad_route;
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
        #pragma unroll 1
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

    #pragma unroll 1
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
    #pragma unroll 1
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
    bool kInlineWeightDequant = false,
    bool kPhaseOrderedWeightDequant = false,
    bool kInlineResidualMXFP8Dgrad = false,
    bool kResidualMXFP8Dgrad = false,
    bool kBuildResidualMXFP8Weights = false,
    bool kExactSourceX = false,
    bool kGateUpPrepared = false,
    ActivationType kActivationType = ActivationType::SwiGLU,
    float kSituBeta = 1.0f,
    float kSituLinearBeta =
        cute::numeric_limits<float>::infinity(),
    bool kFastMath = false,
    RouteWeightMode kRouteWeightMode = RouteWeightMode::PreDown,
    CombineOrderMode kCombineOrderMode = CombineOrderMode::FixedTopK,
    bool kInputsPrepared = false,
    bool kDispatchInputsPrepared = false,
    bool kDirectRemoteGradX = false,
    bool kWriteGradXPool = true,
    bool kClearWgradPadding = false,
    bool kInlineWgrad = false,
    bool kAccumulateWgrad = false,
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
    cutlass::bfloat16_t* backward_grad_x_output,
    const uint32_t num_backward_tokens,
    const layout::TokenSrcMetadata* token_src_metadata,
    const uint32_t num_topk,
    const uint32_t num_pool_rows,
    const uint32_t num_acts_rows,
    const uint32_t acts_sf_stride,
    const __grid_constant__ cute::TmaDescriptor tensor_map_acts,
    const __grid_constant__ cute::TmaDescriptor tensor_map_acts_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_output,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_ye,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_dequant,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_dgrad_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_dgrad_weights_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_scales,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_dequant,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_dgrad_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_dgrad_weights_sf,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_weights,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_scales,
    const __grid_constant__ cute::TmaDescriptor tensor_map_grad_gate_up,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_wgrad_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_wgrad_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w2_wgrad_d,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_wgrad_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_wgrad_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_w13_wgrad_d,
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
    const bool clear_empty_wgrad_expert_outputs,
    const float activation_limit,
    uint64_t* kernel_trace) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;
    using cd_dtype_t = cutlass::bfloat16_t;
    using dgrad_b_dtype_t = cutlass::bfloat16_t;
    using residual_dgrad_dtype_t = cutlass::float_e4m3_t;

    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumDispatchThreads =
        kNumRanks > 1 ? 128 : 0;
    constexpr uint32_t kNumDispatchWarps =
        kNumDispatchThreads / 32;
    // Initial grad-y dispatch and the later route/exact-X pipeline run in
    // distinct phases but need independent transaction-barrier parity.
    constexpr uint32_t kNumDispatchBarrierStages = 2;
    constexpr uint32_t kNumDispatchBarriers =
        kNumDispatchWarps * kNumDispatchBarrierStages;
    constexpr uint32_t kDispatchWarpStart =
        (kNumNonEpilogueThreads + kNumEpilogueThreads) / 32;
    // Inline MXFP4 conversion is scalar CUDA work rather than UMMA work.  Use
    // six warps (1 and 3..7) to feed the dgrad pipeline; 24 warps remain for
    // the pool epilogue and remote grad-x scatter.  The non-inline TMA path
    // retains its original four-warp producer/MMA prefix.
    constexpr uint32_t kDgradEpilogueWarpStart =
        (kInlineWeightDequant || kResidualMXFP8Dgrad) ? 8 : 4;
    constexpr uint32_t kNumDgradEpilogueThreads =
        kNumThreads - kDgradEpilogueWarpStart * 32;
    constexpr uint32_t kNumInlineWeightProducerWarps = 6;
    constexpr uint32_t kNumInlineWeightProducerThreads =
        kNumInlineWeightProducerWarps * 32;
    constexpr uint32_t kInlineWeightProducerBarrier = 1;
    constexpr uint32_t kNumResidualProducerWarps = 7;
    constexpr uint32_t kNumResidualProducerThreads =
        kNumResidualProducerWarps * 32;
    // A/B readiness is centralized on CTA 0 because the 2-SM UMMA leader must
    // observe both operands from both CTAs. Each CTA contributes one A and one
    // B completion, for four cluster arrivals in every path.
    constexpr uint32_t kNumDgradFullBarrierArrivals = 4;
    constexpr uint32_t kResidualWeightProducerBarrier = 2;
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
    constexpr uint32_t DGRAD_BLOCK_K =
        kResidualMXFP8Dgrad ? 128 : 64;
    constexpr uint32_t DGRAD_UMMA_K =
        kResidualMXFP8Dgrad ? 32 : 16;
    // Match MoK's packed-weight path: producer warps convert the current MXFP4
    // tile immediately before its UMMA consumer. This has no cross-CTA cache
    // dependency and overlaps B conversion with A quantization/TMA progress.
    constexpr bool kPrefixedResidualWeightCache = false;
    constexpr bool kOnDemandResidualWeightCache =
        kInlineResidualMXFP8Dgrad;
    // The host rebinds the activation TMA descriptors to the aliased
    // primary/residual planes under this compile-time selector.
    constexpr bool kBuildW2ResidualActsOnce =
        kBuildResidualMXFP8Weights;
    constexpr bool kBuildW13ResidualActsOnce =
        kBuildResidualMXFP8Weights;
    // Once W2 dgrad has consumed its compact transpose, produce dW2 early.
    // Its exact BF16 operands then become scalable W13 residual-activation
    // scratch, avoiding both per-tile requantization and a fixed dW-sized
    // activation-cache ceiling.
    constexpr bool kEarlyW2Wgrad =
        kCompileW13Dgrad && kInlineWgrad &&
        kBuildW13ResidualActsOnce &&
        kRouteWeightMode == RouteWeightMode::PostDown;
    constexpr uint32_t kResidualWeightCacheBarrier = 3;
    // Experimental quality/performance gate: retain the primary MXFP8 dgrad
    // product while measuring whether the error-feedback product is required
    // for Kimi K3's cosine threshold.
    constexpr bool kApplyResidualDgradCorrection = true;
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
    DG_STATIC_ASSERT(
        !kResidualMXFP8Dgrad || (!kBF16Mode && !kInlineWeightDequant),
        "Residual MXFP8 dgrad is exclusive with BF16 and legacy inline dequant modes");
    DG_STATIC_ASSERT(
        !kGateUpPrepared || !kBF16Mode,
        "Prepared gate/up is supported only by the FP8/FP4 MegaMoE path");

    constexpr uint32_t kNumW13WeightTileStates =
        kNumExperts *
        ((2 * kIntermediateHidden) / DGRAD_BLOCK_K) *
        kNumW13DgradBlockNs;
    auto* phase_count =
        weight_tile_states + kNumW2WeightTileStates +
        kNumW13WeightTileStates;
    auto* phase_sense = phase_count + 1;
    // The caller-owned grid state persists across launches. Capture its
    // current phase before the first full-grid barrier and use the following
    // value as this launch's tile-ready epoch. The barrier cannot advance the
    // phase until every CTA has reached it, so every CTA observes the same
    // epoch without host-global state or a clearing launch.
    uint32_t launch_epoch = ptx::ld_acq(phase_sense) + 1u;
    if (launch_epoch == 0)
        launch_epoch = 1;
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
            // Dispatch sites execute only in the dedicated communication
            // warpgroup, so their first lane records them. All other sites
            // retain CTA-thread-zero ownership.
            const bool is_trace_thread =
                threadIdx.x == 0 ||
                ((site == 3 || site == 4) &&
                 threadIdx.x == kDispatchWarpStart * 32);
            if (is_trace_thread) {
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
            const bool is_trace_thread =
                threadIdx.x == 0 ||
                ((site == 3 || site == 4) &&
                 threadIdx.x == kDispatchWarpStart * 32);
            if (is_trace_thread) {
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
        if (trace_site < kTraceSiteCount)
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
                const uint64_t grid_wait_start = clock64();
                while (ptx::ld_acq(phase_sense) ==
                       old_sense) {
                    if (clock64() - grid_wait_start >
                        4000000000ull) {
                        printf(
                            "K3 grid wait timeout rank=%u sm=%u site=%u sense=%u count=%u\n",
                            backward_sym_buffer.rank_idx,
                            blockIdx.x,
                            trace_site,
                            old_sense,
                            ptx::ld_acq(phase_count));
                        asm volatile("trap;");
                    }
                }
            }
        }
        __syncthreads();
        if (trace_site < kTraceSiteCount)
            trace_end(trace_site);
    };

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    // K3's exact-source W13 wgrad can reuse either POST_DOWN route-dot input.
    // Defer the source-X refill until every row's route reader retires; an
    // early refill would turn dot(grad_y, down) into dot(x, down) or
    // dot(grad_y, x), depending on the selected allocation-free alias.
    const bool late_exact_source_x =
        !kBF16Mode && kExactSourceX &&
        (x_pool_output == grad_y_unweighted_output ||
         (kComputeRouteGrad &&
          kRouteWeightMode == RouteWeightMode::PostDown &&
          x_pool_output == down_unweighted_output));
    // Prepared forward preactivations use MegaMoE's [gate(8), up(8)] column
    // layout, whereas W13 dgrad/wgrad consume conventional [all gate | all
    // up] gradients. When both tensors alias, publish each derivative back to
    // the exact column it just consumed, then deinterleave only after every
    // activation CTA has retired. Direct conventional stores would race with
    // another N-tile CTA's still-live preactivation reads.
    const bool inplace_gate_up_grad =
        gate_up_output == grad_gate_up_output;
    uint32_t dispatch_pull_mbarrier_phase = 0;

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_grad_ye);
        cute::prefetch_tma_descriptor(&tensor_map_w2_dequant);
        if constexpr (kResidualMXFP8Dgrad) {
            cute::prefetch_tma_descriptor(&tensor_map_w2_dgrad_weights);
            cute::prefetch_tma_descriptor(&tensor_map_w2_dgrad_weights_sf);
            cute::prefetch_tma_descriptor(&tensor_map_w13_dgrad_weights);
            cute::prefetch_tma_descriptor(&tensor_map_w13_dgrad_weights_sf);
            if constexpr (kBuildResidualMXFP8Weights) {
                cute::prefetch_tma_descriptor(&tensor_map_acts);
                cute::prefetch_tma_descriptor(&tensor_map_acts_sf);
                cute::prefetch_tma_descriptor(&tensor_map_weights);
                cute::prefetch_tma_descriptor(&tensor_map_weights_sf);
                if constexpr (kEarlyW2Wgrad)
                    cute::prefetch_tma_descriptor(&tensor_map_output);
            }
        }
        cute::prefetch_tma_descriptor(&tensor_map_w13_dequant);
        cute::prefetch_tma_descriptor(&tensor_map_grad_gate_up);
        if constexpr (!kBF16Mode) {
            if constexpr (!kGateUpPrepared) {
                cute::prefetch_tma_descriptor(&tensor_map_acts);
                cute::prefetch_tma_descriptor(&tensor_map_acts_sf);
                cute::prefetch_tma_descriptor(&tensor_map_weights);
                cute::prefetch_tma_descriptor(&tensor_map_weights_sf);
                cute::prefetch_tma_descriptor(&tensor_map_output);
            }
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
    constexpr uint32_t SMEM_RESIDUAL_A_SIZE_PER_STAGE =
        kResidualMXFP8Dgrad ? SMEM_A_SIZE_PER_STAGE : 0;
    constexpr uint32_t SMEM_RESIDUAL_SFA_SIZE_PER_STAGE =
        kResidualMXFP8Dgrad ? SMEM_SFA_SIZE_PER_STAGE : 0;
    // Residual-MXFP8 dgrad re-encodes each transposed MXFP4 tile in the
    // persistent pipeline.  Stage the original packed values and FP32 scales
    // with TMA while the producer warps quantize A, then consume them from
    // shared memory.  This replaces scattered scalar global loads without a
    // persistent transformed-weight allocation.
    constexpr uint32_t SMEM_DGRAD_WEIGHT_SOURCE_SIZE =
        kResidualMXFP8Dgrad
        ? DGRAD_BLOCK_K * (LOAD_BLOCK_N / 2) * sizeof(int8_t)
        : 0;
    constexpr uint32_t SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE =
        kResidualMXFP8Dgrad
        ? DGRAD_BLOCK_K * (LOAD_BLOCK_N / kGranK) * sizeof(float)
        : 0;
    constexpr uint32_t kNumWeightLoadBarriers =
        kResidualMXFP8Dgrad ? 2 : 0;
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
        kResidualMXFP8Dgrad ||
            LOAD_BLOCK_M * DGRAD_BLOCK_K * sizeof(cd_dtype_t) ==
                SMEM_A_SIZE_PER_STAGE,
        "Dgrad A alias size mismatch");
    DG_STATIC_ASSERT(
        (kResidualMXFP8Dgrad &&
         LOAD_BLOCK_N * DGRAD_BLOCK_K *
                 sizeof(residual_dgrad_dtype_t) ==
             SMEM_B_SIZE_PER_STAGE) ||
            (!kResidualMXFP8Dgrad &&
             LOAD_BLOCK_N * DGRAD_BLOCK_K *
                     sizeof(dgrad_b_dtype_t) ==
                 SMEM_B_SIZE_PER_STAGE),
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

    auto* residual_a_start_ptr =
        reinterpret_cast<uint8_t*>(smem_sfb[kNumStages]);
    auto smem_dgrad_a_residual =
        utils::PatternVisitor([=](const uint32_t& i) {
            return reinterpret_cast<residual_dgrad_dtype_t*>(
                residual_a_start_ptr +
                i * SMEM_RESIDUAL_A_SIZE_PER_STAGE);
        });
    auto* residual_sfa_start_ptr =
        residual_a_start_ptr +
        kNumStages * SMEM_RESIDUAL_A_SIZE_PER_STAGE;
    auto smem_dgrad_sfa_residual =
        utils::PatternVisitor([=](const uint32_t& i) {
            return reinterpret_cast<uint32_t*>(
                residual_sfa_start_ptr +
                i * SMEM_RESIDUAL_SFA_SIZE_PER_STAGE);
        });

    auto* weight_source_start_ptr =
        residual_sfa_start_ptr +
        kNumStages * SMEM_RESIDUAL_SFA_SIZE_PER_STAGE;
    auto* smem_dgrad_weight_source =
        reinterpret_cast<int8_t*>(weight_source_start_ptr);
    auto* weight_scale_source_start_ptr =
        weight_source_start_ptr +
        SMEM_DGRAD_WEIGHT_SOURCE_SIZE;
    auto* smem_dgrad_weight_scale_source =
        reinterpret_cast<float*>(weight_scale_source_start_ptr);

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        weight_scale_source_start_ptr +
        SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE);
    auto full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto* weight_load_barrier = barrier_start_ptr + 2 * kNumStages;
    auto* residual_mma_barrier = weight_load_barrier + 1;
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages +
            kNumWeightLoadBarriers + i;
    });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages +
            kNumWeightLoadBarriers + kNumEpilogueStages + i;
    });
    auto dispatch_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + 2 * kNumStages +
            kNumWeightLoadBarriers + 2 * kNumEpilogueStages + i;
    });
    auto* dequant_barriers =
        barrier_start_ptr + 2 * kNumStages +
        kNumWeightLoadBarriers + 2 * kNumEpilogueStages +
        kNumDispatchBarriers;
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        dequant_barriers + 2);

    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumResidualSFATmemCols =
        kResidualMXFP8Dgrad ? kNumSFATmemCols : 0;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<
            kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols +
            kNumResidualSFATmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB =
        kNumAccumTmemCols + kNumSFATmemCols;
    constexpr uint32_t kTmemStartColOfResidualSFA =
        kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Backward recompute exceeds TMEM");

    // The grouped BF16 wgrad body is reused twice inside this persistent
    // kernel. Keep its policy at function scope so W2 may run as soon as its
    // dgrad operands retire, rather than forcing both wgrads behind W13.
    constexpr uint32_t kWgradBlockM = 128;
    constexpr uint32_t kWgradBlockN = 256;
    // K-grouped rows are separated by the forward pool's BLOCK_M. Select the
    // largest divisor so a final K tile cannot cross into the next expert.
    constexpr uint32_t kWgradBlockK =
        BLOCK_M % 64 == 0 ? 64 : BLOCK_M % 32 == 0 ? 32 : 16;
    constexpr uint32_t kWgradSwizzle =
        kWgradBlockK * sizeof(cd_dtype_t);
    constexpr uint32_t kWgradStages = 6;
    constexpr uint32_t kWgradRoleThreads = 128;
    constexpr uint32_t kWgradTensorUtil = 100;
    constexpr bool kPublishRemoteGradients =
        kNumRanks > 1 &&
        (kDirectRemoteGradX || kComputeRouteGrad);
    const auto run_wgrad = [&]<
        bool kFuseWgradCombine,
        uint32_t kExtraCombineThreads = 0,
        bool kPublishBeforeCombineReduce = false>(
        const uint32_t shape_m,
        const uint32_t shape_n,
        const cute::TmaDescriptor& tensor_map_a,
        const cute::TmaDescriptor& tensor_map_b,
        const cute::TmaDescriptor& tensor_map_d,
        const bool combine_reduce) {
        sm100_bf16_gemm_body<
            cute::UMMA::Major::MN,
            cute::UMMA::Major::MN,
            0, 0, 0,
            kWgradBlockM, kWgradBlockN, kWgradBlockK,
            kNumExperts,
            kWgradSwizzle, kWgradSwizzle, kWgradSwizzle,
            kWgradStages,
            kWgradRoleThreads, kWgradRoleThreads,
            2, false,
            kNumSMs,
            kWgradBlockK,
            false, true,
            GemmType::KGroupedContiguous,
            kAccumulateWgrad,
            cd_dtype_t,
            kWgradTensorUtil,
            kNumRanks,
            kFuseWgradCombine,
            CombineOrderMode::FixedTopK,
            kExtraCombineThreads,
            kPublishBeforeCombineReduce>(
                reinterpret_cast<int*>(weight_tile_states),
                shape_m, shape_n, num_pool_rows,
                tensor_map_a, tensor_map_b, tensor_map_d,
                backward_sym_buffer, backward_workspace,
                backward_grad_x_output,
                const_cast<cd_dtype_t*>(backward_grad_y),
                nullptr,
                num_backward_tokens,
                backward_workspace.num_max_tokens_per_rank,
                num_topk, kHidden, combine_reduce,
                smem_buffer, false);
    };

    const auto initialize_wgrad_grouped_layout = [&]() {
        for (uint32_t expert_idx = threadIdx.x;
             expert_idx < kNumExperts;
             expert_idx += kNumThreads) {
            weight_tile_states[expert_idx] =
                math::ceil_div(
                    static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx)),
                    BLOCK_M) *
                BLOCK_M;
        }
        __syncthreads();
    };

    const auto clear_w2_wgrad_padding_rows = [&]() {
        uint32_t pad_pool_block_offset = 0;
        uint32_t pad_global_block = 0;
        #pragma unroll 1
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
                if (pad_global_block % kNumSMs == blockIdx.x) {
                    for (uint32_t linear = threadIdx.x;
                         linear <
                             (BLOCK_M - last_valid) *
                                 (kHidden + kIntermediateHidden);
                         linear += kNumThreads) {
                        const uint32_t row_delta =
                            linear /
                            (kHidden + kIntermediateHidden);
                        const uint32_t col =
                            linear -
                            row_delta *
                                (kHidden + kIntermediateHidden);
                        const uint32_t pool_row =
                            pool_block * BLOCK_M +
                            last_valid + row_delta;
                        if (col < kHidden) {
                            grad_ye_output[
                                static_cast<uint64_t>(pool_row) *
                                    kHidden +
                                col] = cd_dtype_t(0.0f);
                        } else {
                            h_weighted_output[
                                static_cast<uint64_t>(pool_row) *
                                    kIntermediateHidden +
                                col - kHidden] =
                                cd_dtype_t(0.0f);
                        }
                    }
                }
                ++pad_global_block;
            }
            pad_pool_block_offset += num_blocks;
        }
    };

    // Re-encode the two native MXFP4 transposes once per launch into aliases
    // carved from the dW destinations. The source values are E2M1 multiplied
    // by power-of-two scales, so the primary E4M3 encoding is exact; the only
    // change is regrouping along the transposed dgrad K dimension.
    const auto build_residual_weights_once = [&]() {
      if constexpr (kBuildResidualMXFP8Weights) {
        constexpr uint32_t kValuesPerLane = 4;
        constexpr uint32_t kWarpsPerCTA = kNumThreads / 32;
        constexpr uint32_t kGlobalWarps = kNumSMs * kWarpsPerCTA;
        const uint32_t global_warp_idx =
            blockIdx.x * kWarpsPerCTA + warp_idx;
        const uint32_t group_idx = lane_idx / 8;
        const uint32_t lane_in_group = lane_idx % 8;

        auto* w2_q = reinterpret_cast<uint8_t*>(w2_dequant_scratch);
        auto* w2_sf = reinterpret_cast<uint32_t*>(
            w2_q + static_cast<uint64_t>(kNumExperts) *
                kIntermediateHidden * kHidden);
        constexpr uint32_t kW2KBlocks = kHidden / 128;
        constexpr uint64_t kW2Tasks =
            static_cast<uint64_t>(kNumExperts) *
            kIntermediateHidden * kW2KBlocks;
        for (uint64_t task = global_warp_idx;
             task < kW2Tasks;
             task += kGlobalWarps) {
            const uint32_t k_block_idx = task % kW2KBlocks;
            const uint64_t row = task / kW2KBlocks;
            const uint32_t n = row % kIntermediateHidden;
            const uint32_t expert_idx = row / kIntermediateHidden;
            if (__ldg(expert_counts + expert_idx) == 0)
                continue;
            const uint32_t global_k = k_block_idx * 128 +
                group_idx * 32 + lane_in_group * kValuesPerLane;
            float values[kValuesPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i) {
                const uint32_t source_k = global_k + i;
                const uint8_t packed = static_cast<uint8_t>(
                    w2_weights[
                        (static_cast<uint64_t>(expert_idx) * kHidden +
                         source_k) * (kIntermediateHidden / 2) + n / 2]);
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
                const float2 decoded = __half22float2(
                    *reinterpret_cast<__half2*>(&fp16x2));
                const float scale = w2_scales[
                    (static_cast<uint64_t>(expert_idx) * kHidden +
                     source_k) * (kIntermediateHidden / kGranK) +
                    n / kGranK];
                values[i] = (n & 1u ? decoded.y : decoded.x) * scale;
            }
            float amax = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i)
                amax = cute::max(amax, cute::abs(values[i]));
            #pragma unroll
            for (uint32_t offset = 4; offset > 0; offset >>= 1)
                amax = cute::max(
                    amax,
                    __shfl_xor_sync(0xffffffff, amax, offset, 8));
            float2 sf_pair;
            float2 sf_inv_pair;
            math::get_e4m3_sf_and_sf_inv(
                make_float2(amax, 0.0f), sf_pair, sf_inv_pair);
            const auto q = __nv_fp8x4_e4m3(make_float4(
                values[0] * sf_inv_pair.x,
                values[1] * sf_inv_pair.x,
                values[2] * sf_inv_pair.x,
                values[3] * sf_inv_pair.x));
            *reinterpret_cast<uint32_t*>(
                w2_q + row * kHidden + global_k) = q.__x;
            const uint32_t scale_byte =
                (*reinterpret_cast<const uint32_t*>(&sf_pair.x)) >> 23;
            uint32_t scale_word = 0;
            #pragma unroll
            for (uint32_t group = 0; group < 4; ++group)
                scale_word |= __shfl_sync(
                    0xffffffff, scale_byte, group * 8) << (group * 8);
            if (lane_idx == 0) {
                w2_sf[
                    static_cast<uint64_t>(expert_idx) *
                        kIntermediateHidden * kW2KBlocks +
                    n + static_cast<uint64_t>(k_block_idx) *
                        kIntermediateHidden] = scale_word;
            }
        }

        auto* w13_q = reinterpret_cast<uint8_t*>(w13_dequant_scratch);
        auto* w13_sf = reinterpret_cast<uint32_t*>(
            w13_q + static_cast<uint64_t>(kNumExperts) *
                kHidden * (2 * kIntermediateHidden));
        constexpr uint32_t kW13KBlocks =
            (2 * kIntermediateHidden) / 128;
        constexpr uint64_t kW13Tasks =
            static_cast<uint64_t>(kNumExperts) *
            kHidden * kW13KBlocks;
        for (uint64_t task = global_warp_idx;
             task < kW13Tasks;
             task += kGlobalWarps) {
            const uint32_t k_block_idx = task % kW13KBlocks;
            const uint64_t row = task / kW13KBlocks;
            const uint32_t n = row % kHidden;
            const uint32_t expert_idx = row / kHidden;
            if (__ldg(expert_counts + expert_idx) == 0)
                continue;
            const uint32_t global_k = k_block_idx * 128 +
                group_idx * 32 + lane_in_group * kValuesPerLane;
            float values[kValuesPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i) {
                const uint32_t source_k = global_k + i;
                const uint32_t source_expert =
                    expert_idx * 2 + source_k / kIntermediateHidden;
                const uint32_t source_row =
                    source_k % kIntermediateHidden;
                const uint8_t packed = static_cast<uint8_t>(
                    w13_weights[
                        (static_cast<uint64_t>(source_expert) *
                             kIntermediateHidden + source_row) *
                            (kHidden / 2) + n / 2]);
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
                const float2 decoded = __half22float2(
                    *reinterpret_cast<__half2*>(&fp16x2));
                const float scale = w13_scales[
                    (static_cast<uint64_t>(source_expert) *
                         kIntermediateHidden + source_row) *
                        (kHidden / kGranK) + n / kGranK];
                values[i] = (n & 1u ? decoded.y : decoded.x) * scale;
            }
            float amax = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i)
                amax = cute::max(amax, cute::abs(values[i]));
            #pragma unroll
            for (uint32_t offset = 4; offset > 0; offset >>= 1)
                amax = cute::max(
                    amax,
                    __shfl_xor_sync(0xffffffff, amax, offset, 8));
            float2 sf_pair;
            float2 sf_inv_pair;
            math::get_e4m3_sf_and_sf_inv(
                make_float2(amax, 0.0f), sf_pair, sf_inv_pair);
            const auto q = __nv_fp8x4_e4m3(make_float4(
                values[0] * sf_inv_pair.x,
                values[1] * sf_inv_pair.x,
                values[2] * sf_inv_pair.x,
                values[3] * sf_inv_pair.x));
            *reinterpret_cast<uint32_t*>(
                w13_q + row * (2 * kIntermediateHidden) + global_k) =
                q.__x;
            const uint32_t scale_byte =
                (*reinterpret_cast<const uint32_t*>(&sf_pair.x)) >> 23;
            uint32_t scale_word = 0;
            #pragma unroll
            for (uint32_t group = 0; group < 4; ++group)
                scale_word |= __shfl_sync(
                    0xffffffff, scale_byte, group * 8) << (group * 8);
            if (lane_idx == 0) {
                w13_sf[
                    static_cast<uint64_t>(expert_idx) *
                        kHidden * kW13KBlocks +
                    n + static_cast<uint64_t>(k_block_idx) * kHidden] =
                    scale_word;
            }
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        full_grid_phase_barrier(kTraceSiteCount);
      }
    };

    // Materialize the exact transposed dgrad operands tile-by-tile. Native
    // MXFP4 values and scales arrive through TMA in their coalesced forward
    // layout; all producer warps then transpose/re-group one 128x128 tile into
    // the MXFP8 aliases carved from the corresponding BF16 dW destination.
    // This replaces the scalar, strided prototype above without allocating a
    // persistent transformed-weight tensor.
    const auto build_residual_weights_tiled_once = [&]() {
      if constexpr (kBuildResidualMXFP8Weights) {
        constexpr uint32_t kValuesPerLane = 4;
        constexpr uint32_t kWarpsPerCTA = kNumThreads / 32;
        constexpr uint32_t kBuilderBlockN = 128;
        constexpr uint32_t kBuilderTmaN = 128;
        constexpr uint32_t kBuilderTmaChunks =
            kBuilderBlockN / kBuilderTmaN;
        constexpr uint32_t kBuilderWeightChunkBytes =
            DGRAD_BLOCK_K * (kBuilderTmaN / 2) * sizeof(int8_t);
        constexpr uint32_t kBuilderScaleChunkBytes =
            DGRAD_BLOCK_K * (kBuilderTmaN / kGranK) * sizeof(float);
        constexpr uint32_t kBuilderWeightBytes =
            kBuilderTmaChunks * kBuilderWeightChunkBytes;
        constexpr uint32_t kBuilderScaleBytes =
            kBuilderTmaChunks * kBuilderScaleChunkBytes;
        auto* builder_weight_source =
            reinterpret_cast<int8_t*>(smem_buffer);
        auto* builder_scale_source = reinterpret_cast<float*>(
            smem_buffer + kBuilderWeightBytes);
        const uint32_t group_idx = lane_idx / 8;
        const uint32_t lane_in_group = lane_idx % 8;
        if (warp_idx == 0 && cute::elect_one_sync()) {
            weight_load_barrier->init(1);
            cutlass::arch::fence_barrier_init();
        }
        __syncthreads();

        uint32_t weight_source_phase = 0;
        const auto transform_operand = [&](const bool w13) {
            const uint32_t output_mn =
                w13 ? kHidden : kIntermediateHidden;
            const uint32_t output_k =
                w13 ? 2 * kIntermediateHidden : kHidden;
            const uint32_t num_n_blocks =
                math::ceil_div(output_mn, kBuilderBlockN);
            const uint32_t num_k_blocks = output_k / DGRAD_BLOCK_K;
            const uint32_t source_rows_per_expert = output_k;
            const uint64_t num_tiles =
                static_cast<uint64_t>(kNumExperts) *
                num_n_blocks * num_k_blocks;
            auto* q = reinterpret_cast<uint8_t*>(
                w13 ? w13_dequant_scratch : w2_dequant_scratch);
            auto* sf = reinterpret_cast<uint32_t*>(
                q + static_cast<uint64_t>(kNumExperts) *
                    output_mn * output_k);
            const auto* weights_map = w13
                ? &tensor_map_w13_weights
                : &tensor_map_w2_weights;
            const auto* scales_map = w13
                ? &tensor_map_w13_scales
                : &tensor_map_w2_scales;

            for (uint64_t tile_idx = blockIdx.x;
                 tile_idx < num_tiles;
                 tile_idx += kNumSMs) {
                const uint32_t k_block_idx = tile_idx % num_k_blocks;
                const uint64_t n_expert_tile = tile_idx / num_k_blocks;
                const uint32_t n_block_idx =
                    n_expert_tile % num_n_blocks;
                const uint32_t expert_idx =
                    n_expert_tile / num_n_blocks;
                if (__ldg(expert_counts + expert_idx) == 0)
                    continue;

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t chunk = 0;
                         chunk < kBuilderTmaChunks; ++chunk) {
                        tma::copy<
                            kBuilderTmaN / 2,
                            DGRAD_BLOCK_K, 0, int8_t>(
                            weights_map,
                            weight_load_barrier,
                            builder_weight_source +
                                chunk * kBuilderWeightChunkBytes,
                            n_block_idx * (kBuilderBlockN / 2) +
                                chunk * (kBuilderTmaN / 2),
                            expert_idx * source_rows_per_expert +
                                k_block_idx * DGRAD_BLOCK_K);
                        tma::copy<
                            kBuilderTmaN / kGranK,
                            DGRAD_BLOCK_K, 0, float>(
                            scales_map,
                            weight_load_barrier,
                            reinterpret_cast<float*>(
                                reinterpret_cast<uint8_t*>(
                                    builder_scale_source) +
                                chunk * kBuilderScaleChunkBytes),
                            n_block_idx *
                                    (kBuilderBlockN / kGranK) +
                                chunk *
                                    (kBuilderTmaN / kGranK),
                            expert_idx * source_rows_per_expert +
                                k_block_idx * DGRAD_BLOCK_K);
                    }
                    weight_load_barrier->arrive_and_expect_tx(
                        kBuilderWeightBytes + kBuilderScaleBytes);
                }
                weight_load_barrier->wait(weight_source_phase);
                __syncthreads();

                const uint32_t global_n_base =
                    n_block_idx * kBuilderBlockN;
                const uint32_t valid_n = cute::min(
                    kBuilderBlockN, output_mn - global_n_base);
                for (uint32_t local_n = warp_idx;
                     local_n < valid_n;
                     local_n += kWarpsPerCTA) {
                    const uint32_t global_n =
                        global_n_base + local_n;
                    const uint32_t local_k =
                        group_idx * 32 +
                        lane_in_group * kValuesPerLane;
                    const uint32_t source_chunk =
                        local_n / kBuilderTmaN;
                    const uint32_t source_n =
                        local_n % kBuilderTmaN;
                    float values[kValuesPerLane];
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kValuesPerLane; ++i) {
                        const uint8_t packed = static_cast<uint8_t>(
                            builder_weight_source[
                                source_chunk *
                                    kBuilderWeightChunkBytes +
                                (local_k + i) *
                                    (kBuilderTmaN / 2) +
                                source_n / 2]);
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
                        const float2 decoded = __half22float2(
                            *reinterpret_cast<__half2*>(&fp16x2));
                        const float scale =
                            builder_scale_source[
                                source_chunk *
                                    (kBuilderScaleChunkBytes /
                                     sizeof(float)) +
                                (local_k + i) *
                                    (kBuilderTmaN / kGranK) +
                                source_n / kGranK];
                        values[i] =
                            (global_n & 1u ? decoded.y : decoded.x) *
                            scale;
                    }
                    float amax = 0.0f;
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kValuesPerLane; ++i)
                        amax = cute::max(
                            amax, cute::abs(values[i]));
                    #pragma unroll
                    for (uint32_t offset = 4;
                         offset > 0; offset >>= 1)
                        amax = cute::max(
                            amax,
                            __shfl_xor_sync(
                                0xffffffff, amax, offset, 8));
                    float2 sf_pair;
                    float2 sf_inv_pair;
                    math::get_e4m3_sf_and_sf_inv(
                        make_float2(amax, 0.0f),
                        sf_pair, sf_inv_pair);
                    const auto quantized =
                        __nv_fp8x4_e4m3(make_float4(
                            values[0] * sf_inv_pair.x,
                            values[1] * sf_inv_pair.x,
                            values[2] * sf_inv_pair.x,
                            values[3] * sf_inv_pair.x));
                    const uint32_t global_k =
                        k_block_idx * DGRAD_BLOCK_K + local_k;
                    const uint64_t row =
                        static_cast<uint64_t>(expert_idx) * output_mn +
                        global_n;
                    *reinterpret_cast<uint32_t*>(
                        q + row * output_k + global_k) = quantized.__x;

                    const uint32_t scale_byte =
                        (*reinterpret_cast<const uint32_t*>(
                             &sf_pair.x)) >> 23;
                    uint32_t scale_word = 0;
                    #pragma unroll
                    for (uint32_t group = 0; group < 4; ++group)
                        scale_word |= __shfl_sync(
                            0xffffffff, scale_byte, group * 8)
                            << (group * 8);
                    if (lane_idx == 0) {
                        const uint32_t sf_n =
                            (global_n & ~127u) +
                            (global_n & 31u) * 4 +
                            ((global_n >> 5) & 3u);
                        sf[
                            static_cast<uint64_t>(expert_idx) *
                                output_mn * num_k_blocks +
                            sf_n +
                            static_cast<uint64_t>(k_block_idx) *
                                output_mn] = scale_word;
                    }
                }
                __syncthreads();
                weight_source_phase ^= 1;
            }
        };

        transform_operand(false);
        transform_operand(true);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        if (warp_idx == 0 && cute::elect_one_sync()) {
            Barrier::invalidate(
                reinterpret_cast<Barrier::ValueType const*>(
                    weight_load_barrier));
        }
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);
      }
    };

    const uint32_t residual_sf_rows =
        num_acts_rows / BLOCK_M * SF_BLOCK_M;

    // Quantize the reverse-dispatched W2 activation once into primary and
    // residual MXFP8 planes. The future grad-x pool holds values, while the
    // dead W13 dequant/output destination holds scales until W2 retires; using
    // grad-gate/up for scales would race SiTU derivative stores from W2's
    // epilogue. Both aliases are overwritten by their eventual consumers, so
    // no allocation survives the backward call.
    const auto build_w2_residual_acts_once = [&]() {
      if constexpr (kBuildW2ResidualActsOnce) {
        auto* primary = reinterpret_cast<uint8_t*>(grad_x_pool_output);
        auto* residual = primary +
            static_cast<uint64_t>(num_acts_rows) * kHidden;
        constexpr uint64_t kW13WeightAliasValues =
            static_cast<uint64_t>(kNumExperts) * kHidden *
            (2 * kIntermediateHidden);
        constexpr uint64_t kW13WeightAliasBytes =
            kW13WeightAliasValues + kW13WeightAliasValues / kGranK;
        auto* primary_sf = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(w13_dequant_scratch) +
            kW13WeightAliasBytes);
        auto* residual_sf = primary_sf +
            static_cast<uint64_t>(residual_sf_rows) * (kHidden / 128);
        const uint64_t num_sf_words =
            static_cast<uint64_t>(residual_sf_rows) * (kHidden / 128);
        for (uint64_t idx =
                 static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                 threadIdx.x;
             idx < num_sf_words;
             idx += static_cast<uint64_t>(kNumSMs) * kNumThreads) {
            primary_sf[idx] = 0x7f7f7f7fu;
            residual_sf[idx] = 0x7f7f7f7fu;
        }
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);

        constexpr uint32_t kValuesPerLane = 4;
        constexpr uint32_t kKBlocks = kHidden / 128;
        constexpr uint32_t kWarpsPerCTA = kNumThreads / 32;
        constexpr uint32_t kGlobalWarps = kNumSMs * kWarpsPerCTA;
        const uint32_t global_warp_idx =
            blockIdx.x * kWarpsPerCTA + warp_idx;
        const uint32_t group_idx = lane_idx / 8;
        const uint32_t lane_in_group = lane_idx % 8;
        uint32_t pool_block_offset = 0;
        #pragma unroll 1
        for (uint32_t expert_idx = 0;
             expert_idx < kNumExperts; ++expert_idx) {
            const uint32_t num_tokens = static_cast<uint32_t>(
                __ldg(expert_counts + expert_idx));
            const uint32_t num_blocks =
                math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t num_padded_tokens = num_blocks * BLOCK_M;
            const uint64_t num_tasks =
                static_cast<uint64_t>(num_padded_tokens) * kKBlocks;
            for (uint64_t task = global_warp_idx;
                 task < num_tasks;
                 task += kGlobalWarps) {
                const uint32_t row_in_expert = task / kKBlocks;
                const uint32_t k_block_idx =
                    task - static_cast<uint64_t>(row_in_expert) * kKBlocks;
                const uint32_t pool_block_idx =
                    pool_block_offset + row_in_expert / BLOCK_M;
                const uint32_t local_m = row_in_expert % BLOCK_M;
                const uint32_t pool_row =
                    pool_block_idx * BLOCK_M + local_m;
                const uint32_t global_k = k_block_idx * 128 +
                    group_idx * 32 + lane_in_group * kValuesPerLane;
                if (row_in_expert >= num_tokens) {
                    const uint64_t q_offset =
                        static_cast<uint64_t>(pool_row) * kHidden + global_k;
                    *reinterpret_cast<uint32_t*>(primary + q_offset) = 0;
                    *reinterpret_cast<uint32_t*>(residual + q_offset) = 0;
                    continue;
                }
                float values[kValuesPerLane];
                #pragma unroll
                for (uint32_t i = 0; i < kValuesPerLane; ++i) {
                    values[i] = static_cast<float>(grad_ye_output[
                        static_cast<uint64_t>(pool_row) * kHidden +
                        global_k + i]);
                }

                float primary_amax = 0.0f;
                #pragma unroll
                for (uint32_t i = 0; i < kValuesPerLane; ++i)
                    primary_amax = cute::max(
                        primary_amax, cute::abs(values[i]));
                #pragma unroll
                for (uint32_t offset = 4; offset > 0; offset >>= 1)
                    primary_amax = cute::max(
                        primary_amax,
                        __shfl_xor_sync(
                            0xffffffff, primary_amax, offset, 8));
                float2 primary_sf_pair;
                float2 primary_sf_inv_pair;
                math::get_e4m3_sf_and_sf_inv(
                    make_float2(primary_amax, 0.0f),
                    primary_sf_pair, primary_sf_inv_pair);
                const auto primary_q = __nv_fp8x4_e4m3(make_float4(
                    values[0] * primary_sf_inv_pair.x,
                    values[1] * primary_sf_inv_pair.x,
                    values[2] * primary_sf_inv_pair.x,
                    values[3] * primary_sf_inv_pair.x));
                const float4 primary_f = static_cast<float4>(primary_q);
                float residual_values[kValuesPerLane] = {
                    values[0] - primary_f.x * primary_sf_pair.x,
                    values[1] - primary_f.y * primary_sf_pair.x,
                    values[2] - primary_f.z * primary_sf_pair.x,
                    values[3] - primary_f.w * primary_sf_pair.x,
                };
                float residual_amax = 0.0f;
                #pragma unroll
                for (uint32_t i = 0; i < kValuesPerLane; ++i)
                    residual_amax = cute::max(
                        residual_amax, cute::abs(residual_values[i]));
                #pragma unroll
                for (uint32_t offset = 4; offset > 0; offset >>= 1)
                    residual_amax = cute::max(
                        residual_amax,
                        __shfl_xor_sync(
                            0xffffffff, residual_amax, offset, 8));
                float2 residual_sf_pair;
                float2 residual_sf_inv_pair;
                math::get_e4m3_sf_and_sf_inv(
                    make_float2(residual_amax, 0.0f),
                    residual_sf_pair, residual_sf_inv_pair);
                const auto residual_q = __nv_fp8x4_e4m3(make_float4(
                    residual_values[0] * residual_sf_inv_pair.x,
                    residual_values[1] * residual_sf_inv_pair.x,
                    residual_values[2] * residual_sf_inv_pair.x,
                    residual_values[3] * residual_sf_inv_pair.x));
                const uint64_t q_offset =
                    static_cast<uint64_t>(pool_row) * kHidden + global_k;
                *reinterpret_cast<uint32_t*>(primary + q_offset) =
                    primary_q.__x;
                *reinterpret_cast<uint32_t*>(residual + q_offset) =
                    residual_q.__x;

                const uint32_t primary_scale_byte =
                    (*reinterpret_cast<const uint32_t*>(
                         &primary_sf_pair.x)) >> 23;
                const uint32_t residual_scale_byte =
                    (*reinterpret_cast<const uint32_t*>(
                         &residual_sf_pair.x)) >> 23;
                uint32_t primary_scale_word = 0;
                uint32_t residual_scale_word = 0;
                #pragma unroll
                for (uint32_t group = 0; group < 4; ++group) {
                    primary_scale_word |= __shfl_sync(
                        0xffffffff, primary_scale_byte, group * 8)
                        << (group * 8);
                    residual_scale_word |= __shfl_sync(
                        0xffffffff, residual_scale_byte, group * 8)
                        << (group * 8);
                }
                if (lane_idx == 0) {
                    const uint32_t sf_m =
                        pool_block_idx * SF_BLOCK_M +
                        (local_m & ~127u) +
                        (local_m & 31u) * 4 +
                        ((local_m >> 5) & 3u);
                    const uint64_t sf_offset = sf_m +
                        static_cast<uint64_t>(k_block_idx) *
                            residual_sf_rows;
                    primary_sf[sf_offset] = primary_scale_word;
                    residual_sf[sf_offset] = residual_scale_word;
                }
            }
            pool_block_offset += num_blocks;
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);
      }
    };

    // W13 dgrad needs MXFP8, while wgrad still needs the exact BF16
    // derivative. In the early-W2 schedule, dW2 is already final and W2's
    // BF16 A/B operands are dead: grad_ye holds the primary plane plus both
    // scale planes, and grad_h/h_weighted holds the residual plane. Every
    // extent scales with num_acts_rows, unlike the legacy fixed dW2 alias.
    // Keeping grad_gate_up untouched also avoids a BF16 restore pass.
    const auto build_w13_residual_acts_once = [&]() {
      if constexpr (kBuildW13ResidualActsOnce) {
        constexpr uint32_t kWidth = 2 * kIntermediateHidden;
        constexpr uint32_t kKBlocks = kWidth / 128;
        constexpr uint32_t kValuesPerLane = 4;
        constexpr uint32_t kWarpsPerCTA = kNumThreads / 32;
        constexpr uint32_t kGlobalWarps = kNumSMs * kWarpsPerCTA;
        const uint64_t plane_bytes =
            static_cast<uint64_t>(num_acts_rows) * kWidth;
        auto* primary = reinterpret_cast<uint8_t*>(
            kEarlyW2Wgrad ? grad_ye_output : w2_dequant_scratch);
        auto* residual = kEarlyW2Wgrad
            ? reinterpret_cast<uint8_t*>(grad_h_output)
            : primary + plane_bytes;
        auto* primary_sf = reinterpret_cast<uint32_t*>(
            kEarlyW2Wgrad ? primary + plane_bytes
                          : residual + plane_bytes);
        auto* residual_sf = primary_sf +
            static_cast<uint64_t>(residual_sf_rows) * kKBlocks;
        const uint64_t num_sf_words =
            static_cast<uint64_t>(residual_sf_rows) * kKBlocks;
        for (uint64_t idx =
                 static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                 threadIdx.x;
             idx < num_sf_words;
             idx += static_cast<uint64_t>(kNumSMs) * kNumThreads) {
            primary_sf[idx] = 0x7f7f7f7fu;
            residual_sf[idx] = 0x7f7f7f7fu;
        }
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);

        const uint32_t group_idx = lane_idx / 8;
        const uint32_t lane_in_group = lane_idx % 8;
        const auto quantize_group = [&] (
            const float (&values)[kValuesPerLane],
            const uint32_t pool_row,
            const uint32_t k_block_idx,
            const uint32_t global_k) {
            float primary_amax = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i)
                primary_amax = cute::max(
                    primary_amax, cute::abs(values[i]));
            #pragma unroll
            for (uint32_t offset = 4; offset > 0; offset >>= 1)
                primary_amax = cute::max(
                    primary_amax,
                    __shfl_xor_sync(
                        0xffffffff, primary_amax, offset, 8));
            float2 primary_sf_pair;
            float2 primary_sf_inv_pair;
            math::get_e4m3_sf_and_sf_inv(
                make_float2(primary_amax, 0.0f),
                primary_sf_pair, primary_sf_inv_pair);
            const auto primary_q = __nv_fp8x4_e4m3(make_float4(
                values[0] * primary_sf_inv_pair.x,
                values[1] * primary_sf_inv_pair.x,
                values[2] * primary_sf_inv_pair.x,
                values[3] * primary_sf_inv_pair.x));
            const float4 primary_f = static_cast<float4>(primary_q);
            float residual_values[kValuesPerLane] = {
                values[0] - primary_f.x * primary_sf_pair.x,
                values[1] - primary_f.y * primary_sf_pair.x,
                values[2] - primary_f.z * primary_sf_pair.x,
                values[3] - primary_f.w * primary_sf_pair.x,
            };
            float residual_amax = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i)
                residual_amax = cute::max(
                    residual_amax, cute::abs(residual_values[i]));
            #pragma unroll
            for (uint32_t offset = 4; offset > 0; offset >>= 1)
                residual_amax = cute::max(
                    residual_amax,
                    __shfl_xor_sync(
                        0xffffffff, residual_amax, offset, 8));
            float2 residual_sf_pair;
            float2 residual_sf_inv_pair;
            math::get_e4m3_sf_and_sf_inv(
                make_float2(residual_amax, 0.0f),
                residual_sf_pair, residual_sf_inv_pair);
            const auto residual_q = __nv_fp8x4_e4m3(make_float4(
                residual_values[0] * residual_sf_inv_pair.x,
                residual_values[1] * residual_sf_inv_pair.x,
                residual_values[2] * residual_sf_inv_pair.x,
                residual_values[3] * residual_sf_inv_pair.x));
            const uint64_t q_offset =
                static_cast<uint64_t>(pool_row) * kWidth + global_k;
            *reinterpret_cast<uint32_t*>(primary + q_offset) =
                primary_q.__x;
            *reinterpret_cast<uint32_t*>(residual + q_offset) =
                residual_q.__x;

            const uint32_t primary_scale_byte =
                (*reinterpret_cast<const uint32_t*>(
                     &primary_sf_pair.x)) >> 23;
            const uint32_t residual_scale_byte =
                (*reinterpret_cast<const uint32_t*>(
                     &residual_sf_pair.x)) >> 23;
            uint32_t primary_scale_word = 0;
            uint32_t residual_scale_word = 0;
            #pragma unroll
            for (uint32_t group = 0; group < 4; ++group) {
                primary_scale_word |= __shfl_sync(
                    0xffffffff, primary_scale_byte, group * 8)
                    << (group * 8);
                residual_scale_word |= __shfl_sync(
                    0xffffffff, residual_scale_byte, group * 8)
                    << (group * 8);
            }
            if (lane_idx == 0) {
                const uint32_t pool_block_idx = pool_row / BLOCK_M;
                const uint32_t local_m = pool_row % BLOCK_M;
                const uint32_t sf_m =
                    pool_block_idx * SF_BLOCK_M +
                    (local_m & ~127u) +
                    (local_m & 31u) * 4 +
                    ((local_m >> 5) & 3u);
                const uint64_t sf_offset = sf_m +
                    static_cast<uint64_t>(k_block_idx) *
                        residual_sf_rows;
                primary_sf[sf_offset] = primary_scale_word;
                residual_sf[sf_offset] = residual_scale_word;
            }
        };

        if (inplace_gate_up_grad) {
            // Snapshot one interleaved SiTU row before any conventional-order
            // stores, then publish BF16 wgrad input and both MXFP8 dgrad
            // planes from that snapshot. This removes a full-capacity
            // deinterleave pass and a second HBM read of every active row.
            auto* row_staging =
                reinterpret_cast<cd_dtype_t*>(smem_gemm_base);
            uint32_t expert_idx = 0;
            uint32_t expert_pool_begin = 0;
            uint32_t expert_pool_end = 0;
            uint32_t expert_num_tokens = 0;
            for (uint32_t pool_row = blockIdx.x;
                 pool_row < num_acts_rows;
                 pool_row += kNumSMs) {
                while (
                    expert_idx < kNumExperts &&
                    pool_row >= expert_pool_end) {
                    expert_pool_begin = expert_pool_end;
                    expert_num_tokens = static_cast<uint32_t>(
                        __ldg(expert_counts + expert_idx));
                    expert_pool_end +=
                        math::ceil_div(expert_num_tokens, BLOCK_M) * BLOCK_M;
                    ++expert_idx;
                }
                const bool valid_row =
                    expert_idx <= kNumExperts &&
                    pool_row - expert_pool_begin < expert_num_tokens;
                const uint64_t row_base =
                    static_cast<uint64_t>(pool_row) * kWidth;
                if (valid_row) {
                    for (uint32_t col = threadIdx.x;
                         col < kWidth;
                         col += kNumThreads)
                        row_staging[col] =
                            grad_gate_up_output[row_base + col];
                }
                __syncthreads();

                for (uint32_t k_block_idx = warp_idx;
                     k_block_idx < kKBlocks;
                     k_block_idx += kWarpsPerCTA) {
                    const uint32_t global_k = k_block_idx * 128 +
                        group_idx * 32 + lane_in_group * kValuesPerLane;
                    const uint64_t q_offset = row_base + global_k;
                    if (!valid_row) {
                        *reinterpret_cast<uint32_t*>(primary + q_offset) = 0;
                        *reinterpret_cast<uint32_t*>(residual + q_offset) = 0;
                        continue;
                    }
                    float values[kValuesPerLane];
                    #pragma unroll
                    for (uint32_t i = 0; i < kValuesPerLane; ++i) {
                        const uint32_t conventional_col = global_k + i;
                        const uint32_t hidden_col =
                            conventional_col < kIntermediateHidden
                            ? conventional_col
                            : conventional_col - kIntermediateHidden;
                        const uint32_t source_col =
                            (hidden_col / 8) * 16 +
                            (conventional_col >= kIntermediateHidden
                                 ? 8
                                 : 0) +
                            (hidden_col & 7);
                        values[i] = static_cast<float>(
                            row_staging[source_col]);
                        grad_gate_up_output[
                            row_base + conventional_col] =
                            cd_dtype_t(values[i]);
                    }
                    quantize_group(
                        values, pool_row, k_block_idx, global_k);
                }
                __syncthreads();
            }
        } else {
            const uint32_t global_warp_idx =
                blockIdx.x * kWarpsPerCTA + warp_idx;
            uint32_t pool_block_offset = 0;
            #pragma unroll 1
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens = static_cast<uint32_t>(
                    __ldg(expert_counts + expert_idx));
                const uint32_t num_blocks =
                    math::ceil_div(num_tokens, BLOCK_M);
                const uint32_t num_padded_tokens = num_blocks * BLOCK_M;
                const uint64_t num_tasks =
                    static_cast<uint64_t>(num_padded_tokens) * kKBlocks;
                for (uint64_t task = global_warp_idx;
                     task < num_tasks;
                     task += kGlobalWarps) {
                    const uint32_t row_in_expert = task / kKBlocks;
                    const uint32_t k_block_idx =
                        task -
                        static_cast<uint64_t>(row_in_expert) * kKBlocks;
                    const uint32_t pool_block_idx =
                        pool_block_offset + row_in_expert / BLOCK_M;
                    const uint32_t local_m = row_in_expert % BLOCK_M;
                    const uint32_t pool_row =
                        pool_block_idx * BLOCK_M + local_m;
                    const uint32_t global_k = k_block_idx * 128 +
                        group_idx * 32 + lane_in_group * kValuesPerLane;
                    const uint64_t q_offset =
                        static_cast<uint64_t>(pool_row) * kWidth + global_k;
                    if (row_in_expert >= num_tokens) {
                        *reinterpret_cast<uint32_t*>(primary + q_offset) = 0;
                        *reinterpret_cast<uint32_t*>(residual + q_offset) = 0;
                        continue;
                    }
                    float values[kValuesPerLane];
                    #pragma unroll
                    for (uint32_t i = 0; i < kValuesPerLane; ++i)
                        values[i] = static_cast<float>(
                            grad_gate_up_output[
                                static_cast<uint64_t>(pool_row) * kWidth +
                                global_k + i]);
                    quantize_group(
                        values, pool_row, k_block_idx, global_k);
                }
                pool_block_offset += num_blocks;
            }
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);
      }
    };

    const auto restore_w13_bf16_acts_once = [&]() {
      if constexpr (kBuildW13ResidualActsOnce) {
        if (!inplace_gate_up_grad)
            return;
        full_grid_phase_barrier(kTraceSiteCount);
        constexpr uint32_t kWidth = 2 * kIntermediateHidden;
        constexpr uint32_t kKBlocks = kWidth / 128;
        auto* q_rows = reinterpret_cast<const uint8_t*>(gate_up_output);
        const auto* primary_sf =
            reinterpret_cast<const uint32_t*>(w2_dequant_scratch);
        const auto* residual_sf = primary_sf +
            static_cast<uint64_t>(residual_sf_rows) * kKBlocks;
        auto* row_staging = reinterpret_cast<uint8_t*>(smem_buffer);
        for (uint32_t pool_row = blockIdx.x;
             pool_row < num_acts_rows;
             pool_row += kNumSMs) {
            const uint64_t row_base =
                static_cast<uint64_t>(pool_row) * (2 * kWidth);
            for (uint32_t idx = threadIdx.x;
                 idx < 2 * kWidth;
                 idx += kNumThreads)
                row_staging[idx] = q_rows[row_base + idx];
            __syncthreads();
            const uint32_t pool_block_idx = pool_row / BLOCK_M;
            const uint32_t local_m = pool_row % BLOCK_M;
            const uint32_t sf_m =
                pool_block_idx * SF_BLOCK_M +
                (local_m & ~127u) +
                (local_m & 31u) * 4 +
                ((local_m >> 5) & 3u);
            for (uint32_t col = threadIdx.x;
                 col < kWidth;
                 col += kNumThreads) {
                const uint32_t k_block_idx = col / 128;
                const uint32_t sf_byte_idx = (col / 32) & 3u;
                const uint64_t sf_offset = sf_m +
                    static_cast<uint64_t>(k_block_idx) * residual_sf_rows;
                const uint32_t primary_exponent =
                    (primary_sf[sf_offset] >> (sf_byte_idx * 8)) & 0xffu;
                const uint32_t residual_exponent =
                    (residual_sf[sf_offset] >> (sf_byte_idx * 8)) & 0xffu;
                const uint32_t primary_bits = primary_exponent << 23;
                const uint32_t residual_bits = residual_exponent << 23;
                const float primary_scale =
                    *reinterpret_cast<const float*>(&primary_bits);
                const float residual_scale =
                    *reinterpret_cast<const float*>(&residual_bits);
                const auto* primary_q =
                    reinterpret_cast<const residual_dgrad_dtype_t*>(
                        row_staging);
                const auto* residual_q =
                    reinterpret_cast<const residual_dgrad_dtype_t*>(
                        row_staging + kWidth);
                grad_gate_up_output[
                    static_cast<uint64_t>(pool_row) * kWidth + col] =
                    cd_dtype_t(
                        static_cast<float>(primary_q[col]) * primary_scale +
                        static_cast<float>(residual_q[col]) * residual_scale);
            }
            __syncthreads();
        }
        __threadfence();
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount);
      }
    };

    // Materialize one canonical BF16 weight matrix at the phase boundary
    // where its backing storage is dead.  The caller can therefore alias W2
    // to the symmetric combine planes and W13 to the consumed gate/up pool
    // without extending peak memory.  Tile epochs let a cluster start the
    // following UMMA phase while slower clusters finish unrelated tiles.
    const auto dequant_noninline_weights_once =
        [&](const bool dequant_w13,
            const bool phase_ordered_dequant,
            const bool initialize_barriers) {
      if constexpr (
          !kBF16Mode && !kInlineWeightDequant &&
          !kResidualMXFP8Dgrad) {
        auto* dequant_barrier =
            phase_ordered_dequant
            ? dequant_barriers + static_cast<uint32_t>(dequant_w13)
            : full_barriers[static_cast<uint32_t>(dequant_w13)];
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
        if (initialize_barriers) {
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0 && cute::elect_one_sync()) {
                if (phase_ordered_dequant) {
                    dequant_barrier->init(1);
                } else {
                    // Preserve the original single-chunk fast path: W2 and
                    // W13 share one initialization and then hand these slots
                    // directly to the main UMMA pipeline for reinitialization.
                    full_barriers[0]->init(1);
                    if constexpr (kCompileW13Dgrad)
                        full_barriers[1]->init(1);
                }
                cutlass::arch::fence_barrier_init();
            }
            comm::cluster_sync_with_relaxed_arrive();
        }
        if (!dequant_w13) {
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
            if (__ldg(expert_counts + expert_idx) == 0)
                continue;
            const uint32_t global_k_base =
                k_tile_idx * kDequantTileK;
            const uint32_t global_n_base =
                n_tile_idx * kDequantTileN;

            if (warp_idx == 0 && cute::elect_one_sync()) {
                tma::copy<
                    kDequantTileN / 2, kDequantTileK, 0,
                    int8_t>(
                    &tensor_map_w2_weights,
                    dequant_barrier, dequant_weights,
                    global_n_base / 2,
                    expert_idx * kHidden + global_k_base);
                tma::copy<
                    kDequantSFsPerK, kDequantTileK, 0,
                    float>(
                    &tensor_map_w2_scales,
                    dequant_barrier, dequant_scales,
                    global_n_base / 32,
                    expert_idx * kHidden + global_k_base);
                dequant_barrier->arrive_and_expect_tx(
                    kDequantWeightBytes +
                    kDequantScaleBytes);
            }
            dequant_barrier->wait(dequant_phase);
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

        } else if constexpr (kCompileW13Dgrad) {
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
                if (__ldg(expert_counts + expert_idx) == 0)
                    continue;
                const uint32_t global_k_base =
                    k_tile_idx * kW13DequantTileK;
                const uint32_t global_n_base =
                    n_tile_idx * kW13DequantTileN;

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    tma::copy<
                        kW13DequantTileN / 2,
                        kW13DequantTileK, 0, int8_t>(
                        &tensor_map_w13_weights,
                        dequant_barrier,
                        dequant_weights,
                        global_n_base / 2,
                        expert_idx *
                                (2 * kIntermediateHidden) +
                            global_k_base);
                    tma::copy<
                        kW13DequantSFsPerK,
                        kW13DequantTileK, 0, float>(
                        &tensor_map_w13_scales,
                        dequant_barrier,
                        dequant_scales,
                        global_n_base / 32,
                        expert_idx *
                                (2 * kIntermediateHidden) +
                            global_k_base);
                    dequant_barrier->arrive_and_expect_tx(
                        kW13DequantWeightBytes +
                        kW13DequantScaleBytes);
                }
                dequant_barrier->wait(
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
        if (phase_ordered_dequant) {
            // A delayed phase can write into symmetric or formerly consumed
            // pool storage. Publish generic stores to TMA's async proxy before
            // the grid-wide phase handoff. The original early path instead
            // relies on its existing tile epochs and pipeline initialization.
            asm volatile(
                "fence.proxy.async.global;" ::: "memory");
            comm::cluster_sync_with_relaxed_arrive();
        }
      }
    };
    // Match MoK's dependency-ordered CLC schedule: all replay/transform
    // producers are resident before a dependent dgrad consumer can run.
    if constexpr (kPrefixedResidualWeightCache)
        build_residual_weights_tiled_once();
    constexpr bool phase_ordered_w2_dequant =
        kPhaseOrderedWeightDequant;
    constexpr bool phase_ordered_w13_dequant =
        kPhaseOrderedWeightDequant;
    constexpr bool kRouteGradBeforeW2 =
        kComputeRouteGrad && !kInputsPrepared &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        (phase_ordered_w2_dequant || kBuildW2ResidualActsOnce);
    if constexpr (!phase_ordered_w2_dequant) {
        dequant_noninline_weights_once(false, false, true);
        if constexpr (!phase_ordered_w13_dequant)
            dequant_noninline_weights_once(true, false, false);
    } else if constexpr (!phase_ordered_w13_dequant) {
        dequant_noninline_weights_once(true, false, true);
    }
    trace_begin(1);
    comm::cluster_sync_with_relaxed_arrive();
    trace_end(1);
    if (warp_idx == 0 && cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(
                kNumDgradFullBarrierArrivals);
            empty_barriers[i]->init(1);
        }
        if constexpr (kResidualMXFP8Dgrad) {
            weight_load_barrier->init(1);
            residual_mma_barrier->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumDispatchBarriers; ++i)
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
        #pragma unroll 1
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
    // Kimi-K3 training asks MegaMoE Forward to publish the exact BF16 SiTU
    // preactivation into gate_up_output.  Preserve the persistent backward
    // role/register layout, but make its otherwise redundant W13 replay
    // schedule empty.  All later reverse dispatch, route-gradient, dgrad, and
    // wgrad phases still execute on the same clustered kernel launch.
    const auto for_each_replay_block = [&](const auto& func) {
        if constexpr (!kGateUpPrepared)
            for_each_block(func);
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
        for_each_replay_block([&](const uint32_t&, const uint32_t& pool_block_offset,
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
        for_each_replay_block([&](const uint32_t& expert_idx, const uint32_t&,
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

            for_each_replay_block([&](const uint32_t&, const uint32_t&,
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
        const uint32_t allocated_tmem =
            ptx::ld_shared(tmem_ptr_in_smem);
        if (allocated_tmem != 0) {
            if (lane_idx == 0) {
                printf(
                    "K3 parent TMEM allocation mismatch rank=%u sm=%u "
                    "ptr=%u\n",
                    backward_sym_buffer.rank_idx,
                    blockIdx.x,
                    allocated_tmem);
            }
            asm volatile("trap;");
        }
        const uint32_t epilogue_warp_idx = warp_idx - 4;
        uint32_t current_iter = 0;
        uint32_t tma_stage_idx = 0;

        for_each_replay_block([&](const uint32_t&, const uint32_t& pool_block_offset,
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
            uint32_t pool_block_offset = 0;

            // Keep source-row dispatch compact for K3's 112 local experts.
            #pragma unroll 1
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
                            dispatch_pull_mbarrier_phase);
                        ptx::tma_store_1d(
                            grad_y_unweighted_output +
                                static_cast<uint64_t>(
                                    pool_row) *
                                    kHidden,
                            pull_buffer,
                            kHidden * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();

                        if constexpr (!kBF16Mode && kExactSourceX) {
                          if (!late_exact_source_x) {
                            // W13 wgrad requires the original BF16 source
                            // activation. Reuse the grad-y TMA staging buffer
                            // after its store drains instead of issuing one
                            // remote scalar load per element.
                            const auto* remote_x =
                                backward_sym_buffer.map(
                                    backward_x +
                                        static_cast<uint64_t>(
                                            metadata.token_idx) *
                                            kHidden,
                                    metadata.rank_idx);
                            ptx::tma_load_1d(
                                pull_buffer, remote_x,
                                pull_mbarrier,
                                kHidden * sizeof(cd_dtype_t));
                            ptx::mbarrier_arrive_and_set_tx(
                                pull_mbarrier,
                                kHidden * sizeof(cd_dtype_t));
                            ptx::mbarrier_wait_and_flip_phase(
                                pull_mbarrier,
                                dispatch_pull_mbarrier_phase);
                            ptx::tma_store_1d(
                                x_pool_output +
                                    static_cast<uint64_t>(pool_row) *
                                    kHidden,
                                pull_buffer,
                                kHidden * sizeof(cd_dtype_t));
                            cute::tma_store_arrive();
                            ptx::tma_store_wait<0>();
                          }
                        }
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
          #pragma unroll 1
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
                        } else if constexpr (kExactSourceX) {
                            // Active exact-BF16 rows are TMA-dispatched above.
                            // These warps retain only the padding clear needed
                            // by the K-rounded W13 wgrad mainloop.
                            continue;
                        } else {
                            if (pool_row >= num_acts_rows) {
                                asm volatile("trap;");
                            }
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
            #pragma unroll 1
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
        const auto clear_and_publish_combine_planes = [&]() {
          if constexpr (
              kDirectRemoteGradX &&
              kCombineOrderMode ==
                  CombineOrderMode::FixedTopK) {
            // FixedTopK consumes every physical slot, including invalid
            // routes. Clear all slot planes only after all grad-y pulls have
            // completed, then publish the clear before remote stores.
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
                combine_buffer[linear] = cd_dtype_t(0.0f);
            }
            if constexpr (kNumRanks > 1) {
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
        };

        const auto prepare_direct_grad_x_planes = [&]() {
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
            clear_and_publish_combine_planes();
          }
        };
        // Exact source X is staged in combine plane one. When x_pool aliases
        // grad-y, retain that plane until the W2/route readers release grad-y;
        // the late TMA pull below drains it immediately before W13 dgrad starts
        // publishing direct grad-x into the same symmetric planes.
        if (!late_exact_source_x)
            prepare_direct_grad_x_planes();
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
            constexpr uint32_t kNumSchedulePasses =
                kOnDemandResidualWeightCache ? 2 : 1;
            // MoK publishes every weight producer before any dependent CLC
            // consumer. Preserve that dependency order here: pass zero owns
            // M=0 and converts each (expert,N,K) tile exactly once; pass one
            // schedules every remaining M tile against the released cache.
            #pragma unroll 1
            for (uint32_t schedule_pass = 0;
                 schedule_pass < kNumSchedulePasses;
                 ++schedule_pass) {
                uint32_t next_assigned_block = blockIdx.x;
                uint32_t global_block = 0;
                uint32_t pool_block_offset = 0;
                // Cloning this full scheduler body for all 112 local K3
                // experts produces a multi-megabyte instruction image and
                // trashes L0/L1 I$.
                #pragma unroll 1
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(expert_counts + expert_idx));
                    const uint32_t num_m_blocks =
                        math::ceil_div(num_tokens, BLOCK_M);
                    const uint32_t first_m_block =
                        kOnDemandResidualWeightCache
                            ? schedule_pass
                            : 0;
                    const uint32_t scheduled_m_blocks =
                        schedule_pass == 0
                            ? cute::min(num_m_blocks, 1u)
                            : num_m_blocks -
                                  cute::min(num_m_blocks, 1u);
                    const uint32_t expert_blocks =
                        (kOnDemandResidualWeightCache
                             ? scheduled_m_blocks
                             : num_m_blocks) *
                        kNumDgradBlockNs;
                    const uint32_t expert_end =
                        global_block + expert_blocks;

                    while (next_assigned_block < global_block)
                        next_assigned_block += kNumSMs;
                    while (next_assigned_block < expert_end) {
                        const uint32_t local_block =
                            next_assigned_block - global_block;
                        const uint32_t m_block_idx =
                            first_m_block +
                            local_block / kNumDgradBlockNs;
                        const uint32_t n_block_idx =
                            local_block % kNumDgradBlockNs;
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
            }
        };

        trace_begin(9);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(9);

        trace_begin(10);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(10);

        if constexpr (
            kComputeRouteGrad && !kInputsPrepared &&
            kRouteWeightMode ==
                RouteWeightMode::PostDown) {
          if constexpr (kRouteGradBeforeW2) {
            // POST_DOWN route gradients depend only on reverse-dispatched
            // grad-y and the saved unweighted down output. Compute them before
            // W2 dgrad so the saved down pool can become W2's dequant
            // workspace.
            // Nontrivial pools use one warp per route in the existing reducer;
            // retain its four-column lane accumulation and shuffle tree.
            constexpr uint32_t kRouteGroupThreads = 32;
            constexpr uint32_t kRouteGroupsPerCTA =
                kNumThreads / kRouteGroupThreads;
            const uint32_t route_group_idx =
                threadIdx.x / kRouteGroupThreads;
            const uint32_t route_group_lane_idx =
                threadIdx.x & (kRouteGroupThreads - 1);
            const uint32_t global_route_group =
                blockIdx.x * kRouteGroupsPerCTA +
                route_group_idx;
            const uint32_t num_route_groups =
                kNumSMs * kRouteGroupsPerCTA;
            uint32_t route_pool_block_offset = 0;
            #pragma unroll 1
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
                    float lane_sums[4] = {
                        0.0f, 0.0f, 0.0f, 0.0f};
                    for (uint32_t col_base =
                             route_group_lane_idx * 4;
                         col_base < kHidden;
                         col_base +=
                             kRouteGroupThreads * 4) {
                        #pragma unroll
                        for (uint32_t i = 0; i < 4; ++i) {
                            const uint32_t col = col_base + i;
                            const float grad_y =
                                static_cast<float>(
                                    grad_y_unweighted_output[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            kHidden + col]);
                            const float down =
                                static_cast<float>(
                                    down_unweighted_output[
                                        static_cast<uint64_t>(
                                            pool_row) *
                                            kHidden + col]);
                            lane_sums[i] = __fadd_rn(
                                lane_sums[i],
                                __fmul_rn(grad_y, down));
                        }
                    }
                    float grad_route = __fadd_rn(
                        __fadd_rn(lane_sums[0], lane_sums[1]),
                        lane_sums[2]);
                    grad_route = __fadd_rn(
                        grad_route, lane_sums[3]);
                    #pragma unroll
                    for (uint32_t offset = 16;
                         offset > 0; offset >>= 1) {
                        grad_route = __fadd_rn(
                            grad_route,
                            __shfl_down_sync(
                                0xffffffff,
                                grad_route, offset));
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
                    __syncwarp();
                }
                route_pool_block_offset +=
                    math::ceil_div(num_tokens, BLOCK_M);
            }
            // No dequant writer may reuse the saved down pool until every CTA
            // has finished its last route dot.
            full_grid_phase_barrier(kTraceSiteCount);
          }
        }

        // The route-gradient dot is complete. A multichunk caller may now
        // reuse the saved down pool as W2's BF16 dequant workspace;
        // single-chunk calls keep using dW2 storage.
        if constexpr (phase_ordered_w2_dequant) {
            dequant_noninline_weights_once(false, true, true);
            full_grid_phase_barrier(kTraceSiteCount);
        }
        build_w2_residual_acts_once();

        // Reinitialize the drained pipelines in-place. The replay roles have
        // joined above, so retire the previous mbarrier objects before
        // resetting their phase. A direct mbarrier.init on a still-valid
        // object is undefined even when the prepared-gate/up replay schedule
        // is empty. The FP32 accumulator columns are phase-aliased; dgrad does
        // not need the SFA/SFB columns.
        if (warp_idx == 0 && cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        full_barriers[i]));
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        empty_barriers[i]));
            }
            if constexpr (kResidualMXFP8Dgrad) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        weight_load_barrier));
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        residual_mma_barrier));
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        tmem_full_barriers[i]));
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        tmem_empty_barriers[i]));
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                full_barriers[i]->init(
                    kNumDgradFullBarrierArrivals);
                empty_barriers[i]->init(1);
            }
            if constexpr (kResidualMXFP8Dgrad) {
                weight_load_barrier->init(1);
                residual_mma_barrier->init(1);
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
        if (
            warp_idx == 0 ||
            (kResidualMXFP8Dgrad &&
             (!kBuildW2ResidualActsOnce ||
              kOnDemandResidualWeightCache) &&
             (warp_idx == 1 ||
              (warp_idx >= 3 && warp_idx < 8)))) {
            // The legacy path uses a 2-SM BF16 TMA load. Residual MXFP8 uses
            // all seven non-MMA producer warps per CTA. They quantize BF16
            // grad-y into primary/residual A operands and re-encode the
            // transposed MXFP4 weight tile exactly into the MXFP8 B operand.
            // Both operands are produced inside the persistent pipeline, so
            // no full-size transformed-weight allocation or extra launch is
            // required.
            uint32_t weight_source_phase = 0;
            for_each_dgrad_block(
                [&](const uint32_t& expert_idx,
                    const uint32_t& pool_block_offset,
                    const uint32_t& m_block_idx,
                    const uint32_t& n_block_idx,
                    const uint32_t& valid_m) {
                    const uint32_t pool_block_idx =
                        pool_block_offset + m_block_idx;
                    #pragma unroll 1
                    for (uint32_t k_block_idx = 0;
                         k_block_idx < kHidden / DGRAD_BLOCK_K;
                         advance_pipeline(k_block_idx)) {
                        const uint32_t producer_warp_idx =
                            warp_idx == 0
                            ? 0
                            : (warp_idx == 1 ? 1 : warp_idx - 1);
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        uint32_t m_idx = pool_block_idx * BLOCK_M;
                        if (!is_leader_cta)
                            m_idx += math::align(valid_m, 16u) / 2;
                        if constexpr (!kResidualMXFP8Dgrad) {
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
                        } else if constexpr (
                            kBuildW2ResidualActsOnce &&
                            !kOnDemandResidualWeightCache) {
                            if (producer_warp_idx == 0 &&
                                cute::elect_one_sync()) {
                                tma::copy<
                                    DGRAD_BLOCK_K, LOAD_BLOCK_M,
                                    128, residual_dgrad_dtype_t>(
                                    &tensor_map_acts,
                                    full_barriers[stage_idx],
                                    reinterpret_cast<
                                        residual_dgrad_dtype_t*>(
                                        smem_dgrad_a[stage_idx]),
                                    k_block_idx * DGRAD_BLOCK_K,
                                    m_idx, 2);
                                tma::copy<
                                    DGRAD_BLOCK_K, LOAD_BLOCK_M,
                                    128, residual_dgrad_dtype_t>(
                                    &tensor_map_acts,
                                    full_barriers[stage_idx],
                                    smem_dgrad_a_residual[stage_idx],
                                    k_block_idx * DGRAD_BLOCK_K,
                                    num_acts_rows + m_idx, 2);
                                tma::copy<SF_BLOCK_M, 1, 0>(
                                    &tensor_map_acts_sf,
                                    full_barriers[stage_idx],
                                    smem_sfa[stage_idx],
                                    pool_block_idx * SF_BLOCK_M,
                                    k_block_idx, 2);
                                tma::copy<SF_BLOCK_M, 1, 0>(
                                    &tensor_map_acts_sf,
                                    full_barriers[stage_idx],
                                    smem_dgrad_sfa_residual[stage_idx],
                                    pool_block_idx * SF_BLOCK_M,
                                    kHidden / (kGranK * 4) +
                                        k_block_idx, 2);
                                if (is_leader_cta) {
                                    full_barriers[stage_idx]
                                        ->arrive_and_expect_tx(
                                            2 *
                                            (SMEM_A_SIZE_PER_STAGE +
                                             SMEM_RESIDUAL_A_SIZE_PER_STAGE +
                                             SMEM_SFA_SIZE_PER_STAGE +
                                             SMEM_RESIDUAL_SFA_SIZE_PER_STAGE));
                                } else {
                                    full_barriers[stage_idx]->arrive(0u);
                                }
                            }
                        } else {
                            constexpr uint32_t kQuantProducerWarps =
                                kNumResidualProducerWarps;
                            constexpr uint32_t kValuesPerLane = 4;
                            const uint32_t group_idx = lane_idx / 8;
                            const uint32_t lane_in_group = lane_idx % 8;
                            if constexpr (kBuildW2ResidualActsOnce) {
                                if (producer_warp_idx == 0 &&
                                    cute::elect_one_sync()) {
                                    tma::copy<
                                        DGRAD_BLOCK_K, LOAD_BLOCK_M,
                                        128, residual_dgrad_dtype_t>(
                                        &tensor_map_acts,
                                        full_barriers[stage_idx],
                                        reinterpret_cast<
                                            residual_dgrad_dtype_t*>(
                                            smem_dgrad_a[stage_idx]),
                                        k_block_idx * DGRAD_BLOCK_K,
                                        m_idx, 2);
                                    tma::copy<
                                        DGRAD_BLOCK_K, LOAD_BLOCK_M,
                                        128, residual_dgrad_dtype_t>(
                                        &tensor_map_acts,
                                        full_barriers[stage_idx],
                                        smem_dgrad_a_residual[stage_idx],
                                        k_block_idx * DGRAD_BLOCK_K,
                                        num_acts_rows + m_idx, 2);
                                    tma::copy<SF_BLOCK_M, 1, 0>(
                                        &tensor_map_acts_sf,
                                        full_barriers[stage_idx],
                                        smem_sfa[stage_idx],
                                        pool_block_idx * SF_BLOCK_M,
                                        k_block_idx, 2);
                                    tma::copy<SF_BLOCK_M, 1, 0>(
                                        &tensor_map_acts_sf,
                                        full_barriers[stage_idx],
                                        smem_dgrad_sfa_residual[stage_idx],
                                        pool_block_idx * SF_BLOCK_M,
                                        kHidden / (kGranK * 4) +
                                            k_block_idx, 2);
                                    if (is_leader_cta) {
                                        full_barriers[stage_idx]
                                            ->arrive_and_expect_tx(
                                                2 *
                                                (SMEM_A_SIZE_PER_STAGE +
                                                 SMEM_RESIDUAL_A_SIZE_PER_STAGE +
                                                 SMEM_SFA_SIZE_PER_STAGE +
                                                 SMEM_RESIDUAL_SFA_SIZE_PER_STAGE));
                                    } else {
                                        full_barriers[stage_idx]->arrive(0u);
                                    }
                                }
                            }
                            bool build_weight_tile =
                                !kPrefixedResidualWeightCache;
                            const uint32_t weight_tile_idx =
                                (expert_idx *
                                     (kHidden / DGRAD_BLOCK_K) +
                                 k_block_idx) *
                                    kNumDgradBlockNs +
                                n_block_idx;
                            if constexpr (kOnDemandResidualWeightCache) {
                                // A 2-SM TMA transaction must be issued by
                                // both CTAs in the cluster. Elect the first M
                                // wave—not one CTA—as the resident producer.
                                // Every later M wave consumes the epoch-published
                                // cache tile, matching MoK's deterministic
                                // producer/consumer schedule.
                                build_weight_tile = m_block_idx == 0;
                                if (producer_warp_idx == 0 && lane_idx == 0 &&
                                    !build_weight_tile) {
                                    auto* state =
                                        weight_tile_states + weight_tile_idx;
                                    const uint64_t cache_wait_start = clock64();
                                    while (ptx::ld_acq(state) != launch_epoch) {
                                        if (clock64() - cache_wait_start >
                                            4000000000ull) {
                                            printf(
                                                "K3 W2 cache wait timeout rank=%u sm=%u tile=%u state=%u ready=%u\n",
                                                backward_sym_buffer.rank_idx,
                                                blockIdx.x,
                                                weight_tile_idx,
                                                ptx::ld_acq(state),
                                                launch_epoch);
                                            asm volatile("trap;");
                                        }
                                    }
                                }
                                cutlass::arch::NamedBarrier::sync(
                                    kNumResidualProducerThreads,
                                    kResidualWeightCacheBarrier);
                            }
                            if constexpr (kInlineResidualMXFP8Dgrad) {
                                if (producer_warp_idx == 0 &&
                                    build_weight_tile &&
                                    cute::elect_one_sync()) {
                                    tma::copy<
                                        LOAD_BLOCK_N / 2,
                                        DGRAD_BLOCK_K, 0, int8_t>(
                                        &tensor_map_w2_weights,
                                        weight_load_barrier,
                                        smem_dgrad_weight_source,
                                        n_block_idx * (LOAD_BLOCK_N / 2),
                                        expert_idx * kHidden +
                                            k_block_idx * DGRAD_BLOCK_K);
                                    tma::copy<
                                        LOAD_BLOCK_N / kGranK,
                                        DGRAD_BLOCK_K, 0, float>(
                                        &tensor_map_w2_scales,
                                        weight_load_barrier,
                                        smem_dgrad_weight_scale_source,
                                        n_block_idx *
                                            (LOAD_BLOCK_N / kGranK),
                                        expert_idx * kHidden +
                                            k_block_idx * DGRAD_BLOCK_K);
                                    weight_load_barrier
                                        ->arrive_and_expect_tx(
                                            SMEM_DGRAD_WEIGHT_SOURCE_SIZE +
                                            SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE);
                                }
                            }
                            if constexpr (!kBuildW2ResidualActsOnce) {
                            const uint32_t effective_m =
                                math::align(valid_m, 16u);
                            const uint32_t local_valid_m = effective_m / 2;
                            auto* primary_bytes =
                                reinterpret_cast<uint8_t*>(
                                    smem_dgrad_a[stage_idx]);
                            auto* residual_bytes =
                                reinterpret_cast<uint8_t*>(
                                    smem_dgrad_a_residual[stage_idx]);
                            // UTCCP always transfers the complete aligned scale
                            // plane. Seed padding rows with scale 1 so a partial
                            // M tile cannot expose stale shared-memory bytes.
                            const uint32_t producer_thread_idx =
                                producer_warp_idx * 32 + lane_idx;
                            for (uint32_t sf_idx = producer_thread_idx;
                                 sf_idx < SF_BLOCK_M;
                                 sf_idx +=
                                     kNumResidualProducerThreads) {
                                smem_sfa[stage_idx][sf_idx] =
                                    0x7f7f7f7fu;
                                smem_dgrad_sfa_residual[
                                    stage_idx][sf_idx] =
                                    0x7f7f7f7fu;
                            }
                            cutlass::arch::NamedBarrier::sync(
                                kNumResidualProducerThreads,
                                kInlineWeightProducerBarrier);
                            const uint32_t num_quant_rows =
                                is_leader_cta ? effective_m : local_valid_m;
                            for (uint32_t local_m = producer_warp_idx;
                                 local_m < num_quant_rows;
                                 local_m += kQuantProducerWarps) {
                                const uint32_t full_m =
                                    local_m +
                                    (is_leader_cta ? 0 : local_valid_m);
                                const bool valid_row =
                                    full_m < valid_m;
                                const bool store_data =
                                    is_leader_cta
                                    ? full_m < local_valid_m
                                    : true;
                                const uint32_t local_store_m =
                                    is_leader_cta ? full_m : local_m;
                                const uint32_t global_k =
                                    k_block_idx * DGRAD_BLOCK_K +
                                    group_idx * 32 +
                                    lane_in_group * kValuesPerLane;
                                float values[kValuesPerLane];
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kValuesPerLane; ++i) {
                                    values[i] = valid_row
                                        ? static_cast<float>(
                                              grad_ye_output[
                                                  static_cast<uint64_t>(
                                                      pool_block_idx * BLOCK_M +
                                                      full_m) *
                                                      kHidden +
                                                  global_k + i])
                                        : 0.0f;
                                }

                                float primary_amax = 0.0f;
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kValuesPerLane; ++i)
                                    primary_amax = cute::max(
                                        primary_amax,
                                        cute::abs(values[i]));
                                #pragma unroll
                                for (uint32_t offset = 4;
                                     offset > 0; offset >>= 1)
                                    primary_amax = cute::max(
                                        primary_amax,
                                        __shfl_xor_sync(
                                            0xffffffff, primary_amax,
                                            offset, 8));
                                float2 primary_sf_pair;
                                float2 primary_sf_inv_pair;
                                math::get_e4m3_sf_and_sf_inv(
                                    make_float2(primary_amax, 0.0f),
                                    primary_sf_pair,
                                    primary_sf_inv_pair);
                                const float primary_sf =
                                    primary_sf_pair.x;
                                const float primary_sf_inv =
                                    primary_sf_inv_pair.x;
                                const auto primary = __nv_fp8x4_e4m3(
                                    make_float4(
                                        values[0] * primary_sf_inv,
                                        values[1] * primary_sf_inv,
                                        values[2] * primary_sf_inv,
                                        values[3] * primary_sf_inv));
                                const float4 primary_float =
                                    static_cast<float4>(primary);
                                float residual[kValuesPerLane] = {
                                    values[0] -
                                        primary_float.x * primary_sf,
                                    values[1] -
                                        primary_float.y * primary_sf,
                                    values[2] -
                                        primary_float.z * primary_sf,
                                    values[3] -
                                        primary_float.w * primary_sf,
                                };
                                float residual_amax = 0.0f;
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kValuesPerLane; ++i)
                                    residual_amax = cute::max(
                                        residual_amax,
                                        cute::abs(residual[i]));
                                #pragma unroll
                                for (uint32_t offset = 4;
                                     offset > 0; offset >>= 1)
                                    residual_amax = cute::max(
                                        residual_amax,
                                        __shfl_xor_sync(
                                            0xffffffff, residual_amax,
                                            offset, 8));
                                float2 residual_sf_pair;
                                float2 residual_sf_inv_pair;
                                math::get_e4m3_sf_and_sf_inv(
                                    make_float2(residual_amax, 0.0f),
                                    residual_sf_pair,
                                    residual_sf_inv_pair);
                                const auto residual_quantized =
                                    __nv_fp8x4_e4m3(make_float4(
                                        residual[0] *
                                            residual_sf_inv_pair.x,
                                        residual[1] *
                                            residual_sf_inv_pair.x,
                                        residual[2] *
                                            residual_sf_inv_pair.x,
                                        residual[3] *
                                            residual_sf_inv_pair.x));

                                const uint32_t row = local_store_m & 7u;
                                const uint32_t logical_k_byte =
                                    group_idx * 32 +
                                    lane_in_group * kValuesPerLane;
                                const uint32_t byte_offset =
                                    (local_store_m >> 3) * 8 * 128 +
                                    row * 128 +
                                    ((logical_k_byte >> 4) ^ row) * 16 +
                                    (logical_k_byte & 15);
                                if (store_data) {
                                    *reinterpret_cast<uint32_t*>(
                                        primary_bytes + byte_offset) =
                                        primary.__x;
                                    *reinterpret_cast<uint32_t*>(
                                        residual_bytes + byte_offset) =
                                        residual_quantized.__x;
                                }

                                const uint32_t primary_scale_byte =
                                    (*reinterpret_cast<const uint32_t*>(
                                         &primary_sf)) >> 23;
                                const uint32_t residual_scale_byte =
                                    (*reinterpret_cast<const uint32_t*>(
                                         &residual_sf_pair.x)) >> 23;
                                uint32_t primary_scale_word = 0;
                                uint32_t residual_scale_word = 0;
                                #pragma unroll
                                for (uint32_t group = 0;
                                     group < 4; ++group) {
                                    primary_scale_word |=
                                        __shfl_sync(
                                            0xffffffff,
                                            primary_scale_byte,
                                            group * 8) <<
                                        (group * 8);
                                    residual_scale_word |=
                                        __shfl_sync(
                                            0xffffffff,
                                            residual_scale_byte,
                                            group * 8) <<
                                        (group * 8);
                                }
                                if (valid_row && lane_idx == 0) {
                                    // MegaMoE pre-transposes each 128-row scale
                                    // tile into the physical order consumed by
                                    // UTCCP, matching its forward ring buffers.
                                    const uint32_t sf_m =
                                        (full_m & ~127u) +
                                        (full_m & 31u) * 4 +
                                        ((full_m >> 5) & 3u);
                                    smem_sfa[stage_idx][sf_m] =
                                        primary_scale_word;
                                    smem_dgrad_sfa_residual[
                                        stage_idx][sf_m] =
                                        residual_scale_word;
                                    if (is_leader_cta) {
                                        constexpr uint32_t kFollowerCTARank = 1;
                                        store_cluster_uint32(
                                            &smem_sfa[stage_idx][sf_m],
                                            kFollowerCTARank,
                                            primary_scale_word);
                                        store_cluster_uint32(
                                            &smem_dgrad_sfa_residual[
                                                stage_idx][sf_m],
                                            kFollowerCTARank,
                                            residual_scale_word);
                                    }
                                }
                            }
                            cutlass::arch::fence_view_async_shared();
                            __threadfence_cluster();
                            cutlass::arch::NamedBarrier::sync(
                                kNumResidualProducerThreads,
                                kInlineWeightProducerBarrier);
                            if (producer_warp_idx == 0 &&
                                cute::elect_one_sync())
                                full_barriers[stage_idx]->arrive(0u);
                            }

                            if constexpr (
                                kInlineResidualMXFP8Dgrad) {
                              if (build_weight_tile) {
                              weight_load_barrier->wait(weight_source_phase);
                            // Re-group the original MXFP4 W2 matrix along the
                            // transposed dgrad K axis. E2M1 * UE8M0 values are
                            // dyadic and fit E4M3 exactly, so this changes only
                            // the encoding—not the represented weight. Each
                            // CTA materializes the complete B tile required by
                            // the two-CTA UMMA instruction.
                            auto* weight_bytes =
                                reinterpret_cast<uint8_t*>(
                                    smem_dgrad_b[stage_idx]);
                            if constexpr (kOnDemandResidualWeightCache) {
                                auto* cached_q =
                                    reinterpret_cast<uint8_t*>(
                                        w2_dequant_scratch);
                                auto* cached_sf =
                                    reinterpret_cast<uint32_t*>(
                                        cached_q +
                                        static_cast<uint64_t>(
                                            kNumExperts) *
                                            kIntermediateHidden * kHidden);
                                k3_mxfp4_to_mxfp8_transposed_tile(
                                    smem_dgrad_weight_source,
                                    smem_dgrad_weight_scale_source,
                                    weight_bytes,
                                    smem_sfb[stage_idx],
                                    cached_q +
                                        static_cast<uint64_t>(expert_idx) *
                                            kIntermediateHidden * kHidden,
                                    cached_sf +
                                        static_cast<uint64_t>(expert_idx) *
                                            kIntermediateHidden *
                                            (kHidden / DGRAD_BLOCK_K),
                                    kIntermediateHidden,
                                    kHidden,
                                    n_block_idx,
                                    k_block_idx,
                                    producer_warp_idx * 32 + lane_idx);
                            } else {
                            for (uint32_t local_n = producer_warp_idx;
                                 local_n < LOAD_BLOCK_N;
                                 local_n += kNumResidualProducerWarps) {
                                const uint32_t global_n =
                                    n_block_idx * LOAD_BLOCK_N + local_n;
                                uint16_t weight_bf16_bits[
                                    kValuesPerLane];
                                uint32_t weight_amax_bits = 0;
                                #pragma unroll
                                for (uint32_t i = 0;
                                     i < kValuesPerLane; ++i) {
                                    const uint8_t packed =
                                        static_cast<uint8_t>(
                                            smem_dgrad_weight_source[
                                                (group_idx * 32 +
                                                 lane_in_group *
                                                     kValuesPerLane +
                                                 i) *
                                                    (LOAD_BLOCK_N / 2) +
                                                local_n / 2]);
                                    const uint32_t source_scale_bits =
                                        *reinterpret_cast<const uint32_t*>(
                                            &smem_dgrad_weight_scale_source[
                                            (group_idx * 32 +
                                             lane_in_group *
                                                 kValuesPerLane +
                                             i) *
                                                (LOAD_BLOCK_N / kGranK) +
                                            local_n / kGranK]);
                                    const uint8_t nibble =
                                        (packed >>
                                         ((global_n & 1u) * 4)) &
                                        0xf;
                                    weight_bf16_bits[i] =
                                        k3_mxfp4_bf16_bits(
                                            nibble,
                                            source_scale_bits);
                                    weight_amax_bits = cute::max(
                                        weight_amax_bits,
                                        static_cast<uint32_t>(
                                            weight_bf16_bits[i] &
                                            0x7fff));
                                }

                                #pragma unroll
                                for (uint32_t offset = 4;
                                     offset > 0; offset >>= 1)
                                    weight_amax_bits = cute::max(
                                        weight_amax_bits,
                                        __shfl_xor_sync(
                                            0xffffffff,
                                            weight_amax_bits,
                                            offset, 8));
                                const uint32_t weight_scale_byte =
                                    cute::max(
                                        88,
                                        static_cast<int>(
                                            (weight_amax_bits >> 7) &
                                            0xff) - 8);
                                const uint32_t weight_quantized =
                                    k3_quantize_bf16x4_e4m3(
                                        weight_bf16_bits,
                                        weight_scale_byte);
                                const uint32_t weight_row = local_n & 7u;
                                const uint32_t logical_k_byte =
                                    group_idx * 32 +
                                    lane_in_group * kValuesPerLane;
                                const uint32_t weight_byte_offset =
                                    (local_n >> 3) * 8 * 128 +
                                    weight_row * 128 +
                                    ((logical_k_byte >> 4) ^ weight_row) *
                                        16 +
                                    (logical_k_byte & 15);
                                *reinterpret_cast<uint32_t*>(
                                    weight_bytes + weight_byte_offset) =
                                    weight_quantized;
                                if constexpr (
                                    kOnDemandResidualWeightCache) {
                                    auto* cached_q =
                                        reinterpret_cast<uint8_t*>(
                                            w2_dequant_scratch);
                                    *reinterpret_cast<uint32_t*>(
                                        cached_q +
                                        (static_cast<uint64_t>(expert_idx) *
                                             kIntermediateHidden +
                                         global_n) *
                                            kHidden +
                                        k_block_idx * DGRAD_BLOCK_K +
                                        logical_k_byte) =
                                        weight_quantized;
                                }

                                uint32_t weight_scale_word = 0;
                                #pragma unroll
                                for (uint32_t group = 0;
                                     group < 4; ++group)
                                    weight_scale_word |=
                                        __shfl_sync(
                                            0xffffffff,
                                            weight_scale_byte,
                                            group * 8) <<
                                        (group * 8);
                                if (lane_idx == 0) {
                                    const uint32_t sf_n =
                                        (local_n & 31u) * 4 +
                                        ((local_n >> 5) & 3u);
                                    smem_sfb[stage_idx][sf_n] =
                                        weight_scale_word;
                                    if constexpr (
                                        kOnDemandResidualWeightCache) {
                                        auto* cached_q =
                                            reinterpret_cast<uint8_t*>(
                                                w2_dequant_scratch);
                                        auto* cached_sf =
                                            reinterpret_cast<uint32_t*>(
                                                cached_q +
                                                static_cast<uint64_t>(
                                                    kNumExperts) *
                                                    kIntermediateHidden *
                                                    kHidden);
                                        const uint32_t global_sf_n =
                                            (global_n & ~127u) +
                                            (global_n & 31u) * 4 +
                                            ((global_n >> 5) & 3u);
                                        cached_sf[
                                            static_cast<uint64_t>(expert_idx) *
                                                kIntermediateHidden *
                                                (kHidden / DGRAD_BLOCK_K) +
                                            k_block_idx *
                                                kIntermediateHidden +
                                            global_sf_n] =
                                            weight_scale_word;
                                    }
                                }
                            }
                            }
                            cutlass::arch::fence_view_async_shared();
                            cutlass::arch::NamedBarrier::sync(
                                kNumResidualProducerThreads,
                                kResidualWeightProducerBarrier);
                              if (producer_warp_idx == 0 &&
                                  cute::elect_one_sync()) {
                                  if constexpr (
                                      kOnDemandResidualWeightCache) {
                                      asm volatile(
                                          "fence.proxy.async.global;"
                                          ::: "memory");
                                      __threadfence();
                                      asm volatile(
                                          "st.release.gpu.global.u32 [%0], %1;"
                                          :: "l"(weight_tile_states +
                                                 weight_tile_idx),
                                             "r"(launch_epoch)
                                          : "memory");
                                  }
                                  full_barriers[stage_idx]->arrive(0u);
                              }
                              weight_source_phase ^= 1;
                              }
                            }
                            if ((!kInlineResidualMXFP8Dgrad ||
                                 !build_weight_tile) &&
                                producer_warp_idx == 0 &&
                                cute::elect_one_sync()) {
                                tma::copy<
                                    DGRAD_BLOCK_K, LOAD_BLOCK_N,
                                    128, residual_dgrad_dtype_t>(
                                    &tensor_map_w2_dgrad_weights,
                                    full_barriers[stage_idx],
                                    reinterpret_cast<
                                        residual_dgrad_dtype_t*>(
                                        smem_dgrad_b[stage_idx]),
                                    k_block_idx * DGRAD_BLOCK_K,
                                    expert_idx * kIntermediateHidden +
                                        n_block_idx * BLOCK_N,
                                    2);
                                tma::copy<BLOCK_N, 1, 0>(
                                    &tensor_map_w2_dgrad_weights_sf,
                                    full_barriers[stage_idx],
                                    smem_sfb[stage_idx],
                                    n_block_idx * BLOCK_N,
                                    expert_idx *
                                            (kHidden / (kGranK * 4)) +
                                        k_block_idx,
                                    2);
                                if (is_leader_cta) {
                                    full_barriers[stage_idx]
                                        ->arrive_and_expect_tx(
                                            (SMEM_B_SIZE_PER_STAGE +
                                             SMEM_SFB_SIZE_PER_STAGE) *
                                            2);
                                } else {
                                    full_barriers[stage_idx]->arrive(0u);
                                }
                            }
                        }
                        __syncwarp();
                    }
                });
        } else if (
            warp_idx == 1 ||
            (kInlineWeightDequant && warp_idx >= 3 && warp_idx < 8)) {
            // Load the in-kernel dequantized transposed W2 workspace.  It is
            // shared by all M tiles for this expert instead of reconverting
            // the same packed weights for every token block.
            for_each_dgrad_block(
                [&](const uint32_t& expert_idx,
                    const uint32_t& pool_block_offset,
                    const uint32_t& m_block_idx,
                    const uint32_t& n_block_idx,
                    const uint32_t& valid_m) {
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
                        if constexpr (
                            !kBF16Mode && !kInlineWeightDequant &&
                            !kResidualMXFP8Dgrad) {
                            while (ptx::ld_acq(
                                       weight_tile_states +
                                       weight_tile_idx) !=
                                   launch_epoch) {
                            }
                        }
                        constexpr bool weight_tile_ready =
                            !kInlineWeightDequant;
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        if (weight_tile_ready) {
                            if (cute::elect_one_sync()) {
                                if constexpr (kResidualMXFP8Dgrad) {
                                    tma::copy<
                                        DGRAD_BLOCK_K, LOAD_BLOCK_N,
                                        128, residual_dgrad_dtype_t>(
                                        &tensor_map_w2_dgrad_weights,
                                        full_barriers[stage_idx],
                                        reinterpret_cast<
                                            residual_dgrad_dtype_t*>(
                                            smem_dgrad_b[stage_idx]),
                                        k_block_idx * DGRAD_BLOCK_K,
                                        expert_idx *
                                                kIntermediateHidden +
                                            n_block_idx * BLOCK_N,
                                        2);
                                    tma::copy<BLOCK_N, 1, 0>(
                                        &tensor_map_w2_dgrad_weights_sf,
                                        full_barriers[stage_idx],
                                        smem_sfb[stage_idx],
                                        n_block_idx * BLOCK_N,
                                        expert_idx *
                                                (kHidden /
                                                 (kGranK * 4)) +
                                            k_block_idx,
                                        2);
                                    if (is_leader_cta) {
                                        full_barriers[stage_idx]
                                            ->arrive_and_expect_tx(
                                                (SMEM_B_SIZE_PER_STAGE +
                                                 SMEM_SFB_SIZE_PER_STAGE) *
                                                2);
                                    } else {
                                        full_barriers[stage_idx]
                                            ->arrive(0u);
                                    }
                                } else {
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
                                        n_block_idx * BLOCK_N,
                                        expert_idx * kHidden +
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
                            }
                        } else {
                            constexpr uint32_t
                                kPairsPerTile =
                                    LOAD_BLOCK_N *
                                    DGRAD_BLOCK_K / 2;
                            auto* smem_b_bytes =
                                reinterpret_cast<uint8_t*>(
                                    smem_dgrad_b[stage_idx]);
                            const uint32_t producer_warp_idx =
                                warp_idx == 1 ? 0 : warp_idx - 2;
                            for (uint32_t pair_idx =
                                     producer_warp_idx * 32 + lane_idx;
                                 pair_idx < kPairsPerTile;
                                 pair_idx +=
                                     kNumInlineWeightProducerThreads) {
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
                                    (static_cast<uint64_t>(expert_idx) *
                                         kHidden +
                                     global_k) *
                                        (kIntermediateHidden / 32) +
                                    n_block_idx *
                                        (LOAD_BLOCK_N / 32) +
                                    (local_n_pair * 2) / 32);
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
                                    __float2half2_rn(scale));
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
                                    // tma::copy splits the 128-wide MN tile
                                    // into two independently swizzled 64x64
                                    // atoms. Within each atom MN is the inner
                                    // dimension and K is the outer dimension.
                                    const uint32_t atom_n =
                                        local_n / 64;
                                    const uint32_t atom_col_byte =
                                        (local_n % 64) *
                                        sizeof(dgrad_b_dtype_t);
                                    const uint32_t row =
                                        local_k & 7;
                                    const uint32_t
                                        byte_offset =
                                            atom_n *
                                                DGRAD_BLOCK_K *
                                                64 *
                                                sizeof(dgrad_b_dtype_t) +
                                            (local_k >> 3) *
                                                8 * 128 +
                                            row * 128 +
                                            ((atom_col_byte >> 4) ^
                                             row) *
                                                16 +
                                            (atom_col_byte & 15);
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
                            cutlass::arch::fence_view_async_shared();
                            cutlass::arch::NamedBarrier::sync(
                                kNumInlineWeightProducerThreads,
                                kInlineWeightProducerBarrier);
                            if (producer_warp_idx == 0 &&
                                cute::elect_one_sync())
                                full_barriers[stage_idx]
                                    ->arrive(0u);
                        }
                        __syncwarp();
                    }
                });
        } else if (warp_idx == 2) {
            if (is_leader_cta) {
              if constexpr (kResidualMXFP8Dgrad) {
                auto instr_desc =
                    cute::UMMA::make_instr_desc_block_scaled<
                        residual_dgrad_dtype_t,
                        residual_dgrad_dtype_t, float,
                        cutlass::float_ue8m0_t,
                        UMMA_M, UMMA_N,
                        cute::UMMA::Major::K,
                        cute::UMMA::Major::K>();
                auto sf_desc = mma::sm100::make_sf_desc(nullptr);
                auto primary_a_desc = mma::sm100::make_umma_desc<
                    cute::UMMA::Major::K, LOAD_BLOCK_M,
                    DGRAD_BLOCK_K, 128>(
                    reinterpret_cast<residual_dgrad_dtype_t*>(
                        smem_dgrad_a[0]),
                    0, 0);
                auto residual_a_desc = mma::sm100::make_umma_desc<
                    cute::UMMA::Major::K, LOAD_BLOCK_M,
                    DGRAD_BLOCK_K, 128>(
                    smem_dgrad_a_residual[0], 0, 0);
                auto b_desc = mma::sm100::make_umma_desc<
                    cute::UMMA::Major::K, LOAD_BLOCK_N,
                    DGRAD_BLOCK_K, 128>(
                    reinterpret_cast<residual_dgrad_dtype_t*>(
                        smem_dgrad_b[0]),
                    0, 0);
                const uint32_t primary_a_desc_lo =
                    lane_idx < kNumStages
                    ? primary_a_desc.lo +
                          lane_idx * SMEM_A_SIZE_PER_STAGE / 16
                    : 0;
                const uint32_t residual_a_desc_lo =
                    lane_idx < kNumStages
                    ? residual_a_desc.lo +
                          lane_idx *
                              SMEM_RESIDUAL_A_SIZE_PER_STAGE / 16
                    : 0;
                const uint32_t b_desc_lo =
                    lane_idx < kNumStages
                    ? b_desc.lo +
                          lane_idx * SMEM_B_SIZE_PER_STAGE / 16
                    : 0;
                uint32_t current_iter = 0;
                uint32_t residual_mma_phase = 0;

                for_each_dgrad_block(
                    [&](const uint32_t&, const uint32_t&,
                        const uint32_t&, const uint32_t&,
                        const uint32_t& valid_m) {
                        mma::sm100::update_instr_desc_with_umma_n(
                            instr_desc,
                            math::align(valid_m, 16u));
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
                            const uint32_t primary_a_desc_base =
                                ptx::exchange(
                                    primary_a_desc_lo, stage_idx);
                            const uint32_t residual_a_desc_base =
                                ptx::exchange(
                                    residual_a_desc_lo, stage_idx);
                            const uint32_t b_desc_base =
                                ptx::exchange(
                                    b_desc_lo, stage_idx);
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
                                            i *
                                                kNumUTCCPAlignedElems);
                                    utccp_t::copy(
                                        sf_desc,
                                        kTmemStartColOfSFA + i * 4);
                                    mma::sm100::replace_smem_desc_addr(
                                        sf_desc,
                                        smem_dgrad_sfa_residual[
                                            stage_idx] +
                                            i *
                                                kNumUTCCPAlignedElems);
                                    utccp_t::copy(
                                        sf_desc,
                                        kTmemStartColOfResidualSFA +
                                            i * 4);
                                }
                                mma::sm100::replace_smem_desc_addr(
                                    sf_desc, smem_sfb[stage_idx]);
                                utccp_t::copy(
                                    sf_desc, kTmemStartColOfSFB);

                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < DGRAD_BLOCK_K /
                                             DGRAD_UMMA_K;
                                     ++k) {
                                    const auto runtime_instr_desc =
                                        mma::sm100::
                                            make_runtime_instr_desc_with_sf_id(
                                                instr_desc, k, k);
                                    primary_a_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::K,
                                                LOAD_BLOCK_M, 128,
                                                residual_dgrad_dtype_t>(
                                                primary_a_desc_base,
                                                0,
                                                k * DGRAD_UMMA_K);
                                    b_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::K,
                                                LOAD_BLOCK_N, 128,
                                                residual_dgrad_dtype_t>(
                                                b_desc_base, 0,
                                                k * DGRAD_UMMA_K);
                                    ptx::
                                        SM100_MMA_MXF8F6F4_2x1SM_SS::
                                            fma(
                                                b_desc,
                                                primary_a_desc,
                                                accum_stage * UMMA_N,
                                                k_block_idx > 0 ||
                                                    k > 0,
                                                runtime_instr_desc,
                                                kTmemStartColOfSFB,
                                                kTmemStartColOfSFA);
                                }
                            }
                            __syncwarp();
                            constexpr uint16_t kCTAMask = 0x3;
                            // Retire the primary four-instruction group before
                            // issuing the residual group. This follows the
                            // proven MegaMoE cadence and avoids overfilling the
                            // asynchronous UMMA issue window.
                            cutlass::arch::
                                umma_arrive_multicast_2x1SM(
                                    reinterpret_cast<uint64_t*>(
                                        residual_mma_barrier),
                                    kCTAMask);
                            residual_mma_barrier->wait(
                                residual_mma_phase);
                            residual_mma_phase ^= 1;
                            if constexpr (kApplyResidualDgradCorrection) {
                              __syncwarp();
                              if (cute::elect_one_sync()) {
                                #pragma unroll
                                for (uint32_t k = 0;
                                     k < DGRAD_BLOCK_K /
                                             DGRAD_UMMA_K;
                                     ++k) {
                                    const auto runtime_instr_desc =
                                        mma::sm100::
                                            make_runtime_instr_desc_with_sf_id(
                                                instr_desc, k, k);
                                    residual_a_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::K,
                                                LOAD_BLOCK_M, 128,
                                                residual_dgrad_dtype_t>(
                                                residual_a_desc_base,
                                                0,
                                                k * DGRAD_UMMA_K);
                                    b_desc.lo =
                                        mma::sm100::
                                            advance_umma_desc_lo<
                                                cute::UMMA::Major::K,
                                                LOAD_BLOCK_N, 128,
                                                residual_dgrad_dtype_t>(
                                                b_desc_base, 0,
                                                k * DGRAD_UMMA_K);
                                    ptx::
                                        SM100_MMA_MXF8F6F4_2x1SM_SS::
                                            fma(
                                                b_desc,
                                                residual_a_desc,
                                                accum_stage * UMMA_N,
                                                true,
                                                runtime_instr_desc,
                                                kTmemStartColOfSFB,
                                                kTmemStartColOfResidualSFA);
                                }
                              }
                              __syncwarp();
                              cutlass::arch::
                                  umma_arrive_multicast_2x1SM(
                                      reinterpret_cast<uint64_t*>(
                                          residual_mma_barrier),
                                      kCTAMask);
                              residual_mma_barrier->wait(
                                  residual_mma_phase);
                              residual_mma_phase ^= 1;
                            }
                            if (cute::elect_one_sync()) {
                                empty_barriers[stage_idx]->arrive();
                                if (k_block_idx ==
                                    kHidden / DGRAD_BLOCK_K - 1)
                                    tmem_full_barriers[accum_stage]
                                        ->arrive();
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
              } else {
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
                    [&](const uint32_t& expert_idx,
                        const uint32_t& pool_block_offset,
                        const uint32_t& m_block_idx,
                        const uint32_t& n_block_idx,
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
            } else if constexpr (kResidualMXFP8Dgrad) {
                // The leader issues 2-CTA UMMA. The follower mirrors only the
                // completion waits, then releases its local shared/TMEM
                // barriers. This preserves cluster-local barrier ownership
                // without relying on a zero-work UMMA commit.
                uint32_t current_iter = 0;
                uint32_t follower_residual_mma_phase = 0;
                for_each_dgrad_block(
                    [&](const uint32_t&, const uint32_t&,
                        const uint32_t&, const uint32_t&,
                        const uint32_t&) {
                        const uint32_t accum_stage =
                            current_iter++ % kNumEpilogueStages;
                        #pragma unroll 1
                        for (uint32_t k_block_idx = 0;
                             k_block_idx <
                                 kHidden / DGRAD_BLOCK_K;
                             advance_pipeline(k_block_idx)) {
                            residual_mma_barrier->wait(
                                follower_residual_mma_phase);
                            follower_residual_mma_phase ^= 1;
                            if constexpr (kApplyResidualDgradCorrection) {
                                residual_mma_barrier->wait(
                                    follower_residual_mma_phase);
                                follower_residual_mma_phase ^= 1;
                            }
                            if (cute::elect_one_sync()) {
                                empty_barriers[stage_idx]->arrive();
                                if (k_block_idx ==
                                    kHidden / DGRAD_BLOCK_K - 1)
                                    tmem_full_barriers[accum_stage]
                                        ->arrive();
                            }
                            __syncwarp();
                        }
                    });
            }
        } else if (warp_idx >= kDgradEpilogueWarpStart) {
            const uint32_t epilogue_warp_idx =
                warp_idx - kDgradEpilogueWarpStart;
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
                            float gate_tanh = 0.0f;
                            float sig;
                            float denom = 0.0f;
                            if constexpr (
                                kActivationType ==
                                ActivationType::SiTU) {
                                gate_tanh =
                                    kFastMath
                                    ? __tanhf(
                                          gate / kSituBeta)
                                    : tanhf(
                                          gate / kSituBeta);
                                if constexpr (kSituBeta == 4.0f) {
                                    // K3 beta=4 makes sigmoid(g) exactly
                                    // expressible from the already-required
                                    // tanh(g / beta): tanh(g / 2) follows
                                    // from the double-angle identity, and
                                    // sigmoid(g) is
                                    // (1 + tanh(g / 2)) / 2. This removes
                                    // one accurate expf per routed
                                    // intermediate while preserving the
                                    // real-valued SiTU formula; no fast-math
                                    // intrinsic or extra BF16 boundary is
                                    // introduced.
                                    const float gate_tanh_sq =
                                        __fmul_rn(
                                            gate_tanh,
                                            gate_tanh);
                                    const float tanh_gate_half =
                                        __fdiv_rn(
                                            __fmul_rn(
                                                2.0f,
                                                gate_tanh),
                                            __fadd_rn(
                                                1.0f,
                                                gate_tanh_sq));
                                    sig = __fmul_rn(
                                        0.5f,
                                        __fadd_rn(
                                            1.0f,
                                            tanh_gate_half));
                                } else {
                                    const float neg_exp =
                                        kFastMath
                                        ? __expf(-gate)
                                        : expf(-gate);
                                    sig = 1.0f / __fadd_rn(
                                        1.0f,
                                        neg_exp);
                                }
                            } else {
                                const float neg_exp =
                                    !kBF16Mode || kFastMath
                                    ? __expf(-z)
                                    : expf(-z);
                                denom =
                                    __fadd_rn(1.0f, neg_exp);
                                sig = 1.0f / denom;
                            }
                            cd_dtype_t h_act_bf16;
                            cd_dtype_t grad_gate_bf16;
                            cd_dtype_t grad_up_bf16;
                            if constexpr (
                                kActivationType ==
                                    ActivationType::SiTU) {
                                static_assert(
                                    kSituBeta > 0.0f,
                                    "SiTU beta must be positive");
                                static_assert(
                                    kSituLinearBeta > 0.0f,
                                    "SiTU linear beta must be positive");
                                const float situ_gate =
                                    __fmul_rn(
                                        __fmul_rn(
                                            kSituBeta,
                                            gate_tanh),
                                        sig);
                                float activated_up = up;
                                float grad_up_scale = 1.0f;
                                if constexpr (
                                    kSituLinearBeta !=
                                    cute::numeric_limits<
                                        float>::infinity()) {
                                    const float up_tanh =
                                        kFastMath
                                        ? __tanhf(
                                              up /
                                              kSituLinearBeta)
                                        : tanhf(
                                              up /
                                              kSituLinearBeta);
                                    activated_up = __fmul_rn(
                                        kSituLinearBeta,
                                        up_tanh);
                                    grad_up_scale = __fsub_rn(
                                        1.0f,
                                        __fmul_rn(
                                            up_tanh,
                                            up_tanh));
                                }
                                const float one_minus_gate_tanh_sq =
                                    __fsub_rn(
                                        1.0f,
                                        __fmul_rn(
                                            gate_tanh,
                                            gate_tanh));
                                const float gate_grad =
                                    __fadd_rn(
                                        __fmul_rn(
                                            one_minus_gate_tanh_sq,
                                            sig),
                                        __fmul_rn(
                                            situ_gate,
                                            __fsub_rn(
                                                1.0f,
                                                sig)));
                                h_act_bf16 = cd_dtype_t(
                                    __fmul_rn(
                                        situ_gate,
                                        activated_up));
                                grad_gate_bf16 = cd_dtype_t(
                                    __fmul_rn(
                                        __fmul_rn(
                                            grad_h,
                                            activated_up),
                                        gate_grad));
                                grad_up_bf16 = cd_dtype_t(
                                    __fmul_rn(
                                        __fmul_rn(
                                            grad_h,
                                            situ_gate),
                                        grad_up_scale));
                            } else if constexpr (
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
                            const uint64_t grad_row_base =
                                static_cast<uint64_t>(pool_row) *
                                (2 * kIntermediateHidden);
                            if (inplace_gate_up_grad) {
                                grad_gate_up_output[
                                    grad_row_base + gate_col] =
                                    grad_gate_bf16;
                                grad_gate_up_output[
                                    grad_row_base + up_col] =
                                    grad_up_bf16;
                            } else {
                                grad_gate_up_output[
                                    grad_row_base + hidden_col] =
                                    grad_gate_bf16;
                                grad_gate_up_output[
                                    grad_row_base +
                                    kIntermediateHidden +
                                    hidden_col] =
                                    grad_up_bf16;
                            }
                        }
                    }
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage]->arrive(0u);
                });

        }

        __syncthreads();
        if constexpr (kCompileW13Dgrad) {
            // K3's POST_DOWN router dot is independent of W13 dgrad once W2
            // has produced grad_gate_up.  The four communication warps are
            // otherwise only extra W13 scatter workers at this point.  Let
            // them reduce router rows while the UMMA/TMA roles compute W13;
            // the publication barrier after W13 joins both paths.  This is a
            // first readiness-driven task split borrowed from MoK: independent
            // work no longer serializes an entire persistent-grid phase.
            constexpr bool kOverlapRouteGradWithW13 =
                kNumDispatchWarps > 0 &&
                kComputeRouteGrad && !kInputsPrepared &&
                kRouteWeightMode == RouteWeightMode::PostDown &&
                !kRouteGradBeforeW2 && !kExactSourceX;
            constexpr uint32_t kNumW13EpilogueThreads =
                kNumDgradEpilogueThreads -
                (kOverlapRouteGradWithW13
                     ? kNumDispatchThreads
                     : 0);
            DG_STATIC_ASSERT(
                kNumW13EpilogueThreads >= kNumEpilogueThreads,
                "W13 needs the four proven TMEM loader warps");

            // W13 dgrad consumes grad_gate_up rows produced by every CTA in
            // the preceding L2-dgrad/SwiGLU phase. Cluster synchronization is
            // insufficient here: an early cluster can otherwise read rows
            // whose owning cluster has not stored them yet.
            full_grid_phase_barrier(12);

            if constexpr (kEarlyW2Wgrad) {
                // W2's dgrad and SiTU epilogue have retired globally, so its
                // two exact BF16 operands are final. Pad only their last
                // expert blocks, publish the grouped layout, and compute dW2
                // before either buffer is repurposed for W13 MXFP8 planes.
                if constexpr (kClearWgradPadding)
                    clear_w2_wgrad_padding_rows();
                initialize_wgrad_grouped_layout();
                asm volatile(
                    "fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount);

                // The embedded BF16 body aliases the whole dynamic shared
                // allocation. Retire the parent barriers before that alias,
                // then recreate the dispatch subset needed by late exact-X
                // after W2 returns. Mainloop barriers are recreated at the
                // W13 phase boundary below.
                if (warp_idx == 0 && cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumStages; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                full_barriers[i]));
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                empty_barriers[i]));
                    }
                    if constexpr (kResidualMXFP8Dgrad) {
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                weight_load_barrier));
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                residual_mma_barrier));
                    }
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kNumEpilogueStages; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                tmem_full_barriers[i]));
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                tmem_empty_barriers[i]));
                    }
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kNumDispatchBarriers; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                dispatch_barriers[i]));
                    }
                }
                comm::cluster_sync_with_relaxed_arrive();
                if (warp_idx == 0)
                    Allocator().free(0, kNumTmemCols);
                __syncthreads();

                run_wgrad.template operator()<false>(
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    false);
                comm::cluster_sync_with_relaxed_arrive();

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kNumDispatchBarriers; ++i)
                        dispatch_barriers[i]->init(1);
                    cutlass::arch::fence_barrier_init();
                }
                comm::cluster_sync_with_relaxed_arrive();
            }

            if (inplace_gate_up_grad && !kBuildW13ResidualActsOnce) {
                // Each CTA owns complete rows. Shared memory snapshots one
                // interleaved row before any conventional-column stores, so
                // the permutation is race-free without another HBM tensor.
                auto* gate_up_row_scratch =
                    reinterpret_cast<cd_dtype_t*>(smem_gemm_base);
                constexpr uint32_t kGateUpColumns =
                    2 * kIntermediateHidden;
                for (uint32_t pool_row = blockIdx.x;
                     pool_row < num_pool_rows;
                     pool_row += kNumSMs) {
                    const uint64_t row_base =
                        static_cast<uint64_t>(pool_row) *
                        kGateUpColumns;
                    for (uint32_t col = threadIdx.x;
                         col < kGateUpColumns;
                         col += kNumThreads) {
                        gate_up_row_scratch[col] =
                            grad_gate_up_output[row_base + col];
                    }
                    __syncthreads();
                    for (uint32_t col = threadIdx.x;
                         col < kGateUpColumns;
                         col += kNumThreads) {
                        const uint32_t hidden_col =
                            col < kIntermediateHidden
                            ? col
                            : col - kIntermediateHidden;
                        const uint32_t source_col =
                            (hidden_col / 8) * 16 +
                            (col >= kIntermediateHidden ? 8 : 0) +
                            (hidden_col & 7);
                        grad_gate_up_output[row_base + col] =
                            gate_up_row_scratch[source_col];
                    }
                    __syncthreads();
                }
                // W13 TMA readers may start on a different CTA, so publish the
                // completed in-place permutation grid-wide.
                full_grid_phase_barrier(kTraceSiteCount);
            }

            build_w13_residual_acts_once();

            // Gate/up has had its final SiTU-backward read. A multichunk
            // caller can now reuse that pool allocation as W13's canonical
            // BF16 dequant matrix before the W13 UMMA phase begins.
            if constexpr (phase_ordered_w13_dequant) {
                dequant_noninline_weights_once(true, true, true);
                full_grid_phase_barrier(kTraceSiteCount);
            }

            if constexpr (kBF16Mode) {
                // In phase-ordered mode these outputs may still contain the
                // forward gate values or reverse-dispatched grad-y in every
                // row that the active activation tiles did not visit. Clear
                // per-expert block padding and the unused capacity tail only
                // after all active gate reads have completed.
                uint32_t padding_pool_block_offset = 0;
                #pragma unroll 1
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
                kComputeRouteGrad && !kInputsPrepared &&
                !kOverlapRouteGradWithW13) {
              // Trace site 14 is otherwise used only by the BF16 PRE_DOWN
              // alias handoff, which is compile-time absent for K3 POST_DOWN.
              trace_begin(14);
              if (!kRouteGradBeforeW2 || late_exact_source_x) {
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
                    // This is a serial count reduction; unrolling 112 loads
                    // only inflates the instruction image.
                    #pragma unroll 1
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
                // The dgrad launch has 32 warps.  Once the W2 epilogue has
                // retired, four additional idle warps can reuse its shared
                // pipeline storage and the completed first dispatch-barrier
                // epoch.  Doubling exact-X producers removes the serial TMA
                // tail without increasing launch size or dynamic SMEM.
                constexpr uint32_t kMaxRouteExactXDispatchWarps =
                    2 * kNumDispatchWarps;
                const uint32_t num_route_exact_x_dispatch_warps =
                    route_group_threads == 32
                    ? kMaxRouteExactXDispatchWarps
                    : kNumDispatchWarps;
                const uint32_t route_exact_x_dispatch_threads =
                    num_route_exact_x_dispatch_warps * 32;
                const uint32_t route_worker_threads =
                    late_exact_source_x
                    ? kNumThreads - route_exact_x_dispatch_threads
                    : kNumThreads;
                const uint32_t num_route_groups_per_cta =
                    route_worker_threads / route_group_threads;
                const uint32_t route_group_idx =
                    threadIdx.x / route_group_threads;
                const uint32_t route_group_lane_idx =
                    threadIdx.x &
                    (route_group_threads - 1);
                // Number route groups in the same wave-major order used by
                // the exact-X TMA warps below.  For K3/32K, CTA-major route
                // numbering activated only the first ~49 of 148 CTAs for an
                // average 585-row expert while every CTA's TMA warp waited.
                // Wave-major numbering makes each CTA produce the four rows
                // its own TMA warps consume before moving to the next wave.
                const uint32_t route_groups_per_dispatch_wave =
                    kNumSMs * num_route_exact_x_dispatch_warps;
                const uint32_t route_group_wave =
                    route_group_idx /
                    num_route_exact_x_dispatch_warps;
                const uint32_t route_dispatch_warp_idx =
                    route_group_idx %
                    num_route_exact_x_dispatch_warps;
                const uint32_t global_route_group =
                    route_group_wave *
                        route_groups_per_dispatch_wave +
                    blockIdx.x *
                        num_route_exact_x_dispatch_warps +
                    route_dispatch_warp_idx;
                const uint32_t route_group_stride =
                    math::ceil_div(
                        num_route_groups_per_cta,
                        num_route_exact_x_dispatch_warps) *
                    route_groups_per_dispatch_wave;
                const uint32_t route_ready_epoch =
                    launch_epoch ^ 0x20000000u;
                // The local route-gradient value is dead immediately after
                // its remote publication. Reuse that exact one-word-per-row
                // storage as the late exact-X readiness epoch. Unlike the
                // fixed weight-tile state array, this alias scales with every
                // true-varlen launch bucket without allocating more memory.
                auto* route_ready_states =
                    reinterpret_cast<uint32_t*>(grad_route_output);
                if (threadIdx.x < route_worker_threads) {
                  uint32_t route_pool_block_offset = 0;
                  #pragma unroll 1
                  for (uint32_t expert_idx = 0;
                       expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(expert_counts + expert_idx));
                    for (uint32_t token_idx = global_route_group;
                         token_idx < num_tokens;
                         token_idx += route_group_stride) {
                        const uint32_t pool_row =
                            route_pool_block_offset * BLOCK_M +
                            token_idx;
                        float grad_route = 0.0f;
                        if constexpr (kRouteGradBeforeW2) {
                            // The once-quantized W2 activation may alias the
                            // POST_DOWN grad-y pool. Its route dot was already
                            // published before that overwrite; retain this
                            // row walk only to release late exact-X dispatch.
                            grad_route = grad_route_output[pool_row];
                        } else if constexpr (false) {
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
                            constexpr uint32_t kRouteVectorValues =
                                kRouteWeightMode ==
                                        RouteWeightMode::PostDown
                                    ? 8
                                    : 4;
                            float lane_sums[kRouteVectorValues] = {};
                            if constexpr (
                                kRouteWeightMode ==
                                RouteWeightMode::PostDown) {
                                for (uint32_t col_base =
                                         route_group_lane_idx *
                                         kRouteVectorValues;
                                     col_base < kHidden;
                                     col_base +=
                                         route_group_threads *
                                         kRouteVectorValues) {
                                    // Eight adjacent BF16 values are naturally
                                    // 16-byte aligned for every production K3
                                    // row. Force one vector transaction per
                                    // operand instead of eight scalar loads.
                                    // Explicit round-to-nearest FP32 math
                                    // keeps the association deterministic;
                                    // parity is enforced by the route-cosine
                                    // release gate.
                                    const auto grad_y_packed =
                                        *reinterpret_cast<const uint4*>(
                                            grad_y_unweighted_output +
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            col_base);
                                    const auto down_packed =
                                        *reinterpret_cast<const uint4*>(
                                            down_unweighted_output +
                                            static_cast<uint64_t>(
                                                pool_row) *
                                                kHidden +
                                            col_base);
                                    const auto* grad_y_values =
                                        reinterpret_cast<
                                            const cd_dtype_t*>(
                                            &grad_y_packed);
                                    const auto* down_values =
                                        reinterpret_cast<
                                            const cd_dtype_t*>(
                                            &down_packed);
                                    #pragma unroll
                                    for (uint32_t i = 0;
                                         i < kRouteVectorValues;
                                         ++i) {
                                        const float grad_y =
                                            static_cast<float>(
                                                grad_y_values[i]);
                                        const float down =
                                            static_cast<float>(
                                                down_values[i]);
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
                            if constexpr (
                                kRouteWeightMode ==
                                RouteWeightMode::PostDown) {
                                const float sum_01 = __fadd_rn(
                                    lane_sums[0], lane_sums[1]);
                                const float sum_23 = __fadd_rn(
                                    lane_sums[2], lane_sums[3]);
                                const float sum_45 = __fadd_rn(
                                    lane_sums[4], lane_sums[5]);
                                const float sum_67 = __fadd_rn(
                                    lane_sums[6], lane_sums[7]);
                                grad_route = __fadd_rn(
                                    __fadd_rn(sum_01, sum_23),
                                    __fadd_rn(sum_45, sum_67));
                            } else {
                                grad_route = __fadd_rn(
                                    __fadd_rn(
                                        lane_sums[0],
                                        lane_sums[1]),
                                    lane_sums[2]);
                                grad_route = __fadd_rn(
                                    grad_route, lane_sums[3]);
                            }

                            if (route_group_threads > 32) {
                                route_lane_sums[threadIdx.x] =
                                    grad_route;
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
                            if (late_exact_source_x) {
                                // Publish completion of every read from this
                                // grad-y row before a TMA dispatch warp reuses
                                // the aliased row for exact source X.
                                // There is exactly one producer per row, so a
                                // release store paired with ``ld_acq`` avoids
                                // the serialization cost of a global atomic.
                                asm volatile(
                                    "st.release.gpu.global.u32 [%0], %1;"
                                    :: "l"(route_ready_states + pool_row),
                                       "r"(route_ready_epoch)
                                    : "memory");
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

                if (
                    late_exact_source_x &&
                    threadIdx.x >= route_worker_threads &&
                    threadIdx.x <
                        route_worker_threads +
                            route_exact_x_dispatch_threads) {
                    const uint32_t dispatch_warp_idx =
                        (threadIdx.x - route_worker_threads) / 32;
                    auto* pull_buffer = dispatch_warp_idx <
                            kNumDispatchWarps
                        ? reinterpret_cast<cd_dtype_t*>(smem_buffer) +
                              dispatch_warp_idx * kHidden
                        : reinterpret_cast<cd_dtype_t*>(
                              smem_gemm_base) +
                              (dispatch_warp_idx -
                               kNumDispatchWarps) *
                                  kHidden;
                    auto* pull_mbarrier =
                        dispatch_warp_idx < kNumDispatchWarps
                        ? dispatch_barriers[
                              kNumDispatchWarps +
                              dispatch_warp_idx]
                        : dispatch_barriers[
                              dispatch_warp_idx -
                              kNumDispatchWarps];
                    uint32_t pull_phase = 0;
                    if (
                        dispatch_warp_idx >= kNumDispatchWarps &&
                        cute::elect_one_sync()) {
                        if constexpr (!kEarlyW2Wgrad) {
                            // Without the embedded W2 body, the extra warps
                            // reuse a completed initial-dispatch barrier.
                            // Recover its next parity from the exact number
                            // of prior transactions. Early W2 recreated every
                            // dispatch barrier, so phase zero is already the
                            // correct starting state in that schedule.
                            const uint32_t reused_dispatch_warp_idx =
                                dispatch_warp_idx - kNumDispatchWarps;
                            const uint32_t first_token_idx =
                                blockIdx.x * kNumDispatchWarps +
                                reused_dispatch_warp_idx;
                            constexpr uint32_t kInitialDispatchStride =
                                kNumSMs * kNumDispatchWarps;
                            uint32_t prior_transactions = 0;
                            #pragma unroll 1
                            for (uint32_t expert_idx = 0;
                                 expert_idx < kNumExperts;
                                 ++expert_idx) {
                                const uint32_t num_tokens =
                                    static_cast<uint32_t>(
                                        __ldg(
                                            expert_counts +
                                            expert_idx));
                                if (first_token_idx < num_tokens) {
                                    prior_transactions +=
                                        1 +
                                        (num_tokens - 1 -
                                         first_token_idx) /
                                            kInitialDispatchStride;
                                }
                            }
                            pull_phase = prior_transactions & 1u;
                        }
                    }
                    uint32_t pool_block_offset = 0;
                    #pragma unroll 1
                    for (uint32_t expert_idx = 0;
                         expert_idx < kNumExperts; ++expert_idx) {
                        const uint32_t num_tokens =
                            static_cast<uint32_t>(
                                __ldg(expert_counts + expert_idx));
                        for (uint32_t token_idx =
                                 blockIdx.x *
                                     num_route_exact_x_dispatch_warps +
                                 dispatch_warp_idx;
                             token_idx < num_tokens;
                             token_idx +=
                                 kNumSMs *
                                     num_route_exact_x_dispatch_warps) {
                            const uint32_t pool_row =
                                pool_block_offset * BLOCK_M +
                                token_idx;
                            while (ptx::ld_acq(
                                       route_ready_states + pool_row) !=
                                   route_ready_epoch) {
                            }
                            const auto metadata =
                                token_src_metadata[pool_row];
                            const auto* remote_x =
                                backward_sym_buffer.map(
                                    backward_x +
                                        static_cast<uint64_t>(
                                            metadata.token_idx) *
                                        kHidden,
                                    metadata.rank_idx);
                            if (cute::elect_one_sync()) {
                                ptx::tma_load_1d(
                                    pull_buffer, remote_x,
                                    pull_mbarrier,
                                    kHidden * sizeof(cd_dtype_t));
                                ptx::mbarrier_arrive_and_set_tx(
                                    pull_mbarrier,
                                    kHidden * sizeof(cd_dtype_t));
                                ptx::mbarrier_wait_and_flip_phase(
                                    pull_mbarrier, pull_phase);
                                ptx::tma_store_1d(
                                    x_pool_output +
                                        static_cast<uint64_t>(pool_row) *
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
                }
              }
              trace_end(14);
            }

            if constexpr (
                kNumRanks > 1 && !kBF16Mode && kExactSourceX) {
              if (late_exact_source_x) {
                // Route groups publish row-level readiness to the exact-X TMA
                // warps above, so exact source dispatch is already complete
                // when this grid join retires. The source combine plane can
                // now be cleared for direct W13 grad-x publication.
                full_grid_phase_barrier(kTraceSiteCount);
                prepare_direct_grad_x_planes();
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
                    #pragma unroll 1
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
                    constexpr uint32_t kNumSchedulePasses =
                        kOnDemandResidualWeightCache
                            ? 2
                            : 1;
                    // Keep W13's packed producers in the same dependency-
                    // ordered prefix as W2. The even number of N tiles keeps
                    // both CTAs in every UMMA cluster on the same pass.
                    #pragma unroll 1
                    for (uint32_t schedule_pass = 0;
                         schedule_pass <
                             kNumSchedulePasses;
                         ++schedule_pass) {
                        uint32_t next_assigned_block =
                            blockIdx.x;
                        uint32_t global_block = 0;
                        uint32_t pool_block_offset = 0;
                        // The inline MXFP4 dequant body is large; runtime-loop
                        // the 112 experts instead of cloning it 112 times.
                        #pragma unroll 1
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
                            const uint32_t first_m_block =
                                kOnDemandResidualWeightCache
                                    ? schedule_pass
                                    : 0;
                            const uint32_t scheduled_m_blocks =
                                schedule_pass == 0
                                    ? cute::min(
                                          num_m_blocks, 1u)
                                    : num_m_blocks -
                                          cute::min(
                                              num_m_blocks,
                                              1u);
                            const uint32_t expert_blocks =
                                (kOnDemandResidualWeightCache
                                     ? scheduled_m_blocks
                                     : num_m_blocks) *
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
                                        first_m_block +
                                        local_block /
                                            kNumW13DgradBlockNs;
                                const uint32_t
                                    n_block_idx =
                                        local_block %
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
                    }
                };

            trace_begin(15);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(15);
            if (warp_idx == 0 &&
                cute::elect_one_sync()) {
                if constexpr (
                    kResidualMXFP8Dgrad &&
                    !kEarlyW2Wgrad) {
                    // W13 reuses the W2 transaction barriers. Invalidate the
                    // completed objects before resetting their phase; a
                    // direct mbarrier.init on a still-valid object is
                    // undefined and can deadlock after harmless code-layout
                    // changes perturb the producer timing.
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumStages; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                full_barriers[i]));
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                empty_barriers[i]));
                    }
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            weight_load_barrier));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            residual_mma_barrier));
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kNumEpilogueStages; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                tmem_full_barriers[i]));
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                tmem_empty_barriers[i]));
                    }
                }
                #pragma unroll
                for (uint32_t i = 0;
                     i < kNumStages; ++i) {
                    full_barriers[i]->init(
                        kNumDgradFullBarrierArrivals);
                    empty_barriers[i]->init(1);
                }
                if constexpr (kResidualMXFP8Dgrad) {
                    weight_load_barrier->init(1);
                    residual_mma_barrier->init(1);
                }
                #pragma unroll
                for (uint32_t i = 0;
                     i < kNumEpilogueStages; ++i) {
                    tmem_full_barriers[i]->init(1);
                    tmem_empty_barriers[i]->init(
                        2 *
                        kNumW13EpilogueThreads);
                }
                cutlass::arch::fence_barrier_init();
            }
            if constexpr (kEarlyW2Wgrad) {
                if (warp_idx == 1)
                    Allocator().allocate(
                        kNumTmemCols, tmem_ptr_in_smem);
            }
            trace_begin(16);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(16);
            trace_begin(21);

            stage_idx = 0;
            phase = 0;
            if (
                warp_idx == 0 ||
                (kResidualMXFP8Dgrad &&
                 (!kBuildW13ResidualActsOnce ||
                  kOnDemandResidualWeightCache) &&
                 (warp_idx == 1 ||
                  (warp_idx >= 3 && warp_idx < 8)))) {
                uint32_t weight_source_phase = 0;
                for_each_w13_dgrad_block(
                    [&](const uint32_t& expert_idx,
                        const uint32_t&
                            pool_block_offset,
                        const uint32_t&
                            m_block_idx,
                        const uint32_t& n_block_idx,
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
                                const uint32_t producer_warp_idx =
                                    warp_idx == 0
                                    ? 0
                                    : (warp_idx == 1 ? 1 : warp_idx - 1);
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
                                if constexpr (!kResidualMXFP8Dgrad) {
                                    if (cute::elect_one_sync()) {
                                        tma::copy<
                                            DGRAD_BLOCK_K,
                                            LOAD_BLOCK_M,
                                            DGRAD_BLOCK_K *
                                                sizeof(cd_dtype_t),
                                            cd_dtype_t>(
                                            &tensor_map_grad_gate_up,
                                            full_barriers[stage_idx],
                                            smem_dgrad_a[stage_idx],
                                            split_idx *
                                                    ((2 *
                                                      kIntermediateHidden) /
                                                     kNumW13DgradSplits) +
                                                k_block_idx *
                                                    DGRAD_BLOCK_K,
                                            m_idx, 2);
                                        if (is_leader_cta) {
                                            full_barriers[stage_idx]
                                                ->arrive_and_expect_tx(
                                                    SMEM_A_SIZE_PER_STAGE *
                                                    2);
                                        } else {
                                            full_barriers[stage_idx]
                                                ->arrive(0u);
                                        }
                                    }
                                } else if constexpr (
                                    kBuildW13ResidualActsOnce &&
                                    !kOnDemandResidualWeightCache) {
                                    const uint32_t global_k_block_idx =
                                        split_idx *
                                            ((2 * kIntermediateHidden) /
                                             (DGRAD_BLOCK_K *
                                              kNumW13DgradSplits)) +
                                        k_block_idx;
                                    if (producer_warp_idx == 0 &&
                                        cute::elect_one_sync()) {
                                        tma::copy<
                                            DGRAD_BLOCK_K,
                                            LOAD_BLOCK_M,
                                            128,
                                            residual_dgrad_dtype_t>(
                                            &tensor_map_weights,
                                            full_barriers[stage_idx],
                                            reinterpret_cast<
                                                residual_dgrad_dtype_t*>(
                                                smem_dgrad_a[stage_idx]),
                                            global_k_block_idx *
                                                DGRAD_BLOCK_K,
                                            m_idx, 2);
                                        if constexpr (kEarlyW2Wgrad) {
                                            tma::copy<
                                                DGRAD_BLOCK_K,
                                                LOAD_BLOCK_M, 128,
                                                residual_dgrad_dtype_t>(
                                                &tensor_map_output,
                                                full_barriers[stage_idx],
                                                smem_dgrad_a_residual[stage_idx],
                                                global_k_block_idx *
                                                    DGRAD_BLOCK_K,
                                                m_idx, 2);
                                        } else {
                                            tma::copy<
                                                DGRAD_BLOCK_K,
                                                LOAD_BLOCK_M, 128,
                                                residual_dgrad_dtype_t>(
                                                &tensor_map_weights,
                                                full_barriers[stage_idx],
                                                smem_dgrad_a_residual[stage_idx],
                                                global_k_block_idx *
                                                    DGRAD_BLOCK_K,
                                                num_acts_rows + m_idx, 2);
                                        }
                                        tma::copy<SF_BLOCK_M, 1, 0>(
                                            &tensor_map_weights_sf,
                                            full_barriers[stage_idx],
                                            smem_sfa[stage_idx],
                                            pool_block_idx * SF_BLOCK_M,
                                            global_k_block_idx, 2);
                                        tma::copy<SF_BLOCK_M, 1, 0>(
                                            &tensor_map_weights_sf,
                                            full_barriers[stage_idx],
                                            smem_dgrad_sfa_residual[stage_idx],
                                            pool_block_idx * SF_BLOCK_M,
                                            (2 * kIntermediateHidden) /
                                                    (kGranK * 4) +
                                                global_k_block_idx, 2);
                                        if (is_leader_cta) {
                                            full_barriers[stage_idx]
                                                ->arrive_and_expect_tx(
                                                    2 *
                                                    (SMEM_A_SIZE_PER_STAGE +
                                                     SMEM_RESIDUAL_A_SIZE_PER_STAGE +
                                                     SMEM_SFA_SIZE_PER_STAGE +
                                                     SMEM_RESIDUAL_SFA_SIZE_PER_STAGE));
                                        } else {
                                            full_barriers[stage_idx]
                                                ->arrive(0u);
                                        }
                                    }
                                } else {
                                    constexpr uint32_t
                                        kQuantProducerWarps =
                                            kNumResidualProducerWarps;
                                    constexpr uint32_t
                                        kValuesPerLane = 4;
                                    const uint32_t group_idx =
                                        lane_idx / 8;
                                    const uint32_t lane_in_group =
                                        lane_idx % 8;
                                    const uint32_t effective_m =
                                        math::align(valid_m, 16u);
                                    const uint32_t local_valid_m =
                                        effective_m / 2;
                                    const uint32_t global_k_block_idx =
                                        split_idx *
                                            ((2 * kIntermediateHidden) /
                                             (DGRAD_BLOCK_K *
                                             kNumW13DgradSplits)) +
                                        k_block_idx;
                                    if constexpr (
                                        kBuildW13ResidualActsOnce) {
                                        if (producer_warp_idx == 0 &&
                                            cute::elect_one_sync()) {
                                            tma::copy<
                                                DGRAD_BLOCK_K,
                                                LOAD_BLOCK_M, 128,
                                                residual_dgrad_dtype_t>(
                                                &tensor_map_weights,
                                                full_barriers[stage_idx],
                                                reinterpret_cast<
                                                    residual_dgrad_dtype_t*>(
                                                    smem_dgrad_a[stage_idx]),
                                                global_k_block_idx *
                                                    DGRAD_BLOCK_K,
                                                m_idx, 2);
                                            if constexpr (kEarlyW2Wgrad) {
                                                tma::copy<
                                                    DGRAD_BLOCK_K,
                                                    LOAD_BLOCK_M, 128,
                                                    residual_dgrad_dtype_t>(
                                                    &tensor_map_output,
                                                    full_barriers[stage_idx],
                                                    smem_dgrad_a_residual[stage_idx],
                                                    global_k_block_idx *
                                                        DGRAD_BLOCK_K,
                                                    m_idx, 2);
                                            } else {
                                                tma::copy<
                                                    DGRAD_BLOCK_K,
                                                    LOAD_BLOCK_M, 128,
                                                    residual_dgrad_dtype_t>(
                                                    &tensor_map_weights,
                                                    full_barriers[stage_idx],
                                                    smem_dgrad_a_residual[stage_idx],
                                                    global_k_block_idx *
                                                        DGRAD_BLOCK_K,
                                                    num_acts_rows + m_idx, 2);
                                            }
                                            tma::copy<SF_BLOCK_M, 1, 0>(
                                                &tensor_map_weights_sf,
                                                full_barriers[stage_idx],
                                                smem_sfa[stage_idx],
                                                pool_block_idx * SF_BLOCK_M,
                                                global_k_block_idx, 2);
                                            tma::copy<SF_BLOCK_M, 1, 0>(
                                                &tensor_map_weights_sf,
                                                full_barriers[stage_idx],
                                                smem_dgrad_sfa_residual[stage_idx],
                                                pool_block_idx * SF_BLOCK_M,
                                                (2 * kIntermediateHidden) /
                                                        (kGranK * 4) +
                                                    global_k_block_idx, 2);
                                            if (is_leader_cta) {
                                                full_barriers[stage_idx]
                                                    ->arrive_and_expect_tx(
                                                        2 *
                                                        (SMEM_A_SIZE_PER_STAGE +
                                                         SMEM_RESIDUAL_A_SIZE_PER_STAGE +
                                                         SMEM_SFA_SIZE_PER_STAGE +
                                                         SMEM_RESIDUAL_SFA_SIZE_PER_STAGE));
                                            } else {
                                                full_barriers[stage_idx]
                                                    ->arrive(0u);
                                            }
                                        }
                                    }
                                    bool build_weight_tile =
                                        !kPrefixedResidualWeightCache;
                                    const uint32_t weight_tile_idx =
                                        (expert_idx *
                                             ((2 * kIntermediateHidden) /
                                              DGRAD_BLOCK_K) +
                                         global_k_block_idx) *
                                            kNumW13DgradBlockNs +
                                        n_block_idx;
                                    const uint32_t w13_launch_epoch =
                                        launch_epoch ^ 0x80000000u;
                                    if constexpr (
                                        kOnDemandResidualWeightCache) {
                                        // SM100_TMA_2SM_LOAD requires both
                                        // CTAs in the cluster to issue the
                                        // transfer.  Elect the first M wave
                                        // as a producer cluster so neither
                                        // peer waits while the other enters
                                        // the paired TMA transaction.
                                        build_weight_tile =
                                            m_block_idx == 0;
                                        if (producer_warp_idx == 0 &&
                                            lane_idx == 0 &&
                                            !build_weight_tile) {
                                            auto* state =
                                                weight_tile_states +
                                                kNumW2WeightTileStates +
                                                weight_tile_idx;
                                            const uint64_t cache_wait_start =
                                                clock64();
                                            while (ptx::ld_acq(state) !=
                                                   w13_launch_epoch) {
                                                if (clock64() -
                                                        cache_wait_start >
                                                    4000000000ull) {
                                                    printf(
                                                        "K3 W13 cache wait timeout rank=%u sm=%u tile=%u state=%u ready=%u\n",
                                                        backward_sym_buffer.rank_idx,
                                                        blockIdx.x,
                                                        weight_tile_idx,
                                                        ptx::ld_acq(state),
                                                        w13_launch_epoch);
                                                    asm volatile("trap;");
                                                }
                                            }
                                        }
                                        cutlass::arch::NamedBarrier::sync(
                                            kNumResidualProducerThreads,
                                            kResidualWeightCacheBarrier);
                                    }
                                    if constexpr (
                                        kInlineResidualMXFP8Dgrad) {
                                        if (producer_warp_idx == 0 &&
                                            build_weight_tile &&
                                            cute::elect_one_sync()) {
                                            tma::copy<
                                                LOAD_BLOCK_N / 2,
                                                DGRAD_BLOCK_K, 0, int8_t>(
                                                &tensor_map_w13_weights,
                                                weight_load_barrier,
                                                smem_dgrad_weight_source,
                                                n_block_idx *
                                                    (LOAD_BLOCK_N / 2),
                                                expert_idx *
                                                        (2 *
                                                         kIntermediateHidden) +
                                                    global_k_block_idx *
                                                        DGRAD_BLOCK_K);
                                            tma::copy<
                                                LOAD_BLOCK_N / kGranK,
                                                DGRAD_BLOCK_K, 0, float>(
                                                &tensor_map_w13_scales,
                                                weight_load_barrier,
                                                smem_dgrad_weight_scale_source,
                                                n_block_idx *
                                                    (LOAD_BLOCK_N / kGranK),
                                                expert_idx *
                                                        (2 *
                                                         kIntermediateHidden) +
                                                    global_k_block_idx *
                                                        DGRAD_BLOCK_K);
                                            weight_load_barrier
                                                ->arrive_and_expect_tx(
                                                    SMEM_DGRAD_WEIGHT_SOURCE_SIZE +
                                                    SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE);
                                        }
                                    }
                                    if constexpr (
                                        !kBuildW13ResidualActsOnce) {
                                    auto* primary_bytes =
                                        reinterpret_cast<uint8_t*>(
                                            smem_dgrad_a[stage_idx]);
                                    auto* residual_bytes =
                                        reinterpret_cast<uint8_t*>(
                                            smem_dgrad_a_residual[
                                                stage_idx]);
                                    // Keep every UTCCP-visible padding row
                                    // finite before publishing valid scales.
                                    const uint32_t producer_thread_idx =
                                        producer_warp_idx * 32 +
                                        lane_idx;
                                    for (uint32_t sf_idx =
                                             producer_thread_idx;
                                         sf_idx < SF_BLOCK_M;
                                         sf_idx +=
                                             kNumResidualProducerThreads) {
                                        smem_sfa[stage_idx][sf_idx] =
                                            0x7f7f7f7fu;
                                        smem_dgrad_sfa_residual[
                                            stage_idx][sf_idx] =
                                            0x7f7f7f7fu;
                                    }
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumResidualProducerThreads,
                                        kInlineWeightProducerBarrier);
                                    const uint32_t num_quant_rows =
                                        is_leader_cta
                                        ? effective_m
                                        : local_valid_m;
                                    for (uint32_t local_m =
                                             producer_warp_idx;
                                         local_m < num_quant_rows;
                                         local_m +=
                                             kQuantProducerWarps) {
                                        const uint32_t full_m =
                                            local_m +
                                            (is_leader_cta
                                                 ? 0
                                                 : local_valid_m);
                                        const bool valid_row =
                                            full_m < valid_m;
                                        const bool store_data =
                                            is_leader_cta
                                            ? full_m < local_valid_m
                                            : true;
                                        const uint32_t local_store_m =
                                            is_leader_cta
                                            ? full_m
                                            : local_m;
                                        const uint32_t global_k =
                                            global_k_block_idx *
                                                DGRAD_BLOCK_K +
                                            group_idx * 32 +
                                            lane_in_group *
                                                kValuesPerLane;
                                        float values[kValuesPerLane];
                                        #pragma unroll
                                        for (uint32_t i = 0;
                                             i < kValuesPerLane; ++i) {
                                            values[i] = valid_row
                                                ? static_cast<float>(
                                                      grad_gate_up_output[
                                                          static_cast<
                                                              uint64_t>(
                                                              pool_block_idx *
                                                                  BLOCK_M +
                                                              full_m) *
                                                              (2 *
                                                               kIntermediateHidden) +
                                                          global_k + i])
                                                : 0.0f;
                                        }
                                        float primary_amax = 0.0f;
                                        #pragma unroll
                                        for (uint32_t i = 0;
                                             i < kValuesPerLane; ++i)
                                            primary_amax = cute::max(
                                                primary_amax,
                                                cute::abs(values[i]));
                                        #pragma unroll
                                        for (uint32_t offset = 4;
                                             offset > 0; offset >>= 1)
                                            primary_amax = cute::max(
                                                primary_amax,
                                                __shfl_xor_sync(
                                                    0xffffffff,
                                                    primary_amax,
                                                    offset, 8));
                                        float2 primary_sf_pair;
                                        float2 primary_sf_inv_pair;
                                        math::get_e4m3_sf_and_sf_inv(
                                            make_float2(
                                                primary_amax, 0.0f),
                                            primary_sf_pair,
                                            primary_sf_inv_pair);
                                        const float primary_sf =
                                            primary_sf_pair.x;
                                        const auto primary =
                                            __nv_fp8x4_e4m3(make_float4(
                                                values[0] *
                                                    primary_sf_inv_pair.x,
                                                values[1] *
                                                    primary_sf_inv_pair.x,
                                                values[2] *
                                                    primary_sf_inv_pair.x,
                                                values[3] *
                                                    primary_sf_inv_pair.x));
                                        const float4 primary_float =
                                            static_cast<float4>(primary);
                                        float residual[kValuesPerLane] = {
                                            values[0] -
                                                primary_float.x *
                                                    primary_sf,
                                            values[1] -
                                                primary_float.y *
                                                    primary_sf,
                                            values[2] -
                                                primary_float.z *
                                                    primary_sf,
                                            values[3] -
                                                primary_float.w *
                                                    primary_sf,
                                        };
                                        float residual_amax = 0.0f;
                                        #pragma unroll
                                        for (uint32_t i = 0;
                                             i < kValuesPerLane; ++i)
                                            residual_amax = cute::max(
                                                residual_amax,
                                                cute::abs(residual[i]));
                                        #pragma unroll
                                        for (uint32_t offset = 4;
                                             offset > 0; offset >>= 1)
                                            residual_amax = cute::max(
                                                residual_amax,
                                                __shfl_xor_sync(
                                                    0xffffffff,
                                                    residual_amax,
                                                    offset, 8));
                                        float2 residual_sf_pair;
                                        float2 residual_sf_inv_pair;
                                        math::get_e4m3_sf_and_sf_inv(
                                            make_float2(
                                                residual_amax, 0.0f),
                                            residual_sf_pair,
                                            residual_sf_inv_pair);
                                        const auto residual_quantized =
                                            __nv_fp8x4_e4m3(make_float4(
                                                residual[0] *
                                                    residual_sf_inv_pair.x,
                                                residual[1] *
                                                    residual_sf_inv_pair.x,
                                                residual[2] *
                                                    residual_sf_inv_pair.x,
                                                residual[3] *
                                                    residual_sf_inv_pair.x));
                                        const uint32_t row =
                                            local_store_m & 7u;
                                        const uint32_t logical_k_byte =
                                            group_idx * 32 +
                                            lane_in_group *
                                                kValuesPerLane;
                                        const uint32_t byte_offset =
                                            (local_store_m >> 3) *
                                                8 * 128 +
                                            row * 128 +
                                            ((logical_k_byte >> 4) ^ row) *
                                                16 +
                                            (logical_k_byte & 15);
                                        if (store_data) {
                                            *reinterpret_cast<uint32_t*>(
                                                primary_bytes + byte_offset) =
                                                primary.__x;
                                            *reinterpret_cast<uint32_t*>(
                                                residual_bytes + byte_offset) =
                                                residual_quantized.__x;
                                        }
                                        const uint32_t primary_scale_byte =
                                            (*reinterpret_cast<
                                                 const uint32_t*>(
                                                 &primary_sf)) >> 23;
                                        const uint32_t residual_scale_byte =
                                            (*reinterpret_cast<
                                                 const uint32_t*>(
                                                 &residual_sf_pair.x)) >> 23;
                                        uint32_t primary_scale_word = 0;
                                        uint32_t residual_scale_word = 0;
                                        #pragma unroll
                                        for (uint32_t group = 0;
                                             group < 4; ++group) {
                                            primary_scale_word |=
                                                __shfl_sync(
                                                    0xffffffff,
                                                    primary_scale_byte,
                                                    group * 8) <<
                                                (group * 8);
                                            residual_scale_word |=
                                                __shfl_sync(
                                                    0xffffffff,
                                                    residual_scale_byte,
                                                    group * 8) <<
                                                (group * 8);
                                        }
                                        if (valid_row && lane_idx == 0) {
                                            const uint32_t sf_m =
                                                (full_m & ~127u) +
                                                (full_m & 31u) * 4 +
                                                ((full_m >> 5) & 3u);
                                            smem_sfa[stage_idx][sf_m] =
                                                primary_scale_word;
                                            smem_dgrad_sfa_residual[
                                                stage_idx][sf_m] =
                                                residual_scale_word;
                                            if (is_leader_cta) {
                                                constexpr uint32_t
                                                    kFollowerCTARank = 1;
                                                store_cluster_uint32(
                                                    &smem_sfa[
                                                        stage_idx][sf_m],
                                                    kFollowerCTARank,
                                                    primary_scale_word);
                                                store_cluster_uint32(
                                                    &smem_dgrad_sfa_residual[
                                                        stage_idx][sf_m],
                                                    kFollowerCTARank,
                                                    residual_scale_word);
                                            }
                                        }
                                    }
                                    cutlass::arch::
                                        fence_view_async_shared();
                                    __threadfence_cluster();
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumResidualProducerThreads,
                                        kInlineWeightProducerBarrier);
                                    if (producer_warp_idx == 0 &&
                                        cute::elect_one_sync())
                                        full_barriers[stage_idx]
                                            ->arrive(0u);
                                    }

                                    if constexpr (
                                        kInlineResidualMXFP8Dgrad) {
                                      if (build_weight_tile) {
                                      weight_load_barrier->wait(
                                          weight_source_phase);
                                    // Re-encode the combined [W1; W3] tile
                                    // along its transposed dgrad K axis. The
                                    // source expert switches at intermediate,
                                    // matching the exact Triton prototype but
                                    // keeping the tile entirely in shared
                                    // memory inside this persistent launch.
                                    auto* weight_bytes =
                                        reinterpret_cast<uint8_t*>(
                                            smem_dgrad_b[stage_idx]);
                                    if constexpr (
                                        kOnDemandResidualWeightCache) {
                                        auto* cached_q =
                                            reinterpret_cast<uint8_t*>(
                                                w13_dequant_scratch);
                                        auto* cached_sf =
                                            reinterpret_cast<uint32_t*>(
                                                cached_q +
                                                static_cast<uint64_t>(
                                                    kNumExperts) *
                                                    kHidden *
                                                    (2 *
                                                     kIntermediateHidden));
                                        k3_mxfp4_to_mxfp8_transposed_tile(
                                            smem_dgrad_weight_source,
                                            smem_dgrad_weight_scale_source,
                                            weight_bytes,
                                            smem_sfb[stage_idx],
                                            cached_q +
                                                static_cast<uint64_t>(
                                                    expert_idx) *
                                                    kHidden *
                                                    (2 *
                                                     kIntermediateHidden),
                                            cached_sf +
                                                static_cast<uint64_t>(
                                                    expert_idx) *
                                                    kHidden *
                                                    ((2 *
                                                      kIntermediateHidden) /
                                                     DGRAD_BLOCK_K),
                                            kHidden,
                                            2 * kIntermediateHidden,
                                            n_block_idx,
                                            global_k_block_idx,
                                            producer_warp_idx * 32 +
                                                lane_idx);
                                    } else {
                                    for (uint32_t local_n =
                                             producer_warp_idx;
                                         local_n < LOAD_BLOCK_N;
                                         local_n +=
                                             kNumResidualProducerWarps) {
                                        const uint32_t global_n =
                                            n_block_idx * LOAD_BLOCK_N +
                                            local_n;
                                        uint16_t weight_bf16_bits[
                                            kValuesPerLane];
                                        uint32_t weight_amax_bits = 0;
                                        #pragma unroll
                                        for (uint32_t i = 0;
                                             i < kValuesPerLane; ++i) {
                                            const uint8_t packed =
                                                static_cast<uint8_t>(
                                                    smem_dgrad_weight_source[
                                                        (group_idx * 32 +
                                                         lane_in_group *
                                                             kValuesPerLane +
                                                         i) *
                                                            (LOAD_BLOCK_N / 2) +
                                                        local_n / 2]);
                                            const uint32_t source_scale_bits =
                                                *reinterpret_cast<
                                                    const uint32_t*>(
                                                    &smem_dgrad_weight_scale_source[
                                                    (group_idx * 32 +
                                                     lane_in_group *
                                                         kValuesPerLane +
                                                     i) *
                                                        (LOAD_BLOCK_N /
                                                         kGranK) +
                                                    local_n / kGranK]);
                                            const uint8_t nibble =
                                                (packed >>
                                                 ((global_n & 1u) * 4)) &
                                                0xf;
                                            weight_bf16_bits[i] =
                                                k3_mxfp4_bf16_bits(
                                                    nibble,
                                                    source_scale_bits);
                                            weight_amax_bits = cute::max(
                                                weight_amax_bits,
                                                static_cast<uint32_t>(
                                                    weight_bf16_bits[i] &
                                                    0x7fff));
                                        }

                                        #pragma unroll
                                        for (uint32_t offset = 4;
                                             offset > 0; offset >>= 1)
                                            weight_amax_bits = cute::max(
                                                weight_amax_bits,
                                                __shfl_xor_sync(
                                                    0xffffffff,
                                                    weight_amax_bits,
                                                    offset, 8));
                                        const uint32_t weight_scale_byte =
                                            cute::max(
                                                88,
                                                static_cast<int>(
                                                    (weight_amax_bits >> 7) &
                                                    0xff) - 8);
                                        const uint32_t weight_quantized =
                                            k3_quantize_bf16x4_e4m3(
                                                weight_bf16_bits,
                                                weight_scale_byte);
                                        const uint32_t weight_row =
                                            local_n & 7u;
                                        const uint32_t logical_k_byte =
                                            group_idx * 32 +
                                            lane_in_group *
                                                kValuesPerLane;
                                        const uint32_t weight_byte_offset =
                                            (local_n >> 3) * 8 * 128 +
                                            weight_row * 128 +
                                            ((logical_k_byte >> 4) ^
                                             weight_row) *
                                                16 +
                                            (logical_k_byte & 15);
                                        *reinterpret_cast<uint32_t*>(
                                            weight_bytes +
                                            weight_byte_offset) =
                                            weight_quantized;
                                        if constexpr (
                                            kOnDemandResidualWeightCache) {
                                            auto* cached_q =
                                                reinterpret_cast<uint8_t*>(
                                                    w13_dequant_scratch);
                                            *reinterpret_cast<uint32_t*>(
                                                cached_q +
                                                (static_cast<uint64_t>(
                                                     expert_idx) *
                                                     kHidden +
                                                 global_n) *
                                                    (2 *
                                                     kIntermediateHidden) +
                                                global_k_block_idx *
                                                    DGRAD_BLOCK_K +
                                                logical_k_byte) =
                                                weight_quantized;
                                        }

                                        uint32_t weight_scale_word = 0;
                                        #pragma unroll
                                        for (uint32_t group = 0;
                                             group < 4; ++group)
                                            weight_scale_word |=
                                                __shfl_sync(
                                                    0xffffffff,
                                                    weight_scale_byte,
                                                    group * 8) <<
                                                (group * 8);
                                        if (lane_idx == 0) {
                                            const uint32_t sf_n =
                                                (local_n & 31u) * 4 +
                                                ((local_n >> 5) & 3u);
                                            smem_sfb[stage_idx][sf_n] =
                                                weight_scale_word;
                                            if constexpr (
                                                kOnDemandResidualWeightCache) {
                                                auto* cached_q =
                                                    reinterpret_cast<uint8_t*>(
                                                        w13_dequant_scratch);
                                                auto* cached_sf =
                                                    reinterpret_cast<uint32_t*>(
                                                        cached_q +
                                                        static_cast<uint64_t>(
                                                            kNumExperts) *
                                                            kHidden *
                                                            (2 *
                                                             kIntermediateHidden));
                                                const uint32_t global_sf_n =
                                                    (global_n & ~127u) +
                                                    (global_n & 31u) * 4 +
                                                    ((global_n >> 5) & 3u);
                                                cached_sf[
                                                    static_cast<uint64_t>(
                                                        expert_idx) *
                                                        kHidden *
                                                        ((2 *
                                                          kIntermediateHidden) /
                                                         DGRAD_BLOCK_K) +
                                                    global_k_block_idx *
                                                        kHidden +
                                                    global_sf_n] =
                                                    weight_scale_word;
                                            }
                                        }
                                    }
                                    }
                                    cutlass::arch::
                                        fence_view_async_shared();
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumResidualProducerThreads,
                                        kResidualWeightProducerBarrier);
                                      if (producer_warp_idx == 0 &&
                                          cute::elect_one_sync()) {
                                          if constexpr (
                                              kOnDemandResidualWeightCache) {
                                              asm volatile(
                                                  "fence.proxy.async.global;"
                                                  ::: "memory");
                                              __threadfence();
                                              asm volatile(
                                                  "st.release.gpu.global.u32 [%0], %1;"
                                                  :: "l"(
                                                         weight_tile_states +
                                                         kNumW2WeightTileStates +
                                                         weight_tile_idx),
                                                     "r"(w13_launch_epoch)
                                                  : "memory");
                                          }
                                          full_barriers[stage_idx]
                                              ->arrive(0u);
                                      }
                                      weight_source_phase ^= 1;
                                      }
                                    }
                                    if ((!kInlineResidualMXFP8Dgrad ||
                                         !build_weight_tile) &&
                                        producer_warp_idx == 0 &&
                                        cute::elect_one_sync()) {
                                        tma::copy<
                                            DGRAD_BLOCK_K, LOAD_BLOCK_N,
                                            128, residual_dgrad_dtype_t>(
                                            &tensor_map_w13_dgrad_weights,
                                            full_barriers[stage_idx],
                                            reinterpret_cast<
                                                residual_dgrad_dtype_t*>(
                                                smem_dgrad_b[stage_idx]),
                                            global_k_block_idx *
                                                DGRAD_BLOCK_K,
                                            expert_idx * kHidden +
                                                n_block_idx * BLOCK_N,
                                            2);
                                        tma::copy<BLOCK_N, 1, 0>(
                                            &tensor_map_w13_dgrad_weights_sf,
                                            full_barriers[stage_idx],
                                            smem_sfb[stage_idx],
                                            n_block_idx * BLOCK_N,
                                            expert_idx *
                                                    ((2 *
                                                      kIntermediateHidden) /
                                                     (kGranK * 4)) +
                                                global_k_block_idx,
                                            2);
                                        if (is_leader_cta) {
                                            full_barriers[stage_idx]
                                                ->arrive_and_expect_tx(
                                                    (SMEM_B_SIZE_PER_STAGE +
                                                     SMEM_SFB_SIZE_PER_STAGE) *
                                                    2);
                                        } else {
                                            full_barriers[stage_idx]
                                                ->arrive(0u);
                                        }
                                    }
                                }
                                __syncwarp();
                            }
                        }
                    });
            } else if (
                warp_idx == 1 ||
                (kInlineWeightDequant && warp_idx >= 3 && warp_idx < 8)) {
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
                                if constexpr (
                                    !kBF16Mode &&
                                    !kInlineWeightDequant &&
                                    !kResidualMXFP8Dgrad) {
                                    while (ptx::ld_acq(
                                               weight_tile_states +
                                               kNumW2WeightTileStates +
                                               weight_tile_idx) !=
                                           w13_launch_epoch) {
                                    }
                                }
                                empty_barriers[stage_idx]
                                    ->wait(phase ^ 1);
                                if constexpr (!kInlineWeightDequant) {
                                    if (cute::elect_one_sync()) {
                                        if constexpr (
                                            kResidualMXFP8Dgrad) {
                                            tma::copy<
                                                DGRAD_BLOCK_K,
                                                LOAD_BLOCK_N,
                                                128,
                                                residual_dgrad_dtype_t>(
                                                &tensor_map_w13_dgrad_weights,
                                                full_barriers[stage_idx],
                                                reinterpret_cast<
                                                    residual_dgrad_dtype_t*>(
                                                    smem_dgrad_b[stage_idx]),
                                                global_k_block_idx *
                                                    DGRAD_BLOCK_K,
                                                expert_idx * kHidden +
                                                    n_block_idx * BLOCK_N,
                                                2);
                                            tma::copy<BLOCK_N, 1, 0>(
                                                &tensor_map_w13_dgrad_weights_sf,
                                                full_barriers[stage_idx],
                                                smem_sfb[stage_idx],
                                                n_block_idx * BLOCK_N,
                                                expert_idx *
                                                        ((2 *
                                                          kIntermediateHidden) /
                                                         (kGranK * 4)) +
                                                    global_k_block_idx,
                                                2);
                                            if (is_leader_cta) {
                                                full_barriers[stage_idx]
                                                    ->arrive_and_expect_tx(
                                                        (SMEM_B_SIZE_PER_STAGE +
                                                         SMEM_SFB_SIZE_PER_STAGE) *
                                                        2);
                                            } else {
                                                full_barriers[stage_idx]
                                                    ->arrive(0u);
                                            }
                                        } else {
                                            tma::copy<
                                                LOAD_BLOCK_N,
                                                DGRAD_BLOCK_K,
                                                DGRAD_BLOCK_K *
                                                    sizeof(
                                                        dgrad_b_dtype_t),
                                                dgrad_b_dtype_t>(
                                                &tensor_map_w13_dequant,
                                                full_barriers[stage_idx],
                                                smem_dgrad_b[stage_idx],
                                                n_block_idx * BLOCK_N,
                                                expert_idx *
                                                        (2 *
                                                         kIntermediateHidden) +
                                                    global_k_block_idx *
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
                                    }
                                } else {
                                    constexpr uint32_t kPairsPerTile =
                                        LOAD_BLOCK_N * DGRAD_BLOCK_K / 2;
                                    auto* smem_b_bytes =
                                        reinterpret_cast<uint8_t*>(
                                            smem_dgrad_b[stage_idx]);
                                    const uint32_t producer_warp_idx =
                                        warp_idx == 1 ? 0 : warp_idx - 2;
                                    for (uint32_t pair_idx =
                                             producer_warp_idx * 32 + lane_idx;
                                         pair_idx < kPairsPerTile;
                                         pair_idx +=
                                             kNumInlineWeightProducerThreads) {
                                        const uint32_t local_k =
                                            pair_idx / (LOAD_BLOCK_N / 2);
                                        const uint32_t local_n_pair =
                                            pair_idx % (LOAD_BLOCK_N / 2);
                                        const uint32_t global_k =
                                            global_k_block_idx *
                                                DGRAD_BLOCK_K +
                                            local_k;
                                        const uint32_t global_n_pair =
                                            n_block_idx *
                                                (LOAD_BLOCK_N / 2) +
                                            local_n_pair;
                                        const uint8_t packed =
                                            static_cast<uint8_t>(__ldg(
                                                w13_weights +
                                                (static_cast<uint64_t>(
                                                     expert_idx) *
                                                     (2 *
                                                      kIntermediateHidden) +
                                                 global_k) *
                                                    (kHidden / 2) +
                                                global_n_pair));
                                        const float scale = __ldg(
                                            w13_scales +
                                            (static_cast<uint64_t>(
                                                 expert_idx) *
                                                 (2 *
                                                  kIntermediateHidden) +
                                             global_k) *
                                                (kHidden / 32) +
                                            n_block_idx *
                                                (LOAD_BLOCK_N / 32) +
                                            (local_n_pair * 2) / 32);
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
                                            *reinterpret_cast<__half2*>(
                                                &fp16x2);
                                        value_pair = __hmul2(
                                            value_pair,
                                            __float2half2_rn(scale));
                                        const auto value_pair_bf16 =
                                            __float22bfloat162_rn(
                                                __half22float2(value_pair));
                                        const uint32_t scaled_pair =
                                            *reinterpret_cast<const uint32_t*>(
                                                &value_pair_bf16);
                                        #pragma unroll
                                        for (uint32_t i = 0; i < 2; ++i) {
                                            const uint32_t local_n =
                                                local_n_pair * 2 + i;
                                            // Match tma::copy's pair of
                                            // independently swizzled 64x64
                                            // MN-major atoms.
                                            const uint32_t atom_n =
                                                local_n / 64;
                                            const uint32_t atom_col_byte =
                                                (local_n % 64) *
                                                sizeof(dgrad_b_dtype_t);
                                            const uint32_t row =
                                                local_k & 7;
                                            const uint32_t byte_offset =
                                                atom_n *
                                                    DGRAD_BLOCK_K *
                                                    64 *
                                                    sizeof(dgrad_b_dtype_t) +
                                                (local_k >> 3) * 8 * 128 +
                                                row * 128 +
                                                ((atom_col_byte >> 4) ^ row) *
                                                    16 +
                                                (atom_col_byte & 15);
                                            *reinterpret_cast<uint16_t*>(
                                                smem_b_bytes + byte_offset) =
                                                static_cast<uint16_t>(
                                                    scaled_pair >> (i * 16));
                                        }
                                    }
                                    cutlass::arch::fence_view_async_shared();
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumInlineWeightProducerThreads,
                                        kInlineWeightProducerBarrier);
                                    if (producer_warp_idx == 0 &&
                                        cute::elect_one_sync())
                                        full_barriers[stage_idx]->arrive(0u);
                                }
                                __syncwarp();
                            }
                        }
                    });
            } else if (warp_idx == 2) {
                if (is_leader_cta) {
                  if constexpr (kResidualMXFP8Dgrad) {
                    auto instr_desc =
                        cute::UMMA::make_instr_desc_block_scaled<
                            residual_dgrad_dtype_t,
                            residual_dgrad_dtype_t, float,
                            cutlass::float_ue8m0_t,
                            UMMA_M, UMMA_N,
                            cute::UMMA::Major::K,
                            cute::UMMA::Major::K>();
                    auto sf_desc = mma::sm100::make_sf_desc(nullptr);
                    auto primary_a_desc =
                        mma::sm100::make_umma_desc<
                            cute::UMMA::Major::K,
                            LOAD_BLOCK_M, DGRAD_BLOCK_K, 128>(
                            reinterpret_cast<
                                residual_dgrad_dtype_t*>(
                                smem_dgrad_a[0]),
                            0, 0);
                    auto residual_a_desc =
                        mma::sm100::make_umma_desc<
                            cute::UMMA::Major::K,
                            LOAD_BLOCK_M, DGRAD_BLOCK_K, 128>(
                            smem_dgrad_a_residual[0], 0, 0);
                    auto b_desc = mma::sm100::make_umma_desc<
                        cute::UMMA::Major::K,
                        LOAD_BLOCK_N, DGRAD_BLOCK_K, 128>(
                        reinterpret_cast<residual_dgrad_dtype_t*>(
                            smem_dgrad_b[0]),
                        0, 0);
                    const uint32_t primary_a_desc_lo =
                        lane_idx < kNumStages
                        ? primary_a_desc.lo +
                              lane_idx * SMEM_A_SIZE_PER_STAGE / 16
                        : 0;
                    const uint32_t residual_a_desc_lo =
                        lane_idx < kNumStages
                        ? residual_a_desc.lo +
                              lane_idx *
                                  SMEM_RESIDUAL_A_SIZE_PER_STAGE / 16
                        : 0;
                    const uint32_t b_desc_lo =
                        lane_idx < kNumStages
                        ? b_desc.lo +
                              lane_idx * SMEM_B_SIZE_PER_STAGE / 16
                        : 0;
                    uint32_t current_iter = 0;
                    uint32_t residual_mma_phase = 0;

                    for_each_w13_dgrad_block(
                        [&](const uint32_t&, const uint32_t&,
                            const uint32_t&, const uint32_t&,
                            const uint32_t& valid_m) {
                            mma::sm100::
                                update_instr_desc_with_umma_n(
                                    instr_desc,
                                    math::align(valid_m, 16u));
                            #pragma unroll
                            for (uint32_t split_idx = 0;
                                 split_idx < kNumW13DgradSplits;
                                 ++split_idx) {
                                const uint32_t accum_stage =
                                    current_iter %
                                    kNumEpilogueStages;
                                const uint32_t accum_phase =
                                    (current_iter++ /
                                     kNumEpilogueStages) &
                                    1;
                                tmem_empty_barriers[accum_stage]
                                    ->wait(accum_phase ^ 1);
                                ptx::tcgen05_after_thread_sync();

                                #pragma unroll 1
                                for (uint32_t k_block_idx = 0;
                                     k_block_idx <
                                         (2 * kIntermediateHidden) /
                                             (DGRAD_BLOCK_K *
                                              kNumW13DgradSplits);
                                     advance_pipeline(k_block_idx)) {
                                    full_barriers[stage_idx]
                                        ->wait(phase);
                                    ptx::tcgen05_after_thread_sync();
                                    const uint32_t
                                        primary_a_desc_base =
                                            ptx::exchange(
                                                primary_a_desc_lo,
                                                stage_idx);
                                    const uint32_t
                                        residual_a_desc_base =
                                            ptx::exchange(
                                                residual_a_desc_lo,
                                                stage_idx);
                                    const uint32_t b_desc_base =
                                        ptx::exchange(
                                            b_desc_lo, stage_idx);
                                    if (cute::elect_one_sync()) {
                                        using utccp_t =
                                            cute::
                                                SM100_UTCCP_4x32dp128bit_2cta;
                                        #pragma unroll
                                        for (uint32_t i = 0;
                                             i < SF_BLOCK_M /
                                                     kNumUTCCPAlignedElems;
                                             ++i) {
                                            mma::sm100::
                                                replace_smem_desc_addr(
                                                    sf_desc,
                                                    smem_sfa[stage_idx] +
                                                        i *
                                                            kNumUTCCPAlignedElems);
                                            utccp_t::copy(
                                                sf_desc,
                                                kTmemStartColOfSFA +
                                                    i * 4);
                                            mma::sm100::
                                                replace_smem_desc_addr(
                                                    sf_desc,
                                                    smem_dgrad_sfa_residual[
                                                        stage_idx] +
                                                        i *
                                                            kNumUTCCPAlignedElems);
                                            utccp_t::copy(
                                                sf_desc,
                                                kTmemStartColOfResidualSFA +
                                                    i * 4);
                                        }
                                        mma::sm100::
                                            replace_smem_desc_addr(
                                                sf_desc,
                                                smem_sfb[stage_idx]);
                                        utccp_t::copy(
                                            sf_desc,
                                            kTmemStartColOfSFB);

                                        #pragma unroll
                                        for (uint32_t k = 0;
                                             k < DGRAD_BLOCK_K /
                                                     DGRAD_UMMA_K;
                                             ++k) {
                                            const auto
                                                runtime_instr_desc =
                                                    mma::sm100::
                                                        make_runtime_instr_desc_with_sf_id(
                                                            instr_desc,
                                                            k, k);
                                            primary_a_desc.lo =
                                                mma::sm100::
                                                    advance_umma_desc_lo<
                                                        cute::UMMA::Major::K,
                                                        LOAD_BLOCK_M,
                                                        128,
                                                        residual_dgrad_dtype_t>(
                                                        primary_a_desc_base,
                                                        0,
                                                        k *
                                                            DGRAD_UMMA_K);
                                            b_desc.lo =
                                                mma::sm100::
                                                    advance_umma_desc_lo<
                                                        cute::UMMA::Major::K,
                                                        LOAD_BLOCK_N,
                                                        128,
                                                        residual_dgrad_dtype_t>(
                                                        b_desc_base,
                                                        0,
                                                        k *
                                                            DGRAD_UMMA_K);
                                            ptx::
                                                SM100_MMA_MXF8F6F4_2x1SM_SS::
                                                    fma(
                                                        b_desc,
                                                        primary_a_desc,
                                                        accum_stage *
                                                            UMMA_N,
                                                        k_block_idx > 0 ||
                                                            k > 0,
                                                        runtime_instr_desc,
                                                        kTmemStartColOfSFB,
                                                        kTmemStartColOfSFA);
                                        }
                                    }
                                    __syncwarp();
                                    constexpr uint16_t kCTAMask = 0x3;
                                    cutlass::arch::
                                        umma_arrive_multicast_2x1SM(
                                            reinterpret_cast<uint64_t*>(
                                                residual_mma_barrier),
                                            kCTAMask);
                                    residual_mma_barrier->wait(
                                        residual_mma_phase);
                                    residual_mma_phase ^= 1;
                                    if constexpr (
                                        kApplyResidualDgradCorrection) {
                                      __syncwarp();
                                      if (cute::elect_one_sync()) {
                                        #pragma unroll
                                        for (uint32_t k = 0;
                                             k < DGRAD_BLOCK_K /
                                                     DGRAD_UMMA_K;
                                             ++k) {
                                            const auto
                                                runtime_instr_desc =
                                                    mma::sm100::
                                                        make_runtime_instr_desc_with_sf_id(
                                                            instr_desc,
                                                            k, k);
                                            residual_a_desc.lo =
                                                mma::sm100::
                                                    advance_umma_desc_lo<
                                                        cute::UMMA::Major::K,
                                                        LOAD_BLOCK_M,
                                                        128,
                                                        residual_dgrad_dtype_t>(
                                                        residual_a_desc_base,
                                                        0,
                                                        k *
                                                            DGRAD_UMMA_K);
                                            b_desc.lo =
                                                mma::sm100::
                                                    advance_umma_desc_lo<
                                                        cute::UMMA::Major::K,
                                                        LOAD_BLOCK_N,
                                                        128,
                                                        residual_dgrad_dtype_t>(
                                                        b_desc_base,
                                                        0,
                                                        k *
                                                            DGRAD_UMMA_K);
                                            ptx::
                                                SM100_MMA_MXF8F6F4_2x1SM_SS::
                                                    fma(
                                                        b_desc,
                                                        residual_a_desc,
                                                        accum_stage *
                                                            UMMA_N,
                                                        true,
                                                        runtime_instr_desc,
                                                        kTmemStartColOfSFB,
                                                        kTmemStartColOfResidualSFA);
                                        }
                                      }
                                      __syncwarp();
                                      cutlass::arch::
                                          umma_arrive_multicast_2x1SM(
                                              reinterpret_cast<uint64_t*>(
                                                  residual_mma_barrier),
                                              kCTAMask);
                                      residual_mma_barrier->wait(
                                          residual_mma_phase);
                                      residual_mma_phase ^= 1;
                                    }
                                    if (cute::elect_one_sync()) {
                                        empty_barriers[stage_idx]
                                            ->arrive();
                                        if (k_block_idx ==
                                            (2 * kIntermediateHidden) /
                                                    (DGRAD_BLOCK_K *
                                                     kNumW13DgradSplits) -
                                                1)
                                            tmem_full_barriers[
                                                accum_stage]
                                                ->arrive();
                                    }
                                    __syncwarp();
                                }
                            }
                        });
                    if (current_iter > 0) {
                        const uint32_t last = current_iter - 1;
                        tmem_empty_barriers[
                            last % kNumEpilogueStages]
                            ->wait(
                                (last / kNumEpilogueStages) & 1);
                    }
                  } else {
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
                } else if constexpr (kResidualMXFP8Dgrad) {
                    uint32_t current_iter = 0;
                    uint32_t follower_residual_mma_phase = 0;
                    for_each_w13_dgrad_block(
                        [&](const uint32_t&, const uint32_t&,
                            const uint32_t&, const uint32_t&,
                            const uint32_t&) {
                            #pragma unroll
                            for (uint32_t split_idx = 0;
                                 split_idx < kNumW13DgradSplits;
                                 ++split_idx) {
                                const uint32_t accum_stage =
                                    current_iter++ %
                                    kNumEpilogueStages;
                                #pragma unroll 1
                                for (uint32_t k_block_idx = 0;
                                     k_block_idx <
                                         (2 * kIntermediateHidden) /
                                             (DGRAD_BLOCK_K *
                                              kNumW13DgradSplits);
                                     advance_pipeline(k_block_idx)) {
                                    residual_mma_barrier->wait(
                                        follower_residual_mma_phase);
                                    follower_residual_mma_phase ^= 1;
                                    if constexpr (
                                        kApplyResidualDgradCorrection) {
                                        residual_mma_barrier->wait(
                                            follower_residual_mma_phase);
                                        follower_residual_mma_phase ^= 1;
                                    }
                                    if (cute::elect_one_sync()) {
                                        empty_barriers[stage_idx]
                                            ->arrive();
                                        if (k_block_idx ==
                                            (2 * kIntermediateHidden) /
                                                    (DGRAD_BLOCK_K *
                                                     kNumW13DgradSplits) -
                                                1)
                                            tmem_full_barriers[
                                                accum_stage]
                                                ->arrive();
                                    }
                                    __syncwarp();
                                }
                            }
                        });
                }
            } else if (
                kOverlapRouteGradWithW13 &&
                warp_idx >= kDispatchWarpStart &&
                warp_idx <
                    kDispatchWarpStart +
                        kNumDispatchWarps) {
                // One warp owns one route row, preserving the exact four-lane
                // accumulation and shuffle tree used by the serialized K3
                // reducer.  Only route-to-warp assignment changes; no row's
                // FP32 reduction order changes.
                const uint32_t route_warp_idx =
                    warp_idx - kDispatchWarpStart;
                const uint32_t global_route_warp =
                    blockIdx.x * kNumDispatchWarps +
                    route_warp_idx;
                const uint32_t num_route_warps =
                    kNumSMs * kNumDispatchWarps;
                uint32_t route_pool_block_offset = 0;
                #pragma unroll 1
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts;
                     ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(
                                expert_counts +
                                expert_idx));
                    for (uint32_t token_idx =
                             global_route_warp;
                         token_idx < num_tokens;
                         token_idx += num_route_warps) {
                        const uint32_t pool_row =
                            route_pool_block_offset *
                                BLOCK_M +
                            token_idx;
                        float lane_sums[4] = {
                            0.0f, 0.0f, 0.0f, 0.0f};
                        for (uint32_t col_base =
                                 lane_idx * 4;
                             col_base < kHidden;
                             col_base += 32 * 4) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < 4; ++i) {
                                const uint32_t col =
                                    col_base + i;
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
                                lane_sums[i] =
                                    __fadd_rn(
                                        lane_sums[i],
                                        __fmul_rn(
                                            grad_y,
                                            down));
                            }
                        }
                        float grad_route =
                            __fadd_rn(
                                __fadd_rn(
                                    lane_sums[0],
                                    lane_sums[1]),
                                lane_sums[2]);
                        grad_route =
                            __fadd_rn(
                                grad_route,
                                lane_sums[3]);
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
                        if (lane_idx == 0) {
                            grad_route_output[pool_row] =
                                grad_route;
                            if (backward_grad_route !=
                                nullptr) {
                                const auto metadata =
                                    token_src_metadata[
                                        pool_row];
                                auto* remote_grad_route =
                                    backward_sym_buffer.map(
                                        backward_grad_route +
                                            static_cast<uint64_t>(
                                                metadata
                                                    .token_idx) *
                                                num_topk +
                                            metadata
                                                .topk_idx,
                                        metadata.rank_idx);
                                *remote_grad_route =
                                    grad_route;
                            }
                        }
                        __syncwarp();
                    }
                    route_pool_block_offset +=
                        math::ceil_div(
                            num_tokens, BLOCK_M);
                }
            } else if (warp_idx >= kDgradEpilogueWarpStart) {
                const uint32_t epilogue_warp_idx =
                    warp_idx - kDgradEpilogueWarpStart -
                    (kOverlapRouteGradWithW13 &&
                             warp_idx >=
                                 kDispatchWarpStart +
                                     kNumDispatchWarps
                         ? kNumDispatchWarps
                         : 0);
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
                                    kNumW13EpilogueThreads,
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
                                    kNumW13EpilogueThreads,
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
                                         kNumW13EpilogueThreads) {
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
                                         kNumW13EpilogueThreads) {
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
                                         kNumW13EpilogueThreads) {
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
        } else if constexpr (
            kComputeRouteGrad && !kInputsPrepared &&
            kRouteWeightMode == RouteWeightMode::PostDown &&
            !kRouteGradBeforeW2) {
            // Route gradients are independent of W13 dgrad. Autograd may
            // request router and expert-weight gradients without requesting
            // dX, which compile-time removes the entire W13 dgrad phase. Keep
            // the POST_DOWN dot and its direct symmetric-memory publication
            // alive in that specialization instead of silently returning
            // zeros. The helper preserves the same fixed-order reduction used
            // by the normal training path.
            bf16_mega_moe_reduce_post_down_route<
                kHidden, kNumExperts, BLOCK_M,
                kNumSMs, kNumThreads, kNumRanks,
                kCombineOrderMode>(
                expert_counts,
                grad_y_unweighted_output,
                down_unweighted_output,
                grad_route_output,
                backward_grad_route,
                token_src_metadata,
                num_topk,
                backward_sym_buffer,
                smem_buffer);
        }

        trace_end(21);
        if constexpr (kNumRanks > 1 && !kInlineWgrad) {
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
            constexpr uint32_t kPaddingColumns =
                kEarlyW2Wgrad
                    ? 2 * kIntermediateHidden
                    : kHidden + 3 * kIntermediateHidden;
            uint32_t pad_pool_block_offset = 0;
            uint32_t pad_global_block = 0;
            #pragma unroll 1
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
                                     kPaddingColumns;
                             linear += kNumThreads) {
                            const uint32_t row_delta =
                                linear / kPaddingColumns;
                            const uint32_t col =
                                linear -
                                row_delta * kPaddingColumns;
                            const uint32_t pool_row =
                                pool_block * BLOCK_M +
                                last_valid + row_delta;
                            if constexpr (kEarlyW2Wgrad) {
                                grad_gate_up_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        (2 * kIntermediateHidden) +
                                    col] = cd_dtype_t(0.0f);
                            } else if (col < kHidden) {
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

        const auto clear_empty_wgrad_experts = [&]() {
            // On the first, non-accumulating chunk the dequant workspaces
            // alias the caller-owned dW buffers. Dequant and Kernel B both
            // skip empty experts, so clear only those expert slices here.
            // This replaces a stream-wide 7.4-GB memset for Kimi-K3 EP=8;
            // nonempty experts are overwritten by Kernel B, while subsequent
            // chunks must retain their accumulated gradients.
            constexpr uint64_t kBF16PerVector =
                sizeof(uint4) / sizeof(cd_dtype_t);
            constexpr uint64_t kW2VectorsPerExpert =
                static_cast<uint64_t>(kHidden) *
                kIntermediateHidden / kBF16PerVector;
            constexpr uint64_t kW13VectorsPerExpert =
                static_cast<uint64_t>(2 * kIntermediateHidden) *
                kHidden / kBF16PerVector;
            constexpr uint64_t kVectorsPerExpert =
                kW2VectorsPerExpert + kW13VectorsPerExpert;
            const uint64_t global_thread =
                static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                threadIdx.x;
            constexpr uint64_t kGlobalThreads =
                static_cast<uint64_t>(kNumSMs) * kNumThreads;
            auto* w2_vectors =
                reinterpret_cast<uint4*>(w2_dequant_scratch);
            auto* w13_vectors =
                reinterpret_cast<uint4*>(w13_dequant_scratch);
            const uint4 zero = {0, 0, 0, 0};
            #pragma unroll 1
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                if (__ldg(expert_counts + expert_idx) != 0)
                    continue;
                for (uint64_t linear = global_thread;
                     linear < kVectorsPerExpert;
                     linear += kGlobalThreads) {
                    if (linear < kW2VectorsPerExpert) {
                        w2_vectors[
                            static_cast<uint64_t>(expert_idx) *
                                kW2VectorsPerExpert +
                            linear] = zero;
                    } else {
                        w13_vectors[
                            static_cast<uint64_t>(expert_idx) *
                                kW13VectorsPerExpert +
                            linear - kW2VectorsPerExpert] = zero;
                    }
                }
            }
        };

        // Standalone Kernel B consumes only these three padded operands. Valid
        // rows are fully overwritten above; clear only the final partial block
        // of each expert instead of memset'ing every active scratch prefix.
        if constexpr (kClearWgradPadding)
            clear_wgrad_padding_rows();

        // This compile-time combination identifies the first chunk whose
        // non-inline dequant workspaces are the uninitialized dW outputs.
        // Inline fallback is used only after live accumulation state exists;
        // phase-ordered dequant aliases consumed activation storage instead.
        if constexpr (
            kClearWgradPadding && !kInlineWeightDequant &&
            !kPhaseOrderedWeightDequant) {
            if (clear_empty_wgrad_expert_outputs)
                clear_empty_wgrad_experts();
        }

        if constexpr (kInlineWgrad) {
            // Padding and empty-expert clears are partitioned across the
            // persistent grid.  The inline grouped GEMM can schedule any
            // expert on any SM, so a CTA-local/cluster-local join is not a
            // sufficient readiness edge.  This also retires every remaining
            // reader before the weight-tile epoch array is reused as the
            // grouped K schedule.
            //
            // Join all producer warps *before* thread 0 contributes this CTA's
            // global ticket. Without this pre-join, another CTA can observe
            // the global sense change while nonzero threads here still write
            // Wgrad operands. Publish generic stores to TMA's async proxy as
            // part of that one-time phase handoff.
            asm volatile(
                "fence.proxy.async.global;" ::: "memory");
            __threadfence();
            __syncthreads();
            full_grid_phase_barrier(kTraceSiteCount);
        } else {
            __syncthreads();
        }
        trace_begin(20);
        if constexpr (kInlineWgrad) {
          if (warp_idx == 0 && cute::elect_one_sync()) {
            // The final grouped wgrad body aliases the entire parent dynamic
            // shared allocation. Retire W13's transaction barriers before
            // that alias; initializing a live mbarrier at a different
            // logical layout is undefined on SM100.
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        full_barriers[i]));
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        empty_barriers[i]));
            }
            if constexpr (kResidualMXFP8Dgrad) {
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        weight_load_barrier));
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        residual_mma_barrier));
            }
            #pragma unroll
            for (uint32_t i = 0;
                 i < kNumEpilogueStages; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        tmem_full_barriers[i]));
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        tmem_empty_barriers[i]));
            }
            #pragma unroll
            for (uint32_t i = 0;
                 i < kNumDispatchBarriers; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<
                        Barrier::ValueType const*>(
                        dispatch_barriers[i]));
            }
          }
        }
        comm::cluster_sync_with_relaxed_arrive();
        if (warp_idx == 0)
            Allocator().free(0, kNumTmemCols);
        __syncthreads();

        if constexpr (kInlineWgrad) {
            // Kernel A's weight-tile epochs are dead after both dgrad phases.
            // Reuse their caller-owned storage for the block-padded K-grouped
            // schedule rather than allocating another CUDA tensor.
            if constexpr (!kEarlyW2Wgrad)
                initialize_wgrad_grouped_layout();

            // Remote dX/dRoute stores have already been issued by the dgrad
            // producers. The otherwise-idle W2 combine warps perform the
            // cross-rank publication barrier while UMMA computes dW2.
            // The barrier's epilogue grid join makes SM-0 completion visible
            // before W13's combine warps consume the remote top-k planes.
            if constexpr (!kEarlyW2Wgrad) {
              trace_begin(17);
              run_wgrad.template operator()<kPublishRemoteGradients>(
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    false);
              trace_end(17);
              // The clustered body frees TMEM before returning. Join both
              // CTAs so no peer begins the next allocation while its peer
              // still drains the final W2 TMA store.
              trace_begin(18);
              comm::cluster_sync_with_relaxed_arrive();
              trace_end(18);
            }
            trace_begin(19);
            run_wgrad.template operator()<
                    kEarlyW2Wgrad
                        ? kPublishRemoteGradients
                        : kDirectRemoteGradX,
                    kDirectRemoteGradX ? 64 : 0,
                    kEarlyW2Wgrad &&
                        kDirectRemoteGradX &&
                        kPublishRemoteGradients>(
                    2 * kIntermediateHidden, kHidden,
                    tensor_map_w13_wgrad_a,
                    tensor_map_w13_wgrad_b,
                    tensor_map_w13_wgrad_d,
                    kDirectRemoteGradX);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(19);
        }

        trace_end(20);
        trace_end(0);
    }
#endif
}

}  // namespace deep_gemm
