#pragma once

#ifndef DG_EXPERIMENTAL_K3_RANGE_WGRAD
#define DG_EXPERIMENTAL_K3_RANGE_WGRAD 0
#endif

// Ready-driven K3 BF16 backward prototype.  Keep this disabled by default and
// compile it only for the exact EP=8, non-residual, inline-wgrad specialization
// selected below.  A per-cluster mailbox gives every BF16 scheduler role the
// same dynamically claimed task batches without re-entering the GEMM body.
#ifndef DG_EXPERIMENTAL_K3_READY_WGRAD
#define DG_EXPERIMENTAL_K3_READY_WGRAD 0
#endif
#ifndef DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS
#define DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS 2
#endif
#ifndef DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS 32
#endif
#ifndef DG_EXPERIMENTAL_K3_MULTI_RANGE_DW2_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_MULTI_RANGE_DW2_BATCH_TASKS 16
#endif
#ifndef DG_EXPERIMENTAL_K3_READY_WGRAD_COMM_TAIL_TASKS
#define DG_EXPERIMENTAL_K3_READY_WGRAD_COMM_TAIL_TASKS 4
#endif

#ifndef DG_EXPERIMENTAL_K3_RANGE_W13_DGRAD_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_RANGE_W13_DGRAD_BATCH_TASKS 32
#endif

#ifndef DG_EXPERIMENTAL_K3_RANGE_DW2_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_RANGE_DW2_BATCH_TASKS 256
#endif

#ifndef DG_EXPERIMENTAL_K3_RANGE_DW13_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_RANGE_DW13_BATCH_TASKS 256
#endif

#ifndef DG_EXPERIMENTAL_K3_INITIAL_BF16_DEQUANT_OVERLAP
#define DG_EXPERIMENTAL_K3_INITIAL_BF16_DEQUANT_OVERLAP 1
#endif

// Exact one-range K3 EP8 terminal BF16 wgrad specialization. It preserves
// MegaMoE's communication/dgrad phases and replaces only the post-readiness
// grouped-wgrad task assignment. The host emits both gates together for the
// measured non-accumulating first-chunk case.
#ifndef DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL
#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL 0
#endif
#ifndef DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL
#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL 0
#endif

// Compile-only integration gate for the allocation-free three-product
// group-32 MXFP8 weight-gradient suffix.  The production default remains the
// exact BF16 grouped-wgrad body until GPU numerics and latency validate this
// specialization.
#ifndef DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD
#define DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD 0
#endif
// One-way multi-range hybrid: preserve the proven ready-driven BF16 dW2
// body, but consume dW13 with the exact three-term MXFP8 UMMA/TMA body.  This
// gate is emitted only for the exact K3 EP=8 training specialization; keeping
// it independent from the all-MXFP8 suffix prevents accidental dW2 changes.
#ifndef DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
#define DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID 0
#endif
// Compile-only SiTU epilogue publication into the allocation-free symmetric
// dW13-A panel ring.  This remains independently disabled until the grouped
// consumer acquires and retires every ticket before direct dX reuses planes
// two and three.
#ifndef DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING
#define DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING 0
#endif
// Two-range K3 suffix that keeps both weight gradients on the BF16 UMMA/TMA
// engine. dW2 and dW13 each enter one persistent body; dW13 claims only
// experts whose final W13 dgrad reads have retired. dW2's disjoint background
// warps publish and fixed-order-reduce remote dX while its UMMA/TMA body runs;
// the two physical K segments remain one fixed-order FP32 reduction and reuse
// the existing union scheduler arena.
#ifndef DG_EXPERIMENTAL_K3_TWO_SEGMENT_BF16_PROGRESSIVE_WGRAD
#define DG_EXPERIMENTAL_K3_TWO_SEGMENT_BF16_PROGRESSIVE_WGRAD 0
#endif
// Once one exact dW13 expert is fully quantized, this many leading clusters
// remain on BF16 dW2 until its global cursor drains. Above this floor, suffix
// clusters transition progressively only when the unclaimed tail fits one
// scheduler batch per retained cluster. Handoff happens at mailbox batch
// boundaries, preserving a nonempty dW2 prefix and exact task ownership.
#ifndef DG_EXPERIMENTAL_K3_MXFP8_DW13_SHEPHERD_CLUSTERS
#define DG_EXPERIMENTAL_K3_MXFP8_DW13_SHEPHERD_CLUSTERS 32
#endif
#ifndef DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP
#define DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP 1
#endif
#ifndef DG_EXPERIMENTAL_K3_MXFP8_DW2_PRODUCER_CLUSTERS
#define DG_EXPERIMENTAL_K3_MXFP8_DW2_PRODUCER_CLUSTERS 0
#endif
#ifndef DG_EXPERIMENTAL_K3_MXFP8_PERSISTENT_DW2_PRODUCER_CLUSTERS
#define DG_EXPERIMENTAL_K3_MXFP8_PERSISTENT_DW2_PRODUCER_CLUSTERS 8
#endif
#ifndef DG_EXPERIMENTAL_K3_MXFP8_WGRAD_BATCH_TASKS
#define DG_EXPERIMENTAL_K3_MXFP8_WGRAD_BATCH_TASKS 32
#endif
#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD || \
    DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
#ifndef DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD
#define DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD 1
#endif
#endif

#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/impls/k3_multirange_backward.hpp>
#include <deep_gemm/impls/k3_mxfp8_exact_dw13b_ring.cuh>
#include <deep_gemm/impls/k3_mxfp8_exact_epilogue_pipeline.cuh>
#include <deep_gemm/impls/sm100_mxfp8_three_term_grouped_wgrad.cuh>
// Keep the fixed-shape BF16 wgrad engine out of the already register-heavy
// K3 parent wave.  The two calls are persistent (one dW2 and one dW13), so
// this trades two device-call boundaries for independent register allocation
// of the TMA/UMMA body and removes its scheduler state from the parent frame.
#define DG_BF16_GEMM_BODY_ATTRIBUTE __noinline__
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#undef DG_BF16_GEMM_BODY_ATTRIBUTE
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
    const uint32_t magnitude_code = nibble & 0x7u;
    const uint32_t scale_exponent = (scale_bits >> 23) & 0xffu;
    const uint32_t value_exponent =
        scale_exponent + (magnitude_code >> 1) - 1u;
    const uint32_t sign = (nibble & 0x8u) << 12;
    const uint32_t mantissa =
        static_cast<uint32_t>(
            magnitude_code > 1u && (magnitude_code & 1u)) << 6;
    const uint32_t nonzero_mask =
        0u - static_cast<uint32_t>(
            (magnitude_code != 0u) & (scale_exponent != 0u));
    return static_cast<uint16_t>(
        (sign | (value_exponent << 7) | mantissa) & nonzero_mask);
}

// Convert two E2M1 values and their power-of-two MX scales together. Packing
// the independent BF16 results into the two 16-bit lanes lets SM100 use its
// packed integer datapath for exponent adjustment and zero masking. This is
// bit-identical to two k3_mxfp4_bf16_bits calls for the finite UE8M0 values
// accepted by the native MXFP4 path.
CUTLASS_DEVICE uint32_t k3_mxfp4_bf16x2_bits(
    const uint8_t first_nibble,
    const uint8_t second_nibble,
    const uint32_t first_scale_bits,
    const uint32_t second_scale_bits) {
    const uint32_t nibbles =
        __byte_perm(first_nibble, second_nibble, 0x5410u);
    const uint32_t magnitude_codes = nibbles & 0x00070007u;
    // A binary32 exponent straddles bytes 2 and 3 (bits 23..30), so it cannot
    // be extracted with one byte permutation. Shift each scale independently
    // before packing the two exponent bytes into the halfword lanes.
    const uint32_t scale_exponents =
        ((first_scale_bits >> 23) & 0xffu) |
        (((second_scale_bits >> 23) & 0xffu) << 16);
    const uint32_t exponent_adjustments =
        (magnitude_codes >> 1) & 0x00030003u;
    const uint32_t value_exponents = __vsub2(
        __vadd2(scale_exponents, exponent_adjustments),
        0x00010001u);
    const uint32_t signs = (nibbles & 0x00080008u) << 12;
    const uint32_t odd_codes = magnitude_codes & 0x00010001u;
    const uint32_t upper_code_bits =
        (magnitude_codes >> 1) | (magnitude_codes >> 2);
    const uint32_t mantissas =
        (odd_codes & upper_code_bits) << 6;
    const uint32_t nonzero_mask =
        __vcmpne2(magnitude_codes, 0u) &
        __vcmpne2(scale_exponents, 0u);
    const uint32_t bf16_exponents =
        (value_exponents & 0x01ff01ffu) << 7;
    return (signs | bf16_exponents | mantissas) & nonzero_mask;
}

// Return exactly the exponent selected by
// ``(k3_mxfp4_bf16_bits(...) & 0x7fff) >> 7`` without constructing the BF16
// sign or mantissa. The first converter pass uses only this exponent to choose
// its UE8M0 scale; full BF16 construction remains in the quantization pass.
CUTLASS_DEVICE uint32_t k3_mxfp4_bf16_exponent(
    const uint8_t nibble, const uint32_t scale_bits) {
    const uint32_t magnitude_code = nibble & 0x7u;
    const uint32_t scale_exponent = (scale_bits >> 23) & 0xffu;
    const uint32_t value_exponent =
        scale_exponent + (magnitude_code >> 1) - 1u;
    const uint32_t nonzero_mask =
        0u - static_cast<uint32_t>(
            (magnitude_code != 0u) & (scale_exponent != 0u));
    return (value_exponent & 0xffu) & nonzero_mask;
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
// MXFP8 layout used by dgrad. The 16 independent (group, 32-row stripe) tasks
// are spread over the phase-specific non-MMA producer warps. A task intentionally uses
// the proven two-pass hardware-E4M3 path: pass one selects the UE8M0 scale;
// pass two rereads and quantizes without retaining a 32-value row in local
// memory. Disjoint byte stores retain the established shared-memory layout
// while cache publication is handled by TMA after all writers rendezvous.
template <uint32_t kProducerWarps>
CUTLASS_DEVICE void k3_mxfp4_to_mxfp8_transposed_tile(
    const int8_t* packed,
    const float* source_scales,
    uint8_t* output,
    uint32_t* output_scales,
    const uint32_t n_block_idx,
    const uint32_t producer_thread_idx) {
    constexpr uint32_t kTile = 128;
    constexpr uint32_t kGroup = 32;
    constexpr uint32_t kPackedStride = kTile / 2;
    constexpr uint32_t kScaleStride = kTile / kGroup;
    static_assert(kProducerWarps > 0, "At least one converter warp is required");
    const uint32_t producer_warp = producer_thread_idx / 32;
    const uint32_t lane = producer_thread_idx % 32;
    if (producer_warp >= kProducerWarps)
        return;

    // Rotating by four keeps the heaviest slot off producer zero. For the
    // seven-warp W13 path this preserves the established 2/3-task split; for
    // the three-warp W2 path it gives producer counts 5, 5, and 6.
    const uint32_t task_slot = (producer_warp + 4) % kProducerWarps;
    #pragma unroll 1
    for (uint32_t task = task_slot;
         task < 4 * kScaleStride;
         task += kProducerWarps) {
        const uint32_t group = task / kScaleStride;
        const uint32_t stripe = task % kScaleStride;
        const uint32_t local_n = stripe * 32 + lane;
        const uint32_t global_n = n_block_idx * kTile + local_n;
        const uint32_t packed_col = local_n / 2;
        const uint32_t nibble_shift = (global_n & 1u) * 4;
        const uint32_t row = local_n & 7u;
        const uint32_t lane_scale_bits =
            *reinterpret_cast<const uint32_t*>(
                &source_scales[
                    (group * kGroup + lane) * kScaleStride + stripe]);
        uint32_t amax_exponent = 0;
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
            const uint32_t first_exponent = k3_mxfp4_bf16_exponent(
                (packed_first >> nibble_shift) & 0xf,
                first_scale_bits);
            const uint32_t second_exponent = k3_mxfp4_bf16_exponent(
                (packed_second >> nibble_shift) & 0xf,
                second_scale_bits);
            amax_exponent = cute::max(
                amax_exponent, first_exponent);
            amax_exponent = cute::max(
                amax_exponent, second_exponent);
        }

        const uint32_t scale_byte = cute::max(
            88,
            static_cast<int>(amax_exponent) - 8);
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
        }
        const uint32_t local_sf_n =
            (local_n & 31u) * 4 + ((local_n >> 5) & 3u);
        reinterpret_cast<uint8_t*>(
            output_scales + local_sf_n)[group] =
            static_cast<uint8_t>(scale_byte);
    }
}

// One-pass specialization of the converter above. Each lane keeps
// the 16 packed BF16 pairs for one (group, 32-row stripe) task live until its
// UE8M0 scale is known, then emits the same eight E4M3 words without rereading
// packed MXFP4 values or repeating the source-scale shuffles. The producer
// count is phase-specific: K3 uses three W2 producers and seven W13 producers.
// Both instantiations preserve the hardware E4M3 conversion and exact output
// byte order of the two-pass implementation.
template <uint32_t kProducerWarps>
CUTLASS_DEVICE void k3_mxfp4_to_mxfp8_transposed_tile_one_pass(
    const int8_t* packed,
    const float* source_scales,
    uint8_t* output,
    uint32_t* output_scales,
    const uint32_t n_block_idx,
    const uint32_t producer_thread_idx) {
    constexpr uint32_t kTile = 128;
    constexpr uint32_t kGroup = 32;
    constexpr uint32_t kPackedStride = kTile / 2;
    constexpr uint32_t kScaleStride = kTile / kGroup;
    static_assert(kProducerWarps > 0, "At least one converter warp is required");
    const uint32_t producer_warp = producer_thread_idx / 32;
    const uint32_t lane = producer_thread_idx % 32;
    if (producer_warp >= kProducerWarps)
        return;

    const uint32_t task_slot =
        (producer_warp + 4) % kProducerWarps;
    #pragma unroll 1
    for (uint32_t task = task_slot;
         task < 4 * kScaleStride;
         task += kProducerWarps) {
        const uint32_t group = task / kScaleStride;
        const uint32_t stripe = task % kScaleStride;
        const uint32_t local_n = stripe * 32 + lane;
        const uint32_t global_n = n_block_idx * kTile + local_n;
        const uint32_t packed_col = local_n / 2;
        const uint32_t nibble_shift = (global_n & 1u) * 4;
        const uint32_t row = local_n & 7u;
        const uint32_t lane_scale_bits =
            *reinterpret_cast<const uint32_t*>(
                &source_scales[
                    (group * kGroup + lane) * kScaleStride + stripe]);
        uint32_t retained_pair_bits[16];
        uint32_t amax_exponent = 0;

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
            const uint32_t retained_bits = k3_mxfp4_bf16x2_bits(
                (packed_first >> nibble_shift) & 0xf,
                (packed_second >> nibble_shift) & 0xf,
                first_scale_bits,
                second_scale_bits);
            retained_pair_bits[pair] = retained_bits;
            const uint16_t first = static_cast<uint16_t>(retained_bits);
            const uint16_t second = static_cast<uint16_t>(retained_bits >> 16);
            amax_exponent = cute::max(
                amax_exponent,
                static_cast<uint32_t>((first & 0x7fffu) >> 7));
            amax_exponent = cute::max(
                amax_exponent,
                static_cast<uint32_t>((second & 0x7fffu) >> 7));
        }

        const uint32_t scale_byte = cute::max(
            88, static_cast<int>(amax_exponent) - 8);
        #pragma unroll
        for (uint32_t word = 0; word < 8; ++word) {
            const uint32_t pair01 = retained_pair_bits[word * 2];
            const uint32_t pair23 = retained_pair_bits[word * 2 + 1];
            const uint16_t word_values[4] = {
                static_cast<uint16_t>(pair01),
                static_cast<uint16_t>(pair01 >> 16),
                static_cast<uint16_t>(pair23),
                static_cast<uint16_t>(pair23 >> 16),
            };
            const uint32_t quantized = k3_quantize_bf16x4_e4m3(
                word_values, scale_byte);
            const uint32_t logical_k_byte =
                group * kGroup + word * 4;
            const uint32_t byte_offset =
                (local_n >> 3) * 8 * kTile +
                row * kTile +
                ((logical_k_byte >> 4) ^ row) * 16 +
                (logical_k_byte & 15);
            *reinterpret_cast<uint32_t*>(output + byte_offset) =
                quantized;
        }
        const uint32_t local_sf_n =
            (local_n & 31u) * 4 + ((local_n >> 5) & 3u);
        reinterpret_cast<uint8_t*>(
            output_scales + local_sf_n)[group] =
            static_cast<uint8_t>(scale_byte);
    }
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

#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD || \
    DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
/** Drain one dynamic exact-dW2 queue while W13 producer clusters join late.
 *
 * This is a device helper, not a separate CUDA launch: UMMA/TMA, scheduler
 * mailboxes, and the symmetric-memory publication barrier remain inside the
 * fused backward kernel.  Outlining only protects the 1024-thread parent's
 * register bound.  The caller owns an already-empty base-zero TMEM allocation;
 * this helper initializes/releases exact barriers without allocating or
 * freeing TMEM.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM, uint32_t kNumSMs,
          uint32_t kNumRanks, uint32_t kNumThreads,
          bool kAccumulateWgrad, uint32_t kBatchTasks,
          bool kFeatureReady>
CUTLASS_DEVICE __noinline__ void k3_mxfp8_run_dynamic_dw2_overlap(
        uint32_t* state,
        const uint32_t launch_epoch,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        const cute::TmaDescriptor& tensor_map_d,
        const layout::SymBuffer<kNumRanks>* backward_sym_buffer,
        const layout::Workspace* backward_workspace,
        uint8_t* smem_buffer) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    using Config = Sm100K3MxFp8ThreeTermWgradConfig<
        kHidden, kIntermediateHidden,
        kNumExperts, kNumSMs, kAccumulateWgrad>;
    using Provider = sched::ExternalKGroupedDynamicRangeProvider<
        Config::kBlockM, Config::kBlockN,
        Config::kNumMulticast, Config::kIsMulticastOnA,
        kNumSMs, kHidden, kIntermediateHidden,
        192u, Config::kScaleKSpan,
        Overlap::kPoolBlockPrefix, Overlap::kActiveExperts,
        4u, 4u, 16u,
        false, 0u,
        Prefix::kValuePrefix, Prefix::kScalePrefix, true,
        kFeatureReady ? 0u : 1u, Overlap::kPoolBlockPrefix,
        Config::kKAlignment, Config::kBlockK, true,
        kFeatureReady, Overlap::kDW2FeatureReadyMasks>;
    using RetainParentTmem = Sm100K3MxFp8WgradBatchResourceHooks<
        true, true, true, false, false>;
    static_assert(
        kNumThreads == 1024u &&
            Provider::kTaskReadyFirstTaskClaim &&
            Provider::kTaskRetirementBias ==
                (kFeatureReady ? 0u : 1u) &&
            Provider::kTaskFeatureReadyFirstTaskClaim ==
                kFeatureReady &&
            Provider::kTaskFeatureReadyWord ==
                Overlap::kDW2FeatureReadyMasks &&
            Provider::kCompleteAcquireMask ==
                Overlap::kExactSchedulerRoleMask &&
            Provider::kNumClusterTasksPerGroup ==
                Overlap::kDW2ClusterTasksPerExpert,
        "Exact dW2 dynamic scheduler contract changed");
    static_assert(
        RetainParentTmem::kInitializeBatchResources &&
            RetainParentTmem::kReleaseBatchResources &&
            !RetainParentTmem::kAllocateTmem &&
            !RetainParentTmem::kFreeTmem,
        "Exact dW2 must retain the parent TMEM allocation");

    const auto* const maps =
        tensor_map_pack->maps + kK3MxFp8DW2ValueAPrimaryMap;
    if (threadIdx.x == 0u) {
        while (ptx::ld_acq(state + Overlap::kEpoch) != launch_epoch)
            __nanosleep(64);
    }
    __syncthreads();

    // Open the disjoint dW13 generation before dW2 starts.  Every cluster that
    // consumes its terminal dW2 mailbox token may then enter the ready-first
    // dW13 queue and continue unclaimed quantization tasks there. Per-expert
    // composite readiness, not this generation word, protects every aliased
    // value and scale plane.
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        const uint32_t dw13_epoch = launch_epoch | 0x80000000u;
        asm volatile(
            "st.release.gpu.global.u32 [%0], %1;"
            :: "l"(state + Overlap::kDW13Epoch), "r"(dw13_epoch)
            : "memory");
    }

    auto* const exact_tmem_ptr = reinterpret_cast<uint32_t*>(
        smem_buffer +
        sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset<Config>());
    if (threadIdx.x == 0u)
        *exact_tmem_ptr = 0u;
    __syncthreads();

    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    auto* const mailbox = state + Overlap::kDW2Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    const sched::ExternalKGroupedRangeStream stream{
        state,
        0u, 0u,
        state + Overlap::kDW2Cursor,
        0u,
        mailbox,
        kBatchTasks,
        Overlap::kDW2ClusterTasksPerExpert,
        state + Overlap::kDW2OperandReady,
        0u,
        state + Overlap::kEpoch,
        launch_epoch,
        state + Overlap::kDW2Tasks,
    };

    constexpr uint32_t kPublishFirstWarp = 8u;
    constexpr uint32_t kPublishWarps = 2u;
    constexpr uint32_t kPublishThreads = kPublishWarps * 32u;
    const auto* const backward_ranges_ptr = &backward_ranges;
    const auto background_work = [=] (
            uint32_t background_warp_idx,
            uint32_t background_lane_idx) {
        if (background_warp_idx >= kPublishFirstWarp &&
            background_warp_idx < kPublishFirstWarp + kPublishWarps) {
            constexpr uint32_t kGridSyncIndex = 2u;
            constexpr uint32_t kBarrierTag = 7u;
            constexpr uint32_t kNamedBarrier = 15u;
            const uint32_t thread_idx =
                (background_warp_idx - kPublishFirstWarp) * 32u +
                background_lane_idx;
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kPublishThreads,
                kGridSyncIndex, kBarrierTag>(
                    *backward_workspace, *backward_sym_buffer,
                    blockIdx.x, thread_idx,
                    [=]() {
                        ptx::sync_aligned(
                            kPublishThreads, kNamedBarrier);
                    },
                    true, true);
        }
        detail::k3_mxfp8_stream_dw13_operands_background<
            kHidden, kIntermediateHidden, kNumExperts, kBlockM,
            kNumSMs, kNumThreads, true>(
                expert_counts, backward_ranges_ptr, k_capacity,
                scale_arena_source, tensor_map_pack, state,
                smem_buffer, background_warp_idx,
                background_lane_idx);
    };
    const auto input_tile_retired = [=] (
            uint32_t expert, uint32_t, uint32_t) {
        const uint32_t previous = ptx::atomic_add_acq_rel(
            state + Overlap::kDW2InputRetired + expert, 1u);
        DG_DEVICE_ASSERT(
            previous < Overlap::kDW2ClusterTasksPerExpert);
    };
    sm100_k3_mxfp8_three_term_grouped_wgrad_body<
        Config, Provider, RetainParentTmem>(
            reinterpret_cast<int*>(
                const_cast<sched::ExternalKGroupedRangeStream*>(&stream)),
            state[Prefix::kValuePrefix + kNumExperts],
            maps[0], maps[1], maps[2], maps[3],
            maps[4], maps[5], maps[6], maps[7],
            tensor_map_d, smem_buffer, false,
            background_work, input_tile_retired);

}

/** Execute dW2 and dW13 as one phase-tagged ready-first exact engine.
 *
 * Mailbox word one packs the phase in bit 31 and a phase-local logical task in
 * bits 0..30.  Its single publisher also owns words two/three; scheduler roles
 * only acquire those payload words and release-OR acknowledgements into word
 * zero.  The sticky dW2 producer has retired its temporary word-one ticket
 * before this function is entered, and background quantization only reads the
 * final zero-count token, so the four-word ownership is unambiguous.
 *
 * dW2 panel publication makes its first UMMA quantum claimable.  Each retired
 * dW2 output task increments exactly one A-pair and one B-pair counter; those
 * counters unlock only the aliased dW13 feature panels they protect.  The
 * background engines quantize a complete unlocked panel, release-publish its
 * scale-row count, and the same scheduler immediately prioritizes ready dW13
 * work.  One barrier/TMEM lifetime therefore covers both descriptor phases.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads, bool kAccumulateWgrad,
          uint32_t kBatchTasks, bool kClearEmptyOutputs,
          bool kStreamDW2SuffixPanels = false>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_run_dynamic_unified_wgrad_overlap() {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    using TensorMapPack =
        K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>;
    using AuxSlot = K3MxFp8WgradAuxSlot<
        cute::TmaDescriptor, cutlass::bfloat16_t,
        cutlass::float_e4m3_t>;
    using Handoff = K3MxFp8WgradSuffixHandoff<
        TensorMapPack, layout::SymBuffer<kNumRanks>, layout::Workspace,
        AuxSlot>;
    static_assert(k3_mxfp8_wgrad_suffix_handoff_abi<
        TensorMapPack, layout::SymBuffer<kNumRanks>, layout::Workspace,
        AuxSlot>());
    constexpr uint32_t kHandoffOffset =
        math::constexpr_align(kK3MxFp8DW13QuantBodySmemBytes, 8u);
    static_assert(kHandoffOffset == 153736u);
    static_assert(
        kHandoffOffset + sizeof(Handoff) <=
        kK3MxFp8DW13QuantScratchBegin);
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const volatile auto* const handoff =
        reinterpret_cast<const volatile Handoff*>(
            smem_buffer + kHandoffOffset);
    const auto* const tensor_map_pack = handoff->tensor_map_pack;
    const auto* const backward_sym_buffer =
        handoff->backward_sym_buffer;
    const auto* const backward_workspace =
        handoff->backward_workspace;
    const auto* const dw2_output_slot = handoff->dw2_output_slot;
    const auto* const dw13_output_slot = handoff->dw13_output_slot;
    const auto* const ranges_slot = handoff->ranges_slot;
    const auto* const args_slot = handoff->args_slot;
    const uint32_t launch_epoch = handoff->expected_launch_epoch;
    const auto* const suffix_context = &args_slot->exact_args;
    const auto& backward_ranges = ranges_slot->exact_ranges;
    const auto* const expert_counts = suffix_context->expert_counts;
    auto* const state = suffix_context->state;
    // CTA zero publishes the scheduler epoch after metadata preparation's
    // grid edge. Other CTAs can enter this suffix before that release store,
    // so compare against the parent CTA's expected epoch instead of accepting
    // whichever generation happens to be visible on the first acquire load.
    if (threadIdx.x == 0u) {
        while (ptx::ld_acq(state + Overlap::kEpoch) != launch_epoch)
            __nanosleep(64);
    }
    __syncthreads();
    const uint32_t k_capacity = suffix_context->k_capacity;
    auto* const grad_ye_output = suffix_context->grad_ye_output;
    auto* const h_weighted_output = suffix_context->h_weighted_output;
    auto* const x_pool_output = suffix_context->x_pool_output;
    auto* const grad_y_unweighted_output =
        suffix_context->grad_y_unweighted_output;
    const auto* const down_unweighted_output =
        suffix_context->down_unweighted_output;
    const auto* const scale_arena_source =
        suffix_context->scale_arena_source;
    const auto& dw2_output_map = dw2_output_slot->exact_output_map;
    const auto& dw13_output_map = dw13_output_slot->exact_output_map;
    auto* const w2_output = suffix_context->w2_output;
    auto* const w13_output = suffix_context->w13_output;
    const bool clear_empty_outputs =
        suffix_context->clear_empty_outputs != 0u;
    auto* const backward_grad_x_output =
        suffix_context->backward_grad_x_output;
    const auto* const backward_grad_y = suffix_context->backward_grad_y;
    const uint32_t num_backward_tokens =
        suffix_context->num_backward_tokens;
    const uint32_t first_range_tokens =
        suffix_context->first_range_tokens;
    const uint32_t second_range_begin =
        suffix_context->second_range_begin;
    const uint32_t num_max_tokens_per_rank =
        backward_workspace->num_max_tokens_per_rank;
    const uint32_t num_topk = suffix_context->num_topk;
    using Config = Sm100K3MxFp8ThreeTermWgradConfig<
        kHidden, kIntermediateHidden,
        kNumExperts, kNumSMs, kAccumulateWgrad>;
    using Provider = sched::ExternalKGroupedDynamicRangeProvider<
        Config::kBlockM, Config::kBlockN,
        Config::kNumMulticast, Config::kIsMulticastOnA,
        kNumSMs, kHidden, kIntermediateHidden,
        192u, Config::kScaleKSpan,
        Overlap::kPoolBlockPrefix, Overlap::kActiveExperts,
        4u, 4u, 16u,
        false, 0u,
        Prefix::kValuePrefix, Prefix::kScalePrefix, true,
        0u, Overlap::kPoolBlockPrefix,
        Config::kKAlignment, Config::kBlockK, true,
        true, Overlap::kDW2FeatureReadyMasks,
        true, Overlap::kDW13CompositeReady,
        Overlap::kDW13FeatureDone, Overlap::kDW13QuantDone>;
    using UnifiedResources =
        Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks;
    static_assert(
        kNumThreads == 1024u && Provider::kTaskPhaseTagged &&
            Provider::kTaskFeatureReadyFirstTaskClaim &&
            Provider::kTaskRetirementBias == 0u &&
            Provider::kNumClusterTasksPerGroup ==
                Overlap::kDW2ClusterTasksPerExpert &&
            Provider::kK3DW13ClusterTasksPerGroup ==
                Overlap::kDW13ClusterTasksPerExpert &&
            Provider::kCompleteAcquireMask ==
                Overlap::kExactSchedulerRoleMask,
        "Unified exact-wgrad provider contract changed");
    static_assert(
        UnifiedResources::kInitializeBatchResources &&
            UnifiedResources::kReleaseBatchResources &&
            !UnifiedResources::kAllocateTmem &&
            UnifiedResources::kFreeTmem,
        "Unified exact wgrad must reuse then release parent TMEM");

    if constexpr (kClearEmptyOutputs) {
        if (clear_empty_outputs) {
            constexpr uint64_t kBF16PerVector =
                sizeof(uint4) / sizeof(cutlass::bfloat16_t);
            constexpr uint64_t kW2VectorsPerExpert =
                static_cast<uint64_t>(kHidden) * kIntermediateHidden /
                kBF16PerVector;
            constexpr uint64_t kW13VectorsPerExpert =
                static_cast<uint64_t>(2u * kIntermediateHidden) * kHidden /
                kBF16PerVector;
            constexpr uint64_t kVectorsPerExpert =
                kW2VectorsPerExpert + kW13VectorsPerExpert;
            const uint64_t global_thread =
                static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                threadIdx.x;
            constexpr uint64_t kGlobalThreads =
                static_cast<uint64_t>(kNumSMs) * kNumThreads;
            auto* const w2_vectors = reinterpret_cast<uint4*>(w2_output);
            auto* const w13_vectors = reinterpret_cast<uint4*>(w13_output);
            const uint4 zero = {0u, 0u, 0u, 0u};
            #pragma unroll 1
            for (uint32_t expert = 0u;
                 expert < kNumExperts; ++expert) {
                bool active = false;
                #pragma unroll
                for (uint32_t range_idx = 0u;
                     range_idx < kK3MaxBackwardRanges; ++range_idx) {
                    if (range_idx >= backward_ranges.num_ranges)
                        break;
                    active |= __ldg(
                        expert_counts +
                            backward_ranges.expert_counts_begin(
                                range_idx, kNumExperts) +
                            expert) !=
                        0;
                }
                if (active)
                    continue;
                for (uint64_t linear = global_thread;
                     linear < kVectorsPerExpert;
                     linear += kGlobalThreads) {
                    if (linear < kW2VectorsPerExpert) {
                        w2_vectors[
                            static_cast<uint64_t>(expert) *
                                kW2VectorsPerExpert + linear] = zero;
                    } else {
                        w13_vectors[
                            static_cast<uint64_t>(expert) *
                                kW13VectorsPerExpert +
                            linear - kW2VectorsPerExpert] = zero;
                    }
                }
            }
        }
    }

    if (threadIdx.x == 0u) {
        while (ptx::ld_acq(state + Overlap::kEpoch) != launch_epoch)
            __nanosleep(64);
    }
    __syncthreads();

    auto* const exact_tmem_ptr = reinterpret_cast<uint32_t*>(
        smem_buffer +
        sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset<Config>());
    if (threadIdx.x == 0u)
        *exact_tmem_ptr = 0u;
    __syncthreads();

    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    auto* const mailbox = state + Overlap::kDW2Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    const sched::ExternalKGroupedRangeStream stream{
        state,
        0u, 0u,
        state + Overlap::kDW2Cursor,
        0u,
        mailbox,
        kBatchTasks,
        Overlap::kDW2ClusterTasksPerExpert,
        state + Overlap::kDW2OperandReady,
        14u,
        state + Overlap::kEpoch,
        launch_epoch,
        state + Overlap::kDW2Tasks,
    };

    constexpr uint32_t kReduceFirstWarp = 8u;
    constexpr uint32_t kReduceWarps = 4u;
    constexpr uint32_t kQuantFirstWarp = 12u;
    constexpr uint32_t kQuantWarps =
        kK3MxFp8DW13QuantNumEngines *
        kK3MxFp8DW13QuantWarpsPerEngine;
    const auto* const backward_ranges_ptr = &backward_ranges;
    const auto background_work = [=] (
            uint32_t background_warp_idx,
            uint32_t background_lane_idx) {
        if (background_warp_idx >= kReduceFirstWarp &&
            background_warp_idx < kReduceFirstWarp + kReduceWarps) {
            constexpr uint32_t kGridSyncIndex = 2u;
            constexpr uint32_t kBarrierTag = 7u;
            constexpr uint32_t kNamedBarrier = 15u;
            constexpr uint32_t kPublishThreads = kReduceWarps * 32u;
            const uint32_t thread_idx =
                (background_warp_idx - kReduceFirstWarp) * 32u +
                background_lane_idx;
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kPublishThreads,
                kGridSyncIndex, kBarrierTag>(
                    *backward_workspace, *backward_sym_buffer,
                    blockIdx.x, thread_idx,
                    [=]() {
                        ptx::sync_aligned(
                            kPublishThreads, kNamedBarrier);
                    },
                    true, true);
            k3_mxfp8_wgrad_fixed_topk_combine<
                kNumSMs, kReduceWarps>(
                    backward_grad_x_output,
                    backward_grad_y,
                    backward_ranges_ptr,
                    num_backward_tokens,
                    num_max_tokens_per_rank,
                    num_topk, kHidden,
                    background_warp_idx - kReduceFirstWarp,
                    background_lane_idx);
        }
        if (background_warp_idx >= kQuantFirstWarp &&
            background_warp_idx < kQuantFirstWarp + kQuantWarps) {
            if constexpr (kStreamDW2SuffixPanels) {
                // The two-range exact engine owns dW2 preparation here, after
                // the immutable union metadata and all BF16 padding writes are
                // published.  Complete 128-feature panels release directly to
                // the feature-ready scheduler; dW13 then reuses the same four
                // engine scratch slices after dW2 invalidates its barriers.
                detail::k3_mxfp8_stream_dw2_operands_background<
                    kHidden, kIntermediateHidden, kNumExperts, kBlockM,
                    kNumSMs, kNumThreads>(
                        expert_counts, backward_ranges_ptr, k_capacity,
                        grad_ye_output, h_weighted_output,
                        x_pool_output, grad_y_unweighted_output,
                        down_unweighted_output, grad_ye_output,
                        scale_arena_source, state, smem_buffer,
                        background_warp_idx, background_lane_idx);
            }
            detail::k3_mxfp8_stream_dw13_operands_background<
                kHidden, kIntermediateHidden, kNumExperts, kBlockM,
                kNumSMs, kNumThreads, false>(
                    expert_counts, backward_ranges_ptr, k_capacity,
                    scale_arena_source, tensor_map_pack, state,
                    smem_buffer, background_warp_idx,
                    background_lane_idx);
        }
    };
    const auto input_tile_retired = [=] (
            uint32_t expert, uint32_t m_block,
            uint32_t n_block, uint32_t wgrad_phase) {
        if (wgrad_phase != 0u)
            return;
        const uint32_t previous = ptx::atomic_add_acq_rel(
            state + Overlap::kDW2InputRetired + expert, 1u);
        DG_DEVICE_ASSERT(
            previous < Overlap::kDW2ClusterTasksPerExpert);
        auto* const pair_counters =
            state + Overlap::kDW2InputPairRetired +
            expert * Overlap::kDW2InputPairCountersPerExpert;
        DG_DEVICE_ASSERT(m_block < kHidden / 256u);
        const uint32_t a_previous = ptx::atomic_add_acq_rel(
            pair_counters + m_block, 1u);
        DG_DEVICE_ASSERT(a_previous < kIntermediateHidden / 256u);
        DG_DEVICE_ASSERT(n_block % 2u == 0u);
        const uint32_t b_pair = n_block / 2u;
        DG_DEVICE_ASSERT(b_pair < kIntermediateHidden / 256u);
        const uint32_t b_previous = ptx::atomic_add_acq_rel(
            pair_counters + kHidden / 256u + b_pair, 1u);
        DG_DEVICE_ASSERT(b_previous < kHidden / 256u);
    };

    const auto* const dw2_maps =
        tensor_map_pack->maps + kK3MxFp8DW2ValueAPrimaryMap;
    const auto* const dw13_maps =
        tensor_map_pack->maps + kK3MxFp8DW13ValueAPrimaryMap;
    sm100_k3_mxfp8_three_term_grouped_wgrad_body<
        Config, Provider, UnifiedResources>(
            reinterpret_cast<int*>(
                const_cast<sched::ExternalKGroupedRangeStream*>(&stream)),
            state[Prefix::kValuePrefix + kNumExperts],
            dw2_maps[0], dw2_maps[1], dw2_maps[2], dw2_maps[3],
            dw2_maps[4], dw2_maps[5], dw2_maps[6], dw2_maps[7],
            dw2_output_map, smem_buffer, false,
            background_work, input_tile_retired,
            dw13_maps, &dw13_output_map);
}

/** Drain the readiness-driven exact dW13 queue and release parent TMEM.
 *
 * The producer subset publishes a distinct dW13 generation after all aliased
 * dW2 inputs retire. Every CTA acquire-waits that generation before entering
 * the grouped body with immutable grid-constant descriptors. Per-expert
 * scheduling then waits for `14 * pool_blocks + 1` composite
 * credits: fourteen W13-dgrad cluster retirements per pool block plus the
 * producer's operand-ready credit. Warps 8--11 optionally publish remote dX
 * before performing the fixed-top-k reduction. Warps 12--15 of every
 * transitioned cluster keep producing exact group-32 operands while the same
 * CTAs' main warps drain dW13 UMMA/TMA. The full grid reuses the existing
 * CTA-local scratch and global claim/completion words; it adds no allocation
 * or saved activation.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads, bool kAccumulateWgrad,
          uint32_t kBatchTasks, bool kClearEmptyOutputs,
          bool kReuseParentTmem = true,
          bool kRunFixedTopKCombine = true,
          bool kPublishBeforeCombine = false,
          bool kUseExactEpilogueRing = false>
CUTLASS_DEVICE __noinline__ void k3_mxfp8_run_dynamic_dw13_overlap(
        uint32_t* state,
        const uint32_t expected_epoch,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        const cute::TmaDescriptor& tensor_map_d,
        const layout::SymBuffer<kNumRanks>* backward_sym_buffer,
        const layout::Workspace* backward_workspace,
        uint8_t* smem_buffer,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        cutlass::bfloat16_t* w2_output,
        cutlass::bfloat16_t* w13_output,
        const bool clear_empty_outputs,
        cutlass::bfloat16_t* backward_grad_x_output,
        const cutlass::bfloat16_t* backward_grad_y,
        const uint32_t num_backward_tokens,
        const uint32_t first_range_tokens,
        const uint32_t second_range_begin,
        const uint32_t num_max_tokens_per_rank,
        const uint32_t num_topk) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    using Config = Sm100K3MxFp8ThreeTermWgradConfig<
        2u * kIntermediateHidden, kHidden,
        kNumExperts, kNumSMs, kAccumulateWgrad>;
    using Provider = sched::ExternalKGroupedDynamicRangeProvider<
        Config::kBlockM, Config::kBlockN,
        Config::kNumMulticast, Config::kIsMulticastOnA,
        kNumSMs, 2u * kIntermediateHidden, kHidden,
        192u, Config::kScaleKSpan,
        Overlap::kPoolBlockPrefix, Overlap::kActiveExperts,
        4u, 4u, 16u,
        false, 0u,
        Prefix::kValuePrefix, Prefix::kScalePrefix, true,
        1u, Overlap::kPoolBlockPrefix,
        Config::kKAlignment, Config::kBlockK, true>;
    using TerminalResources = cute::conditional_t<
        kUseExactEpilogueRing,
        Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks,
        cute::conditional_t<
            kReuseParentTmem,
            Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks,
            Sm100K3MxFp8WgradDefaultBatchResourceHooks>>;
    static_assert(
        kNumThreads == 1024u &&
            Provider::kCompleteAcquireMask ==
                Overlap::kExactSchedulerRoleMask &&
            Provider::kNumClusterTasksPerGroup ==
                Overlap::kDW13ClusterTasksPerExpert &&
            Provider::kTaskRetirementBias == 1u &&
            Provider::kTaskReadyFirstTaskClaim,
        "Exact dW13 dynamic scheduler contract changed");
    static_assert(
        TerminalResources::kInitializeBatchResources &&
            TerminalResources::kReleaseBatchResources &&
            (kUseExactEpilogueRing
                 ? (!TerminalResources::kAllocateTmem &&
                    !TerminalResources::kFreeTmem)
                 : (TerminalResources::kFreeTmem &&
                    (kReuseParentTmem
                         ? !TerminalResources::kAllocateTmem
                         : TerminalResources::kAllocateTmem))),
        "Terminal exact dW13 must own a complete, explicit TMEM lifetime");
    static_assert(
        !kPublishBeforeCombine ||
            (kRunFixedTopKCombine && kNumRanks > 1u),
        "Remote dX publication must feed the terminal fixed-top-k combine");
    if constexpr (kClearEmptyOutputs) {
        if (clear_empty_outputs) {
            constexpr uint64_t kBF16PerVector =
                sizeof(uint4) / sizeof(cutlass::bfloat16_t);
            constexpr uint64_t kW2VectorsPerExpert =
                static_cast<uint64_t>(kHidden) * kIntermediateHidden /
                kBF16PerVector;
            constexpr uint64_t kW13VectorsPerExpert =
                static_cast<uint64_t>(2u * kIntermediateHidden) * kHidden /
                kBF16PerVector;
            constexpr uint64_t kVectorsPerExpert =
                kW2VectorsPerExpert + kW13VectorsPerExpert;
            const uint64_t global_thread =
                static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                threadIdx.x;
            constexpr uint64_t kGlobalThreads =
                static_cast<uint64_t>(kNumSMs) * kNumThreads;
            auto* const w2_vectors = reinterpret_cast<uint4*>(w2_output);
            auto* const w13_vectors = reinterpret_cast<uint4*>(w13_output);
            const uint4 zero = {0u, 0u, 0u, 0u};
            #pragma unroll 1
            for (uint32_t expert = 0u;
                 expert < kNumExperts; ++expert) {
                bool active_in_union = false;
                #pragma unroll 1
                for (uint32_t range_idx = 0u;
                     range_idx < backward_ranges.num_ranges;
                     ++range_idx) {
                    active_in_union |= __ldg(
                        expert_counts +
                        backward_ranges.expert_counts_begin(
                            range_idx, kNumExperts) + expert) != 0;
                }
                if (active_in_union)
                    continue;
                for (uint64_t linear = global_thread;
                     linear < kVectorsPerExpert;
                     linear += kGlobalThreads) {
                    if (linear < kW2VectorsPerExpert) {
                        w2_vectors[
                            static_cast<uint64_t>(expert) *
                                kW2VectorsPerExpert + linear] = zero;
                    } else {
                        w13_vectors[
                            static_cast<uint64_t>(expert) *
                                kW13VectorsPerExpert +
                            linear - kW2VectorsPerExpert] = zero;
                    }
                }
            }
        }
    }

    const auto* const maps =
        tensor_map_pack->maps + kK3MxFp8DW13ValueAPrimaryMap;
    const auto* const ring_maps =
        tensor_map_pack->maps + kK3MxFp8DW13RingValueAPrimaryMap;
    if (threadIdx.x == 0u) {
        while (ptx::ld_acq(state + Overlap::kDW13Epoch) !=
               expected_epoch) {
            __nanosleep(64);
        }
    }
    __syncthreads();

    auto* const exact_tmem_ptr = reinterpret_cast<uint32_t*>(
        smem_buffer +
        sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset<Config>());
    if (threadIdx.x == 0u)
        *exact_tmem_ptr = 0u;
    __syncthreads();

    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    auto* const mailbox = state + Overlap::kDW13Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    const sched::ExternalKGroupedRangeStream stream{
        state,
        0u, 0u,
        state + Overlap::kDW13Cursor,
        0u,
        mailbox,
        kBatchTasks,
        Overlap::kDW13ClusterTasksPerExpert,
        kUseExactEpilogueRing
            ? nullptr : state + Overlap::kDW13CompositeReady,
        kUseExactEpilogueRing ? 0u : 14u,
        state + Overlap::kDW13Epoch,
        expected_epoch,
        state + Overlap::kDW13Tasks,
    };

    constexpr uint32_t kReduceFirstWarp = 8u;
    constexpr uint32_t kReduceWarps = 4u;
    constexpr uint32_t kQuantFirstWarp = 12u;
    constexpr uint32_t kQuantWarps =
        kK3MxFp8DW13QuantNumEngines *
        kK3MxFp8DW13QuantWarpsPerEngine;
    const auto* const backward_ranges_ptr = &backward_ranges;
    const auto background_work = [=] (
            uint32_t background_warp_idx,
            uint32_t background_lane_idx) {
        if constexpr (kRunFixedTopKCombine) {
          if (background_warp_idx >= kReduceFirstWarp &&
              background_warp_idx < kReduceFirstWarp + kReduceWarps) {
            if constexpr (kPublishBeforeCombine) {
              // Two-range BF16 dW2 deliberately omits its blocking rank
              // rendezvous. Run the identical publication protocol here on
              // dW13's otherwise-idle reduction warps so it overlaps exact
              // UMMA/TMA. The subsequent fixed-top-k reduction is ordered
              // after both the local-grid and cross-rank completion edges.
              if (backward_ranges_ptr->num_ranges != 3u) {
                constexpr uint32_t kPublishThreads =
                    kReduceWarps * 32u;
                constexpr uint32_t kGridSyncIndex = 2u;
                constexpr uint32_t kBarrierTag = 7u;
                constexpr uint32_t kNamedBarrier = 15u;
                const uint32_t publish_thread_idx =
                    (background_warp_idx - kReduceFirstWarp) * 32u +
                    background_lane_idx;
                comm::nvlink_barrier<
                    kNumRanks, kNumSMs, kPublishThreads,
                    kGridSyncIndex, kBarrierTag>(
                        *backward_workspace, *backward_sym_buffer,
                        blockIdx.x, publish_thread_idx,
                        [=]() {
                            ptx::sync_aligned(
                                kPublishThreads, kNamedBarrier);
                        },
                        true, true);
              }
            }
            k3_mxfp8_wgrad_fixed_topk_combine<
                kNumSMs, kReduceWarps>(
                    backward_grad_x_output,
                    backward_grad_y,
                    backward_ranges_ptr,
                    num_backward_tokens,
                    num_max_tokens_per_rank,
                    num_topk, kHidden,
                    background_warp_idx - kReduceFirstWarp,
                    background_lane_idx);
          }
        }
        if (background_warp_idx >= kQuantFirstWarp &&
            background_warp_idx < kQuantFirstWarp + kQuantWarps) {
            detail::k3_mxfp8_stream_dw13_operands_background<
                kHidden, kIntermediateHidden, kNumExperts, kBlockM,
                kNumSMs, kNumThreads, false,
                kQuantFirstWarp, kK3MxFp8DW13QuantNumEngines,
                kK3MxFp8DW13QuantScratchBegin,
                Overlap::kExactSchedulerRoleMask,
                kUseExactEpilogueRing>(
                    expert_counts, backward_ranges_ptr, k_capacity,
                    scale_arena_source, tensor_map_pack, state,
                    smem_buffer, background_warp_idx,
                    background_lane_idx);
        }
    };
    if constexpr (kUseExactEpilogueRing) {
        using Lifecycle = K3MxFp8EpilogueGroupedConsumerLifecycle<
            kNumExperts, kK3MaxBackwardRanges,
            kHidden, 2u * kIntermediateHidden, kBlockM>;
        const auto ring = k3_mxfp8_make_epilogue_panel_ring<
            kHidden, 2u * kIntermediateHidden, kBlockM>(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(
                    backward_grad_y)),
                num_max_tokens_per_rank,
                backward_ranges.total_pool_rows / kBlockM,
                expected_epoch ^ 0x80000000u);
        DG_DEVICE_ASSERT(
            ring.total_pool_blocks <= ring.depth &&
            ring.depth != 0u);
        constexpr uint32_t kLifecycleOffset =
            math::constexpr_align(
                kK3MxFp8DW13QuantBodySmemBytes, 8u);
        static_assert(
            kLifecycleOffset + sizeof(Lifecycle) <=
                kK3MxFp8DW13QuantScratchBegin,
            "Ring lifecycle must fit before exact quantizer scratch");
        auto* const lifecycle = reinterpret_cast<Lifecycle*>(
            smem_buffer + kLifecycleOffset);
        if (threadIdx.x == 0u) {
            *lifecycle = {
                ring, expert_counts, &backward_ranges,
                state + Prefix::kValuePrefix,
                state + Prefix::kPhysicalRangePrefix};
        }
        __syncthreads();
        Sm100K3MxFp8NoInputTileRetired no_input_tile_retired;
        sm100_k3_mxfp8_three_term_grouped_wgrad_body<
            Config, Provider, TerminalResources,
            decltype(background_work),
            Sm100K3MxFp8NoInputTileRetired, Lifecycle>(
                reinterpret_cast<int*>(
                    const_cast<sched::ExternalKGroupedRangeStream*>(
                        &stream)),
                state[Prefix::kValuePrefix + kNumExperts],
                ring_maps[0], ring_maps[1], maps[2], maps[3],
                maps[4], maps[5], maps[6], maps[7],
                tensor_map_d, smem_buffer, false,
                background_work, no_input_tile_retired,
                nullptr, nullptr, lifecycle);

        // Every produced generation is now reachable by the selected
        // consumer. Distribute the terminal close checks after the grouped
        // body; no pre-producer wait or whole-grid rendezvous is needed.
        for (uint32_t slot =
                 static_cast<uint32_t>(blockIdx.x) * kNumThreads +
                     threadIdx.x;
             slot < ring.depth;
             slot += kNumSMs * kNumThreads) {
            k3_mxfp8_epilogue_ring_wait_terminal_retirement(ring, slot);
        }
        __syncthreads();
    } else {
        sm100_k3_mxfp8_three_term_grouped_wgrad_body<
            Config, Provider, TerminalResources>(
                reinterpret_cast<int*>(
                    const_cast<sched::ExternalKGroupedRangeStream*>(
                        &stream)),
                state[Prefix::kValuePrefix + kNumExperts],
                maps[0], maps[1], maps[2], maps[3],
                maps[4], maps[5], maps[6], maps[7],
                tensor_map_d, smem_buffer, false,
                background_work);
    }
}

/** Terminal exact-wgrad continuation for the fused K3 backward parent.
 *
 * One terminal call prevents the parent's completed W13 pipeline state from
 * remaining live across operand production and the two dynamic UMMA bodies.
 * Host-specialized input maps remain in grid-constant parameter space, while
 * communication callbacks, scheduler mailboxes, quantization, and TMEM
 * lifetime remain inside the original CUDA kernel invocation.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads, bool kAccumulateWgrad,
          uint32_t kBatchTasks, uint32_t kNumProducerCTAs,
          uint32_t kW13DgradCTAs, bool kClearEmptyOutputs>
CUTLASS_DEVICE __noinline__ void k3_mxfp8_run_overlap_suffix(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t k_capacity,
        cutlass::bfloat16_t* grad_ye_output,
        cutlass::bfloat16_t* h_weighted_output,
        cutlass::bfloat16_t* grad_gate_up_output,
        cutlass::bfloat16_t* x_pool_output,
        cutlass::bfloat16_t* grad_y_unweighted_output,
        const cutlass::bfloat16_t* down_unweighted_output,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        const cute::TmaDescriptor& dw2_output_map,
        const cute::TmaDescriptor& dw13_output_map,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t launch_epoch,
        const layout::SymBuffer<kNumRanks> backward_sym_buffer,
        const layout::Workspace backward_workspace,
        cutlass::bfloat16_t* w2_output,
        cutlass::bfloat16_t* w13_output,
        const bool clear_empty_outputs,
        cutlass::bfloat16_t* backward_grad_x_output,
        const cutlass::bfloat16_t* backward_grad_y,
        const uint32_t num_backward_tokens,
        const uint32_t first_range_tokens,
        const uint32_t second_range_begin,
        const uint32_t num_topk) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    static_assert(
        kNumProducerCTAs > 0u &&
            kW13DgradCTAs + kNumProducerCTAs == kNumSMs &&
            kNumProducerCTAs % 2u == 0u &&
            kW13DgradCTAs % 2u == 0u,
        "Exact overlap must partition complete CTA clusters");
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        asm volatile(
            "st.release.gpu.global.u32 [%0], %1;"
            :: "l"(state + Overlap::kEpoch), "r"(launch_epoch)
            : "memory");
    }
    if (blockIdx.x >= kW13DgradCTAs) {
        const uint32_t producer_cta_idx =
            static_cast<uint32_t>(blockIdx.x) - kW13DgradCTAs;
        uint32_t produced_total_k = 0u;
        if (backward_ranges.num_ranges == 1u) {
            produced_total_k =
                produce_k3_mxfp8_dw2_expert_sticky_operands<
                    kHidden, kIntermediateHidden,
                    kNumExperts, kBlockM, kNumSMs, kNumThreads>(
                        expert_counts, backward_ranges,
                        grad_ye_output, h_weighted_output, k_capacity,
                        x_pool_output, grad_y_unweighted_output,
                        down_unweighted_output, grad_ye_output,
                        scale_arena_source, tensor_map_pack,
                        state, smem_buffer);
        } else {
            produced_total_k =
                produce_k3_mxfp8_dw2_overlap_operands<
                    kHidden, kIntermediateHidden,
                    kNumExperts, kBlockM, kNumSMs, kNumThreads,
                    kNumProducerCTAs>(
                        expert_counts, backward_ranges,
                        grad_ye_output, h_weighted_output, k_capacity,
                        x_pool_output, grad_y_unweighted_output,
                        down_unweighted_output, grad_ye_output,
                        scale_arena_source, tensor_map_pack,
                        state, smem_buffer, producer_cta_idx);
        }
        DG_DEVICE_ASSERT(
            produced_total_k ==
                state[Prefix::kValuePrefix + kNumExperts]);
    }

    k3_mxfp8_run_dynamic_unified_wgrad_overlap<
        kHidden, kIntermediateHidden,
        kNumExperts, kBlockM, kNumSMs, kNumRanks, kNumThreads,
        kAccumulateWgrad, kBatchTasks, kClearEmptyOutputs>(
            state, launch_epoch, expert_counts, backward_ranges,
            k_capacity, scale_arena_source, tensor_map_pack,
            dw2_output_map, dw13_output_map,
            &backward_sym_buffer, &backward_workspace, smem_buffer,
            w2_output, w13_output, clear_empty_outputs,
            backward_grad_x_output, backward_grad_y,
            num_backward_tokens, first_range_tokens,
            second_range_begin,
            backward_workspace.num_max_tokens_per_rank,
            num_topk);
}
#endif

/** Pull exact source X and reclaim fixed-top-k combine plane one.
 *
 * This helper is called by W13-idle warp three only.  Keeping the long TMA
 * and rank-barrier loop out of the already register-dense parent preserves
 * the W13 epilogue's register budget without changing any storage lifetime.
 */
template <
    uint32_t kHidden, uint32_t kNumExperts, uint32_t kBlockM,
    uint32_t kNumSMs, uint32_t kNumRanks>
CUTLASS_DEVICE __noinline__ void
k3_exact_source_x_pull_and_reclaim_plane_one(
    const int* expert_counts,
    const layout::TokenSrcMetadata* token_src_metadata,
    const cutlass::bfloat16_t* backward_x,
    cutlass::bfloat16_t* x_pool_output,
    const layout::SymBuffer<kNumRanks>& backward_sym_buffer,
    const layout::Workspace& backward_workspace,
    const uint32_t pool_block_begin,
    cutlass::bfloat16_t* pull_buffer,
    cutlass::arch::ClusterTransactionBarrier* pull_mbarrier) {
    const uint32_t lane_idx = ptx::get_lane_idx();
    uint32_t pull_phase = 0u;
    uint32_t pool_block_offset = pool_block_begin;

    #pragma unroll 1
    for (uint32_t expert_idx = 0u;
         expert_idx < kNumExperts; ++expert_idx) {
        const uint32_t num_tokens = static_cast<uint32_t>(
            __ldg(expert_counts + expert_idx));
        const uint32_t num_blocks =
            math::ceil_div(num_tokens, kBlockM);
        const uint32_t num_padded_tokens = num_blocks * kBlockM;
        for (uint32_t token_idx = blockIdx.x;
             token_idx < num_padded_tokens;
             token_idx += kNumSMs) {
            const bool valid_row = token_idx < num_tokens;
            const uint32_t pool_row =
                pool_block_offset * kBlockM + token_idx;
            if (valid_row) {
                const auto metadata = token_src_metadata[pool_row];
                if (lane_idx == 0u) {
                    const auto* const remote_x =
                        backward_sym_buffer.map(
                            backward_x +
                                static_cast<uint64_t>(
                                    metadata.token_idx) *
                                    kHidden,
                            metadata.rank_idx);
                    ptx::tma_load_1d(
                        pull_buffer, remote_x, pull_mbarrier,
                        kHidden * sizeof(cutlass::bfloat16_t));
                    ptx::mbarrier_arrive_and_set_tx(
                        pull_mbarrier,
                        kHidden * sizeof(cutlass::bfloat16_t));
                    ptx::mbarrier_wait_and_flip_phase(
                        pull_mbarrier, pull_phase);
                }
            } else {
                for (uint32_t col = lane_idx;
                     col < kHidden; col += 32u)
                    pull_buffer[col] = cutlass::bfloat16_t(0.0f);
            }
            __syncwarp();
            if (!valid_row)
                cutlass::arch::fence_view_async_shared();
            if (lane_idx == 0u) {
                ptx::tma_store_1d(
                    x_pool_output +
                        static_cast<uint64_t>(pool_row) * kHidden,
                    pull_buffer,
                    kHidden * sizeof(cutlass::bfloat16_t));
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            }
            __syncwarp();
        }
        pool_block_offset += num_blocks;
    }

    // Bridge completed x-pool TMA stores to global visibility before the
    // source-warp grid/rank edge advertises that every pull has retired.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    constexpr uint32_t kPlaneOneGridSyncIndex = 3u;
    const auto source_warp_sync = []() { __syncwarp(); };
    comm::nvlink_barrier<
        kNumRanks, kNumSMs, 32u,
        kPlaneOneGridSyncIndex, 9u>(
            backward_workspace, backward_sym_buffer,
            blockIdx.x, lane_idx, source_warp_sync);

    auto* const source_plane =
        const_cast<cutlass::bfloat16_t*>(backward_x);
    const uint64_t values_per_plane =
        static_cast<uint64_t>(
            backward_workspace.num_max_tokens_per_rank) *
        kHidden;
    for (uint64_t offset =
             static_cast<uint64_t>(blockIdx.x) * 32u + lane_idx;
         offset < values_per_plane;
         offset += static_cast<uint64_t>(kNumSMs) * 32u) {
        source_plane[offset] = cutlass::bfloat16_t(0.0f);
    }

    comm::nvlink_barrier<
        kNumRanks, kNumSMs, 32u,
        kPlaneOneGridSyncIndex, 10u>(
            backward_workspace, backward_sym_buffer,
            blockIdx.x, lane_idx, source_warp_sync);
}

/** Scatter one CTA-owned W13 N tile of deferred slot-one dX. */
template <
    uint32_t kHidden, uint32_t kBlockM, uint32_t kBlockN,
    uint32_t kNumRanks, uint32_t kNumThreads>
CUTLASS_DEVICE __noinline__ void
k3_scatter_deferred_plane_one_grad_x_tile(
    const layout::TokenSrcMetadata* token_src_metadata,
    const cutlass::bfloat16_t* staged_grad_x,
    const cutlass::bfloat16_t* backward_x,
    const layout::SymBuffer<kNumRanks>& backward_sym_buffer,
    const uint32_t pool_block_offset,
    const uint32_t m_block_idx,
    const uint32_t n_block_idx,
    const uint32_t valid_m) {
    constexpr uint32_t kValuesPerVector =
        sizeof(uint4) / sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kWarpsPerCTA = kNumThreads / 32u;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t pool_block = pool_block_offset + m_block_idx;

    for (uint32_t local_m = warp_idx;
         local_m < valid_m; local_m += kWarpsPerCTA) {
        const uint32_t pool_row = pool_block * kBlockM + local_m;
        const auto metadata = token_src_metadata[pool_row];
        if (metadata.topk_idx != 1u)
            continue;
        const uint32_t out_col = n_block_idx * kBlockN;
        auto* const remote_slot_one =
            backward_sym_buffer.map(
                const_cast<cutlass::bfloat16_t*>(backward_x) +
                    static_cast<uint64_t>(metadata.token_idx) *
                        kHidden +
                    out_col,
                metadata.rank_idx);
        const auto* const staged_slot_one =
            staged_grad_x +
            static_cast<uint64_t>(pool_row) * kHidden + out_col;
        for (uint32_t n = lane_idx * kValuesPerVector;
             n < kBlockN; n += 32u * kValuesPerVector) {
            *reinterpret_cast<uint4*>(remote_slot_one + n) =
                    *reinterpret_cast<const uint4*>(
                        staged_slot_one + n);
        }
    }
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
    bool kMultiRangeBackward = false,
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
    const __grid_constant__ K3BackwardRangeSet backward_ranges,
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
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w2_wgrad_slot_a,
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w2_wgrad_slot_b,
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w2_wgrad_slot_d,
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w13_wgrad_slot_a,
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w13_wgrad_slot_b,
    const __grid_constant__
        K3MxFp8WgradAuxSlot<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>
            tensor_map_w13_wgrad_slot_d,
    const __grid_constant__
        K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>
            tensor_map_mxfp8_wgrad_pack,
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
    static_assert(
        k3_mxfp8_wgrad_tensor_map_pack_abi<cute::TmaDescriptor>(),
        "K3 exact-wgrad TensorMap pack ABI changed");
    static_assert(
        k3_mxfp8_wgrad_aux_slot_abi<
            cute::TmaDescriptor, cutlass::bfloat16_t,
            cutlass::float_e4m3_t>(),
        "K3 wgrad auxiliary-slot ABI changed");
    const auto& tensor_map_w2_wgrad_a =
        tensor_map_w2_wgrad_slot_a.legacy_map;
    const auto& tensor_map_w2_wgrad_b =
        tensor_map_w2_wgrad_slot_b.legacy_map;
    const auto& tensor_map_w2_wgrad_d =
        tensor_map_w2_wgrad_slot_d.legacy_map;
    const auto& tensor_map_w13_wgrad_a =
        tensor_map_w13_wgrad_slot_a.legacy_map;
    const auto& tensor_map_w13_wgrad_b =
        tensor_map_w13_wgrad_slot_b.legacy_map;
    const auto& tensor_map_w13_wgrad_d =
        tensor_map_w13_wgrad_slot_d.legacy_map;

    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumDispatchThreads =
        kNumRanks > 1 ? 128 : 0;
    constexpr uint32_t kNumDispatchWarps =
        kNumDispatchThreads / 32;
    // K3's production residual-MXFP8 path can construct every W2 activation
    // representation in the reverse-dispatch warpgroup.  Keep this selector
    // narrow: the fallback specializations retain the proven phase-ordered
    // dispatch and conversion path below.
    constexpr bool kResidualPipelinedGradYDispatch =
        kNumRanks > 1 && !kBF16Mode &&
        kResidualMXFP8Dgrad && kBuildResidualMXFP8Weights &&
        !kPhaseOrderedWeightDequant &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        !kDispatchInputsPrepared;
    // The exact native-MXFP4 specialization keeps BF16 W2 activations and
    // canonical BF16 dequantized weights.  Its only scheduling change is to
    // stream reverse dispatch, POST_DOWN route weighting/adjoint, and source-X
    // staging from the four compile-empty prepared-replay warps while W2
    // consumes independently published pool blocks.  Restricting this to the
    // full-training SiTU shape prevents any inference or fallback path from
    // silently changing phase ordering.
    constexpr bool kExactBF16PipelinedGradYDispatch =
        kNumRanks > 1 && !kBF16Mode &&
        !kInlineWeightDequant && !kPhaseOrderedWeightDequant &&
        !kResidualMXFP8Dgrad && !kBuildResidualMXFP8Weights &&
        kCompileW13Dgrad && kInlineWgrad && kComputeRouteGrad &&
        kExactSourceX && kGateUpPrepared && !kInputsPrepared &&
        kActivationType == ActivationType::SiTU &&
        kSituBeta == 4.0f && kSituLinearBeta == 25.0f &&
        kHidden == 3584 && kIntermediateHidden == 3072 &&
        kNumExperts == 112 &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        !kDispatchInputsPrepared;
    static_assert(
        !kMultiRangeBackward || kExactBF16PipelinedGradYDispatch,
        "multi-range backward is restricted to exact K3 SiTU training");
    static_assert(
        !kMultiRangeBackward || kNumRanks == 8,
        "the first multi-range ABI milestone is EP=8 only");
    static_assert(
        !kMultiRangeBackward ||
            (K3MultiRangeWgradBodyContract::kLogicalBlockN == 256 &&
             K3MultiRangeWgradBodyContract::kAccumulatorStages == 2 &&
             !K3MultiRangeWgradBodyContract::kPairAdjacentNTiles),
        "multi-range must retain the N256 two-accumulator wgrad body");
    constexpr bool kPipelinedGradYDispatch =
        kResidualPipelinedGradYDispatch ||
        kExactBF16PipelinedGradYDispatch;
    // Initial grad-y dispatch and the later route/exact-X pipeline run in
    // distinct phases but need independent transaction-barrier parity.
    constexpr uint32_t kNumDispatchBarrierStages = 2;
    constexpr uint32_t kNumDispatchBarriers =
        kNumDispatchWarps * kNumDispatchBarrierStages;
    constexpr uint32_t kReplayEpilogueWarpEnd =
        (kNumNonEpilogueThreads + kNumEpilogueThreads) / 32;
    constexpr bool kUseReducedW2ProducerSet =
        kPipelinedGradYDispatch && kGateUpPrepared && kExactSourceX &&
        kInlineResidualMXFP8Dgrad;
    constexpr bool kUsePreparedReplayDispatch =
        kUseReducedW2ProducerSet ||
        kExactBF16PipelinedGradYDispatch;
    // K3's split weight-cache specialization has four dispatch warps and a
    // 24-warp W2 epilogue. Once reverse dispatch drains, borrow one epilogue
    // warpgroup so eight warps share the 16 independent MXFP4 conversion
    // tasks. The remaining 20 epilogue warps retain the historically verified
    // W2 store map; no UMMA, TMA-load, or symmetric-memory role moves.
    constexpr bool kWideDispatchW2WeightBuilder =
        kUseReducedW2ProducerSet && kCompileW13Dgrad && kInlineWgrad &&
        kBuildResidualMXFP8Weights &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        kHidden == 3584 && kIntermediateHidden == 3072 &&
        kNumExperts == 112;
    constexpr uint32_t kNumDispatchW2WeightBuilderWarps =
        kWideDispatchW2WeightBuilder ? 8 : kNumDispatchWarps;
    constexpr uint32_t kNumDispatchW2WeightBuilderThreads =
        kNumDispatchW2WeightBuilderWarps * 32;
    // Prepared K3 preactivations make the replay epilogue empty. Reuse that
    // otherwise-idle warpgroup for pipelined reverse dispatch. The narrow
    // control retains 24 W2 epilogue warps; the wide builder above deliberately
    // borrows the adjacent warpgroup and retains the verified 20-warp map.
    constexpr uint32_t kDispatchWarpStart =
        kUsePreparedReplayDispatch
        ? 4
        : kReplayEpilogueWarpEnd;
    // Split the 28 non-dispatch warps in proportion to the exact number of
    // canonical BF16 tiles: W13 has 2x W2's tile count. The W2 role is
    // intentionally noncontiguous so v409's warps 4..7 stay dedicated to
    // reverse dispatch; named barriers synchronize participants by count and
    // do not require contiguous warp IDs.
    constexpr bool kOverlapInitialBF16WeightDequant =
        DG_EXPERIMENTAL_K3_INITIAL_BF16_DEQUANT_OVERLAP &&
        kNumRanks == 8 && kExactBF16PipelinedGradYDispatch &&
        !kAccumulateWgrad && kHidden == 3584 &&
        kIntermediateHidden == 3072 && kNumExperts == 112 &&
        kNumThreads == 1024;
    constexpr uint32_t kInitialW2DequantFirstEnd = 4;
    constexpr uint32_t kInitialW2DequantSecondStart = 8;
    constexpr uint32_t kInitialW2DequantSecondEnd = 13;
    constexpr uint32_t kInitialW13DequantWarpStart = 13;
    constexpr uint32_t kInitialW2DequantWarps =
        kInitialW2DequantFirstEnd +
        (kInitialW2DequantSecondEnd -
         kInitialW2DequantSecondStart);
    constexpr uint32_t kInitialW13DequantWarps =
        kNumThreads / 32 - kInitialW13DequantWarpStart;
    constexpr uint32_t kInitialW2DequantThreads =
        kInitialW2DequantWarps * 32;
    constexpr uint32_t kInitialW13DequantThreads =
        kInitialW13DequantWarps * 32;
    // CUTLASS adds its eight reserved barriers to these user IDs, giving
    // hardware barriers 13 and 14. Reverse dispatch directly owns 15.
    constexpr uint32_t kInitialW2DequantNamedBarrier = 5;
    constexpr uint32_t kInitialW13DequantNamedBarrier = 6;
    // W13's optional route reducer must remain on the original idle
    // warpgroup.  The reduced W2 producer map is phase-local: W13 restores
    // warps 3..7 as its seven-producer packed-weight bridge.
    constexpr uint32_t kW13RouteWarpStart =
        kReplayEpilogueWarpEnd;
    // Inline MXFP4 conversion is scalar CUDA work rather than UMMA work.  Use
    // six warps (1 and 3..7) to feed the dgrad pipeline; 24 warps remain for
    // the pool epilogue and remote grad-x scatter.  The non-inline TMA path
    // retains its original four-warp producer/MMA prefix.
    constexpr uint32_t kDgradEpilogueWarpStart =
        (kInlineWeightDequant || kResidualMXFP8Dgrad) ? 8 : 4;
    constexpr uint32_t kNumDgradEpilogueThreads =
        kNumThreads - kDgradEpilogueWarpStart * 32;
    // During W2 the dispatch warpgroup remains live until every pool block has
    // published its primary/residual activation planes.  Shift only W2's
    // epilogue role; W13 keeps the original 24-warp mapping for this prototype.
    constexpr uint32_t kW2DgradEpilogueWarpStart =
        kPipelinedGradYDispatch
        ? kDispatchWarpStart + kNumDispatchW2WeightBuilderWarps
        : kDgradEpilogueWarpStart;
    constexpr uint32_t kNumW2DgradEpilogueThreads =
        kNumThreads - kW2DgradEpilogueWarpStart * 32;
    constexpr uint32_t kNumInlineWeightProducerWarps = 6;
    constexpr uint32_t kNumInlineWeightProducerThreads =
        kNumInlineWeightProducerWarps * 32;
    constexpr uint32_t kInlineWeightProducerBarrier = 1;
    constexpr uint32_t kNumW2ResidualProducerWarps =
        kUseReducedW2ProducerSet ? 3 : 7;
    constexpr uint32_t kNumW2ResidualProducerThreads =
        kNumW2ResidualProducerWarps * 32;
    constexpr uint32_t kNumW13ResidualProducerWarps = 7;
    constexpr uint32_t kNumW13ResidualProducerThreads =
        kNumW13ResidualProducerWarps * 32;
    // A/B readiness is centralized on CTA 0 because the 2-SM UMMA leader must
    // observe both operands from both CTAs. Each CTA contributes one A and one
    // B completion, for four cluster arrivals in every path.
    constexpr uint32_t kNumDgradFullBarrierArrivals = 4;
    constexpr uint32_t kResidualWeightProducerBarrier = 2;
    // TMA cache stores can overlap the current UMMA, but every producer must
    // observe their completion before recycling the shared-memory stage.
    constexpr uint32_t kResidualWeightCacheStoreDoneBarrier = 4;
    // The reverse-dispatch warpgroup has a private 28 KiB shared-memory
    // partition. Once those four warps finish their assigned rows, reuse that
    // partition as a readiness-driven W2 MXFP4 -> MXFP8 cache builder while
    // W2 UMMA continues in the other warp roles. In K3, the adjacent epilogue
    // warpgroup joins only after the four dispatch warps drain; barrier five is
    // private to this eight-warp handoff and no CTA-wide barrier is introduced.
    constexpr uint32_t kDispatchWeightBuilderBarrier = 5;
    // Post-dW2 W13 uses warp zero for TMA plus 16 converter warps. The other
    // 15 warps wait only at the phase boundary, not at every weight tile.
    constexpr uint32_t kPostDW2WeightBuilderBarrier = 6;
    // In the split K3 path, W13's weight cache uses warp 0 plus converter
    // warps 4..19.  The complementary 15 warps quantize W13 activations in
    // parallel and synchronize row-staging reuse on the final user named
    // barrier.  The two roles touch disjoint shared-memory partitions.
    constexpr uint32_t kPostDW2ActivationBuilderBarrier = 7;
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
    // The K3 specialization dedicates a bounded prefix of complete 2-CTA
    // clusters to exact packed-weight conversion while the remaining clusters
    // execute W2 dgrad. Producers publish one epoch per compact MXFP8 tile;
    // consumers may therefore start as soon as their own B tile is ready and
    // never wait for a whole-grid conversion pass.
    //
    // v340 measured 21.970 ms of W2+W13 conversion work versus a 17.554 ms
    // W2-dgrad+dW2 overlap window. Balancing those streams over 74 clusters
    // gives ceil(74 * 21.970 / (21.970 + 17.554)) = 42 producers.
    constexpr bool kConcurrentResidualWeightCache = false;
    constexpr uint32_t kNumWeightProducerClusters =
        kConcurrentResidualWeightCache ? 42 : 0;
    constexpr uint32_t kNumWeightProducerCTAs =
        2 * kNumWeightProducerClusters;
    constexpr uint32_t kNumW2ConsumerCTAs =
        kNumSMs - kNumWeightProducerCTAs;
    const bool is_weight_producer_cta =
        kConcurrentResidualWeightCache &&
        blockIdx.x < kNumWeightProducerCTAs;
    const uint32_t w2_consumer_block_idx =
        blockIdx.x - kNumWeightProducerCTAs;

    // Fallback specializations retain MoK's local packed-weight path.
    constexpr bool kPrefixedResidualWeightCache = false;
    constexpr bool kOnDemandResidualWeightCache =
        kInlineResidualMXFP8Dgrad &&
        !kConcurrentResidualWeightCache;
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
    // Split the two residual-weight producers at their first useful windows:
    // dispatch warps publish W2 during W2 dgrad, while a full-CTA producer
    // publishes W13 immediately after early dW2. Keeping W13 out of the W2
    // critical path is valid only when early dW2 supplies that CTA-wide phase.
    constexpr bool kSplitResidualWeightCache =
        kOnDemandResidualWeightCache &&
        kUseReducedW2ProducerSet &&
        kEarlyW2Wgrad &&
        kHidden == 3584 &&
        kIntermediateHidden == 3072 &&
        kNumExperts == 112;
    constexpr bool kCoSchedulePostDW2W13Builders =
        kSplitResidualWeightCache && kBuildW13ResidualActsOnce;
    // Without the dispatch-side builder, the first M wave remains the local
    // producer for every weight tile.  The dispatch-side schedule instead
    // publishes all tiles independently and lets every M wave be a consumer.
    constexpr bool kFirstMWaveBuildsResidualWeightCache =
        kOnDemandResidualWeightCache &&
        !kSplitResidualWeightCache;
    constexpr uint32_t kResidualWeightCacheBarrier = 3;
    // Experimental quality/performance gate: retain the primary MXFP8 dgrad
    // product while measuring whether the error-feedback product is required
    // for Kimi K3's cosine threshold.
    constexpr bool kApplyResidualDgradCorrection = true;
    constexpr uint32_t kNumW2WeightTileStates =
        kNumExperts * (kHidden / DGRAD_BLOCK_K) *
        kNumDgradBlockNs;
    // The terminal BF16 backend is intentionally limited to the measured
    // one-range or exact two-range, overwrite-only K3 EP8 specialization. Its
    // A/B operands remain the immutable training-reference BF16 tensors bound
    // by the legacy TMA maps; the mutually-exclusive exact-MXFP8 producer may
    // not alias them.
    constexpr bool kK3BranchMajorBF16WgradTail =
        DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL &&
        kCompileW13Dgrad && kInlineWgrad && kNumRanks == 8 &&
        !kBF16Mode && !kInlineWeightDequant &&
        !kPhaseOrderedWeightDequant &&
        !kInlineResidualMXFP8Dgrad && !kResidualMXFP8Dgrad &&
        !kBuildResidualMXFP8Weights && kExactSourceX &&
        kGateUpPrepared && kDirectRemoteGradX && kClearWgradPadding &&
        !kAccumulateWgrad && kComputeRouteGrad &&
        kActivationType == ActivationType::SiTU &&
        kSituBeta == 4.0f && kSituLinearBeta == 25.0f &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        kCombineOrderMode == CombineOrderMode::FixedTopK &&
        kHidden == 3584 && kIntermediateHidden == 3072 &&
        kNumExperts == 112 && BLOCK_M == 192 &&
        kNumSMs == 148 && kNumThreads == 1024;
    constexpr bool kK3BranchMajorBF16DynamicTail =
        DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL &&
        kK3BranchMajorBF16WgradTail;
    constexpr bool kK3MxFp8ThreeTermWgrad =
        DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD &&
        kCompileW13Dgrad && kInlineWgrad &&
        (kNumRanks == 4 || kNumRanks == 8) &&
        !kBF16Mode && kClearWgradPadding &&
        !kInlineResidualMXFP8Dgrad && !kResidualMXFP8Dgrad &&
        !kBuildResidualMXFP8Weights && kExactSourceX &&
        kGateUpPrepared && kDirectRemoteGradX && kComputeRouteGrad &&
        kActivationType == ActivationType::SiTU &&
        kSituBeta == 4.0f && kSituLinearBeta == 25.0f &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        kCombineOrderMode == CombineOrderMode::FixedTopK &&
        kHidden == 3584 && kIntermediateHidden == 3072 &&
        kNumExperts == 112 && BLOCK_M == 192 &&
        kNumSMs == 148 && kNumThreads == 1024;
    constexpr bool kK3MxFp8WgradOverlap =
        kK3MxFp8ThreeTermWgrad &&
        DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP;
    // Packed ranges defer dW2 operand production to the unified suffix.  Its
    // exact physical-row dependency masks replace the older W13-time elastic
    // producer, whose expert-local A/B handoff was unsafe for range-major
    // sources.  One-range keeps the already validated low-latency producer.
    constexpr bool kK3MxFp8SuffixPanelStream =
        kK3MxFp8WgradOverlap && kMultiRangeBackward;
    constexpr uint32_t kK3MxFp8NumClusters = kNumSMs / 2u;
    constexpr uint32_t kK3MxFp8DW2ProducerClusters =
        DG_EXPERIMENTAL_K3_MXFP8_DW2_PRODUCER_CLUSTERS;
    constexpr uint32_t kK3MxFp8DW2ProducerCTAs =
        2u * kK3MxFp8DW2ProducerClusters;
    constexpr uint32_t kK3MxFp8W13DgradCTAs =
        kNumSMs - kK3MxFp8DW2ProducerCTAs;
    constexpr uint32_t kK3MxFp8PersistentDW2ProducerClusters =
        DG_EXPERIMENTAL_K3_MXFP8_PERSISTENT_DW2_PRODUCER_CLUSTERS;
    constexpr uint32_t kK3MxFp8PersistentDW2ProducerCTAs =
        2u * kK3MxFp8PersistentDW2ProducerClusters;
    constexpr uint32_t kK3MxFp8PersistentDW2ConsumerCTAs =
        kNumSMs - kK3MxFp8PersistentDW2ProducerCTAs;
    constexpr uint32_t kK3MxFp8WgradBatchTasks =
        DG_EXPERIMENTAL_K3_MXFP8_WGRAD_BATCH_TASKS;
    using K3MxFp8OverlapState = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD ||
            kK3MxFp8ThreeTermWgrad,
        "The three-term MXFP8 wgrad gate is exact K3 EP4/EP8 SiTU only");
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL ||
            kK3BranchMajorBF16WgradTail,
        "The terminal BF16 tail is exact one- or two-range K3 EP8 only");
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL ||
            kK3BranchMajorBF16DynamicTail,
        "The dynamic BF16 tail is exact one- or two-range K3 EP8 only");
    DG_STATIC_ASSERT(
        !kK3BranchMajorBF16WgradTail || !kK3MxFp8ThreeTermWgrad,
        "BF16 and three-term MXFP8 wgrad suffixes are mutually exclusive");
    DG_STATIC_ASSERT(
        !kK3MxFp8ThreeTermWgrad ||
            (kHidden % 128u == 0u &&
             kIntermediateHidden % 128u == 0u &&
             BLOCK_M % 32u == 0u),
        "K3 MXFP8 wgrad requires 128-wide features and group-32 K");
    DG_STATIC_ASSERT(
        !kK3MxFp8WgradOverlap ||
            (kK3MxFp8DW2ProducerClusters == 0u &&
             kK3MxFp8DW2ProducerCTAs == 0u &&
             kK3MxFp8W13DgradCTAs == kNumSMs &&
             kK3MxFp8W13DgradCTAs ==
                 2u * kK3MxFp8NumClusters),
        "Persistent exact overlap requires every complete cluster to run W13");
    DG_STATIC_ASSERT(
        !kK3MxFp8WgradOverlap ||
            (kK3MxFp8PersistentDW2ProducerClusters > 0u &&
             kK3MxFp8PersistentDW2ProducerCTAs > 0u &&
             kK3MxFp8PersistentDW2ProducerCTAs % 2u == 0u &&
             kK3MxFp8PersistentDW2ConsumerCTAs % 2u == 0u &&
             kK3MxFp8PersistentDW2ProducerCTAs < kNumSMs),
        "Post-W13 persistent dW2 roles must reserve complete CTA pairs");
    DG_STATIC_ASSERT(
        !kK3MxFp8WgradOverlap ||
            (kK3MxFp8WgradBatchTasks >= 32u &&
             kK3MxFp8WgradBatchTasks % 4u == 0u),
        "Exact dynamic batches must preserve epilogue phase reset");
    DG_STATIC_ASSERT(
        !kK3MxFp8WgradOverlap ||
            K3MxFp8OverlapState::kNumWords <= kNumW2WeightTileStates,
        "Exact overlap state exceeds retired W2 tile storage");
    constexpr bool kReadyWgradSchedule =
        DG_EXPERIMENTAL_K3_READY_WGRAD &&
        !kK3MxFp8WgradOverlap &&
        kCompileW13Dgrad && kInlineWgrad && kNumRanks == 8 &&
        !kBF16Mode && !kInlineWeightDequant &&
        !kPhaseOrderedWeightDequant &&
        !kInlineResidualMXFP8Dgrad && !kResidualMXFP8Dgrad &&
        !kBuildResidualMXFP8Weights && kExactSourceX &&
        kGateUpPrepared && kExactBF16PipelinedGradYDispatch &&
        kOverlapInitialBF16WeightDequant &&
        kDirectRemoteGradX &&
        kActivationType == ActivationType::SiTU &&
        kRouteWeightMode == RouteWeightMode::PostDown &&
        kHidden == 3584 && kIntermediateHidden == 3072 &&
        kNumExperts == 112 && BLOCK_M == 192 &&
        kNumSMs == 148;
    // The exact two-range branch-major suffix has immutable dW2 operands
    // before W13 starts.  Let each completed W13 cluster enter the existing
    // dynamic dW2 queue immediately; later clusters join the same queue after
    // retiring their local W13 work.  dW13 deliberately remains in its faster
    // terminal body after the rank-wide publication edge.
    constexpr bool kK3BranchMajorBF16EarlyDW2Overlap =
        kK3BranchMajorBF16DynamicTail && kReadyWgradSchedule &&
        kMultiRangeBackward;
    // The one-range terminal BF16 path stages source X in fixed-top-k combine
    // plane one. Keep that plane readable while W13 runs: idle warp three
    // pulls source X, W13 stages only slot-one dX in the now-dead unweighted
    // grad-y pool, and every other slot is scattered directly. After the
    // source warp's rank-wide read-retirement/clear protocol and a terminal
    // grid join, the staged slot is copied into the reclaimed plane. This is
    // deliberately narrower than the generic exact-X selector because the
    // dead-pool and terminal-publication proofs are one-range, overwrite-only
    // branch-major contracts.
    constexpr bool kOverlapExactSourceXPlaneOneWithW13 =
        kK3BranchMajorBF16DynamicTail &&
        kExactBF16PipelinedGradYDispatch &&
        !kReadyWgradSchedule && !kMultiRangeBackward &&
        kNumRanks == 8 && kDirectRemoteGradX &&
        kCombineOrderMode == CombineOrderMode::FixedTopK;
    DG_STATIC_ASSERT(
        !kOverlapExactSourceXPlaneOneWithW13 ||
            (kDispatchWarpStart == 4 && kNumDispatchWarps == 4 &&
             kNumDispatchBarrierStages == 2 &&
             kDgradEpilogueWarpStart == 4 &&
             !kWriteGradXPool && !kAccumulateWgrad),
        "W13 source-X overlap requires the idle warp-three and dead-pool ABI");
    constexpr bool kK3MxFp8DW13Hybrid =
        DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID &&
        kReadyWgradSchedule && kMultiRangeBackward &&
        kClearWgradPadding && !kAccumulateWgrad &&
        kCombineOrderMode == CombineOrderMode::FixedTopK &&
        kSituBeta == 4.0f && kSituLinearBeta == 25.0f &&
        kNumThreads == 1024;
    constexpr bool kK3MxFp8ExactEpilogueRing =
        DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING &&
        kK3MxFp8DW13Hybrid && kExactBF16PipelinedGradYDispatch &&
        kMultiRangeBackward && kNumRanks == 8 &&
        kCombineOrderMode == CombineOrderMode::FixedTopK;
    // The concurrent exact-epilogue consumer stores dW13 before W13 dgrad.
    // In the allocation-free training ABI the dW13 destination aliases the
    // canonical BF16 W13-dequant arena, so that early store must not be
    // followed by a TMA read from ``tensor_map_w13_dequant``.  Reuse the
    // existing bit-exact inline MXFP4 -> BF16 tile producer for W13 only.
    // W2 deliberately retains the non-inline path selected by the public
    // configuration; changing both dgrad operands would be a different
    // numerical/performance experiment.  Key the lifetime change off the
    // selected specialization, not the global experiment macro, so unrelated
    // kernel instantiations in the same extension keep their original path.
    constexpr bool kInlineW13WeightDequant =
        kInlineWeightDequant || kK3MxFp8ExactEpilogueRing;
    constexpr bool kK3TwoSegmentBF16ProgressiveWgrad =
        DG_EXPERIMENTAL_K3_TWO_SEGMENT_BF16_PROGRESSIVE_WGRAD &&
        kReadyWgradSchedule && kMultiRangeBackward &&
        kClearWgradPadding && !kAccumulateWgrad &&
        kCombineOrderMode == CombineOrderMode::FixedTopK &&
        kSituBeta == 4.0f && kSituLinearBeta == 25.0f &&
        kNumThreads == 1024;
    constexpr uint32_t kReadyNumClusters = kNumSMs / 2u;
    constexpr uint32_t kK3MxFp8DW13ShepherdClusters =
        DG_EXPERIMENTAL_K3_MXFP8_DW13_SHEPHERD_CLUSTERS;
    constexpr uint32_t kReadyInitialWgradClusters =
        DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS;
    constexpr uint32_t kReadyW13ProducerClusters =
        kReadyNumClusters - kReadyInitialWgradClusters;
    constexpr uint32_t kReadyW13ProducerCTAs =
        2u * kReadyW13ProducerClusters;
    constexpr uint32_t kReadyBatchTasks =
        DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS;
    // Twelve tasks preserve the four-task BF16 phase reset while dividing
    // both K3 terminal grids (168 dW2 and 336 dW13 tasks per expert).  This
    // cuts cursor/mailbox traffic without sacrificing expert-level balance.
    constexpr uint32_t kReadyThreeSegmentBatchTasks = 12u;

    // Reuse the retired W2 tile-state prefix.  W13 keeps using its disjoint
    // state range until each expert's last BF16 weight TMA read is acquired.
    // The layout deliberately matches ExternalKGroupedRangeProvider's compact
    // pool-prefix/active-expert ABI and adds only counters plus one publication
    // slot per physical two-CTA cluster.
    constexpr uint32_t kReadyPoolPrefixWord = 31u;
    constexpr uint32_t kReadyActiveExpertWord = 144u;
    constexpr uint32_t kReadyW13RetiredWord = 256u;
    constexpr uint32_t kReadyDW2ClusterSlotWord = 368u;
    constexpr uint32_t kReadyClusterSlotWords = 4u;
    constexpr uint32_t kReadyDW13ClusterSlotWord =
        kReadyDW2ClusterSlotWord +
        kReadyNumClusters * kReadyClusterSlotWords;
    constexpr uint32_t kReadyCompleteRoleMask = 0x7ffu;
    constexpr uint32_t kReadyStateWords =
        kReadyDW13ClusterSlotWord +
        kReadyNumClusters * kReadyClusterSlotWords;
    constexpr uint32_t kReadyRangeStateStride =
        (kReadyStateWords + 127u) & ~127u;
    // Once W2 dgrad retires, the scheduler prefix occupies at most three
    // compact range arenas. A fourth arena owns the immutable terminal union plus
    // independent dW2/dW13 cursors and mailboxes. Keeping the union disjoint is
    // required because late W13 producer clusters may still acquire either
    // range generation while early clusters are already consuming dW2 work.
    constexpr uint32_t kReadyTerminalUnionStateWord =
        kK3MaxBackwardRanges * kReadyRangeStateStride;
    // Reuse the following dead W2 tile-state words as one completion counter
    // per physical 192-row expert-pool block; this adds no allocation and is
    // disjoint from all three retained scheduler generations.
    constexpr uint32_t kDirectGradXPoolCompletionWord =
        kReadyTerminalUnionStateWord + kReadyRangeStateStride;
    constexpr uint32_t kK3MxFp8DW13HybridStateAlignmentWords = 32u;
    constexpr uint32_t kK3MxFp8DW13HybridStateSlackWords =
        kK3MxFp8DW13HybridStateAlignmentWords - 1u;
    constexpr uint32_t kK3MxFp8DW13HybridMaxPoolCompletionWords =
        kNumW2WeightTileStates - kDirectGradXPoolCompletionWord -
        K3MxFp8OverlapState::kNumWords -
        kK3MxFp8DW13HybridStateSlackWords;
    constexpr uint32_t kReadyMagicWord = 0u;
    constexpr uint32_t kReadyEpochWord = 1u;
    constexpr uint32_t kReadyDW2CursorWord = 2u;
    constexpr uint32_t kReadyDW13CursorWord = 3u;
    constexpr uint32_t kReadyActiveCountWord = 5u;
    constexpr uint32_t kReadyPoolBlocksWord = 6u;
    constexpr uint32_t kReadyDW2TasksWord = 7u;
    constexpr uint32_t kReadyDW13TasksWord = 8u;
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule ||
            kReadyInitialWgradClusters < kReadyNumClusters,
        "K3 ready wgrad must retain at least one producer cluster");
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule ||
            (kReadyBatchTasks >= 4u && kReadyBatchTasks % 4u == 0u),
        "K3 ready wgrad batches must preserve expert phase alignment");
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule || kReadyStateWords <= kNumW2WeightTileStates,
        "K3 ready scheduler exceeds retired W2 tile-state storage");
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule ||
            kDirectGradXPoolCompletionWord <=
                kNumW2WeightTileStates,
        "K3 range and two-segment schedulers exceed retired W2 storage");
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID ||
            kK3MxFp8DW13Hybrid,
        "The one-way MXFP8 dW13 hybrid is exact K3 EP8 multi-range only");
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING ||
            kK3MxFp8ExactEpilogueRing,
        "The exact epilogue ring is compile-only K3 EP8 multi-range dW13");
    DG_STATIC_ASSERT(
        !kK3MxFp8ExactEpilogueRing ||
            (kHidden == 3584u && 2u * kIntermediateHidden == 6144u &&
             BLOCK_M == 192u && kHidden / 128u == 28u &&
             kInlineW13WeightDequant && !kInlineWeightDequant),
        "The ring layout and W13-only inline dequant lifetime are K3-exact");
    DG_STATIC_ASSERT(
        !DG_EXPERIMENTAL_K3_TWO_SEGMENT_BF16_PROGRESSIVE_WGRAD ||
            kK3TwoSegmentBF16ProgressiveWgrad,
        "Progressive two-segment BF16 wgrad is exact K3 EP8 training only");
    DG_STATIC_ASSERT(
        !kK3TwoSegmentBF16ProgressiveWgrad || !kK3MxFp8DW13Hybrid,
        "BF16 progressive and exact-MXFP8 dW13 suffixes are mutually exclusive");
    DG_STATIC_ASSERT(
        !kK3MxFp8DW13Hybrid ||
            (kK3MxFp8DW13ShepherdClusters > 0u &&
             kK3MxFp8DW13ShepherdClusters < kReadyNumClusters),
        "Progressive hybrid pipeline must retain a nonempty BF16 dW2 shepherd prefix");
    DG_STATIC_ASSERT(
        !kK3MxFp8DW13Hybrid ||
            (kDirectGradXPoolCompletionWord +
                 kK3MxFp8DW13HybridMaxPoolCompletionWords +
                 kK3MxFp8DW13HybridStateSlackWords +
                 K3MxFp8OverlapState::kNumWords ==
             kNumW2WeightTileStates),
        "Hybrid state capacity proof must include the complete ready pool tail");
    // Early dW2 no longer needs W2's compact transpose. Reuse state words
    // after the grouped-layout prefix for one cursor per persistent CTA. A
    // cursor counts complete (expert, N) groups whose every K tile has been
    // release-published by the background converter. W13 wgrad continues to
    // read only the first kNumExperts words as its grouped layout.
    constexpr bool kBackgroundW13WeightCache =
        kSplitResidualWeightCache;
    // Six groups reach the measured dW2 overlap ceiling at K3 EP=4 while
    // retaining complete-group cursor publication for the K128 continuation.
    constexpr uint32_t kBackgroundW13GroupsPerCTA = 6;
    DG_STATIC_ASSERT(
        !kBackgroundW13WeightCache ||
            kNumExperts + kNumSMs <= kNumW2WeightTileStates,
        "W13 background cursors exceed dead W2 tile-state storage");
    auto* w13_background_group_cursors =
        weight_tile_states + kNumExperts;
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
    DG_STATIC_ASSERT(
        !kWideDispatchW2WeightBuilder ||
            (kDispatchWarpStart == 4 &&
             kNumDispatchWarps == 4 &&
             kNumDispatchW2WeightBuilderWarps == 8 &&
             kW2DgradEpilogueWarpStart == 12 &&
             kNumW2DgradEpilogueThreads == 640),
        "Wide W2 builder requires K3's 4/8/20 dispatch-builder-epilogue map");
    DG_STATIC_ASSERT(
        !kExactBF16PipelinedGradYDispatch ||
            (kDispatchWarpStart == 4 &&
             kNumDispatchWarps == 4 &&
             kW2DgradEpilogueWarpStart == 8 &&
             kNumW2DgradEpilogueThreads == 768),
        "Exact BF16 dispatch requires K3's 4/4/24 core-dispatch-epilogue map");
    DG_STATIC_ASSERT(
        !kOverlapInitialBF16WeightDequant ||
            (kDispatchWarpStart == kInitialW2DequantFirstEnd &&
             kNumDispatchWarps == 4 &&
             kDispatchWarpStart + kNumDispatchWarps ==
                 kInitialW2DequantSecondStart &&
             kInitialW2DequantSecondEnd ==
                 kInitialW13DequantWarpStart &&
             kInitialW2DequantThreads == 288 &&
             kInitialW13DequantThreads == 608),
        "Initial BF16 overlap requires K3's 9/4/19 dequant-dispatch map");
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
        !kConcurrentResidualWeightCache ||
            (kNumWeightProducerCTAs % 2 == 0 &&
             kNumW2ConsumerCTAs % 2 == 0 &&
             kNumWeightProducerCTAs < kNumSMs),
        "Concurrent weight conversion must reserve complete 2-CTA clusters");
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
    // Tag the per-pool-block grad-y completion epoch so no partial count can
    // equal another launch's ready value.
    uint32_t active_range_iteration = 0u;
    uint32_t active_range_index = 0u;
    uint32_t active_token_begin = 0u;
    uint32_t active_pool_row_begin = 0u;
    uint32_t active_pool_block_begin = 0u;
    uint32_t active_num_pool_rows = num_pool_rows;
    uint32_t active_num_acts_rows = num_acts_rows;
    uint32_t active_range_epoch = launch_epoch;
    bool active_accumulate_wgrad = kAccumulateWgrad;
    const int* active_expert_counts = expert_counts;
    uint32_t active_grad_y_counter_base =
        active_range_epoch ^ 0x80000000u;
    uint32_t active_grad_y_ready_value =
        active_grad_y_counter_base + BLOCK_M;
    const uint32_t combine_first_range_tokens =
        kMultiRangeBackward
        ? backward_ranges.ranges[0].num_tokens
        : num_backward_tokens;
    const uint32_t combine_second_range_begin =
        kMultiRangeBackward && backward_ranges.num_ranges > 1u
        ? backward_ranges.token_begin(1u)
        : combine_first_range_tokens;
    const uint32_t combine_third_range_begin =
        kMultiRangeBackward && backward_ranges.num_ranges > 2u
        ? backward_ranges.token_begin(2u)
        : combine_second_range_begin;
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
        [&](const uint32_t trace_site, const uint32_t watchdog_site) {
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
                        // Device printf itself faults under compute-sanitizer on
                        // SM103. Keep one source-distinct trap per barrier call
                        // so a line-info build identifies the deadlocked phase.
                        switch (watchdog_site) {
                            case 1: asm volatile("trap;"); break;
                            case 2: asm volatile("trap;"); break;
                            case 3: asm volatile("trap;"); break;
                            case 4: asm volatile("trap;"); break;
                            case 5: asm volatile("trap;"); break;
                            case 6: asm volatile("trap;"); break;
                            case 7: asm volatile("trap;"); break;
                            case 8: asm volatile("trap;"); break;
                            case 9: asm volatile("trap;"); break;
                            case 10: asm volatile("trap;"); break;
                            case 11: asm volatile("trap;"); break;
                            case 12: asm volatile("trap;"); break;
                            case 13: asm volatile("trap;"); break;
                            case 14: asm volatile("trap;"); break;
                            case 15: asm volatile("trap;"); break;
                            case 16: asm volatile("trap;"); break;
                            case 17: asm volatile("trap;"); break;
                            case 18: asm volatile("trap;"); break;
                            case 19: asm volatile("trap;"); break;
                            case 20: asm volatile("trap;"); break;
                            case 21: asm volatile("trap;"); break;
                            case 22: asm volatile("trap;"); break;
                            case 23: asm volatile("trap;"); break;
                            case 24: asm volatile("trap;"); break;
                            case 25: asm volatile("trap;"); break;
                            case 26: asm volatile("trap;"); break;
                            case 27: asm volatile("trap;"); break;
                            case 28: asm volatile("trap;"); break;
                            case 29: asm volatile("trap;"); break;
                            default: asm volatile("trap;"); break;
                        }
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
    // K3's exact-source W13 wgrad can reuse either POST_DOWN route-dot input
    // or, in the pipelined W2 schedule, the compact primary/residual dgrad
    // planes. Defer the source-X refill until every aliased reader retires; an
    // early refill would corrupt the route dot or overwrite W2's MXFP8 input.
    const bool late_exact_source_x =
        !kExactBF16PipelinedGradYDispatch &&
        !kBF16Mode && kExactSourceX &&
        (x_pool_output == grad_y_unweighted_output ||
         (kComputeRouteGrad &&
          kRouteWeightMode == RouteWeightMode::PostDown &&
          x_pool_output == down_unweighted_output) ||
         (kPipelinedGradYDispatch &&
          x_pool_output == grad_x_pool_output));
    if constexpr (kReadyWgradSchedule) {
        // v409 streams exact X from the prepared-replay dispatch warps and
        // therefore deliberately disables the legacy late-X alias path.  Both
        // streams already converge at the existing route/dispatch grid join
        // where v412 publishes scheduler state; refuse any third topology.
        DG_DEVICE_ASSERT(
            late_exact_source_x ||
            kExactBF16PipelinedGradYDispatch);
    }
    // Legacy FP8/FP4 replay materializes MegaMoE's [gate(8), up(8)] column
    // layout. Prepared forward preactivations, like the BF16 path, are already
    // canonical [all gate | all up]. Only an aliased legacy replay therefore
    // needs same-column stores followed by a deinterleave pass.
    const bool inplace_gate_up_grad =
        gate_up_output == grad_gate_up_output;
    const bool inplace_interleaved_gate_up_grad =
        inplace_gate_up_grad && !kBF16Mode && !kGateUpPrepared;
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
        kResidualMXFP8Dgrad ? 3 : 0;
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
    auto* primary_mma_barrier = weight_load_barrier + 1;
    auto* residual_mma_barrier = weight_load_barrier + 2;
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

    const uint32_t residual_sf_rows =
        num_acts_rows / BLOCK_M * SF_BLOCK_M;

    // Allocation-free W2 activation planes.  These are the same aliases used
    // by build_w2_residual_acts_once(); the pipelined path writes them directly
    // from the reverse-dispatch shared row and compiles that later pass out.
    auto* pipelined_w2_primary =
        reinterpret_cast<uint8_t*>(grad_x_pool_output);
    auto* pipelined_w2_residual =
        pipelined_w2_primary +
        static_cast<uint64_t>(num_acts_rows) * kHidden;
    constexpr uint64_t kPipelinedW13WeightAliasValues =
        static_cast<uint64_t>(kNumExperts) * kHidden *
        (2 * kIntermediateHidden);
    constexpr uint64_t kPipelinedW13WeightAliasBytes =
        kPipelinedW13WeightAliasValues +
        kPipelinedW13WeightAliasValues / kGranK;
    auto* pipelined_w2_primary_sf =
        reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(w13_dequant_scratch) +
            kPipelinedW13WeightAliasBytes);
    auto* pipelined_w2_residual_sf =
        pipelined_w2_primary_sf +
        static_cast<uint64_t>(residual_sf_rows) * (kHidden / 128);
    const uint64_t pipelined_w2_num_sf_words =
        static_cast<uint64_t>(residual_sf_rows) * (kHidden / 128);
    // Prepared gate/up compiles the L1 replay completely out, so its FP8
    // activation-scale ring is dead from kernel entry through W13 wgrad.  The
    // exact BF16 stream reuses its first one-word-per-pool-block prefix for
    // readiness without extending peak memory.  Keep the proven residual
    // specialization on its existing post-cache alias.
    auto* exact_bf16_grad_y_block_ready =
        const_cast<uint32_t*>(acts_sf_ptr);
    auto* residual_grad_y_block_ready =
        pipelined_w2_residual_sf + pipelined_w2_num_sf_words;
    auto* grad_y_block_ready =
        kExactBF16PipelinedGradYDispatch
        ? exact_bf16_grad_y_block_ready
        : residual_grad_y_block_ready;
    // Option-A keeps the first physical slot of fixed-top-k plane two as a
    // pointer-free control arena.  The generic A/B ring ABI shifts all value
    // descriptors by the same 192 rows, so this state can stay live while
    // rolling generations use the remaining slots.  Keeping the address
    // derivation before the dispatch lambda lets its striped source-X writers
    // publish readiness without capturing a later shared-memory context.
    constexpr uint32_t kExactRingW2PrefixCTAs = 132u;
    constexpr uint32_t kExactRingConsumerCTAs =
        kNumSMs - kExactRingW2PrefixCTAs;
    constexpr uint32_t kExactRingControlRows = 192u;
    constexpr uint32_t kExactRingGUReadyTarget =
        kIntermediateHidden / BLOCK_N;
    constexpr uint32_t kExactRingXReadyTarget = BLOCK_M;
    constexpr uint32_t kExactRingXReadyIncrement = 0x100u;
    DG_STATIC_ASSERT(
        !kK3MxFp8ExactEpilogueRing ||
            (kNumSMs == 148u && kExactRingW2PrefixCTAs == 132u &&
             kExactRingConsumerCTAs == 16u &&
             kExactRingConsumerCTAs % 2u == 0u &&
             kExactRingControlRows == BLOCK_M &&
             kExactRingGUReadyTarget == 24u &&
             kExactRingXReadyTarget == 192u),
        "Exact A/B ring role and packed-readiness geometry changed");
    const auto k3_mxfp8_exact_ring_control_state = [&]() -> uint32_t* {
        if constexpr (kK3MxFp8ExactEpilogueRing) {
            const uint64_t plane_bytes =
                static_cast<uint64_t>(
                    backward_workspace.num_max_tokens_per_rank) *
                kHidden * sizeof(cutlass::bfloat16_t);
            return reinterpret_cast<uint32_t*>(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(
                    backward_grad_y)) +
                kK3MxFp8EpilogueScratchPrimaryPlane * plane_bytes);
        }
        return nullptr;
    };
    const auto k3_mxfp8_exact_ring_source_ready = [&]() -> uint32_t* {
        if constexpr (kK3MxFp8ExactEpilogueRing) {
            return k3_mxfp8_exact_ring_control_state() +
                K3MxFp8OverlapState::kNumWords;
        }
        return nullptr;
    };
    // Pull, route, round and publish one complete physical pool row.
    // A row is the smallest useful unit for the existing 1-D remote TMA and it
    // keeps the exact native boundary:
    //   BF16(float(raw BF16 dY) * FP32 route weight) -> group-32 MXFP8.
    // Once all BLOCK_M physical rows (including zero padding) arrive, W2 may
    // consume that pool block without waiting for the remaining expert pool.
    const auto run_pipelined_grad_y_dispatch = [&]() {
      if constexpr (kPipelinedGradYDispatch) {
        if constexpr (kK3MxFp8ExactEpilogueRing) {
            // The last sixteen CTAs are complete cluster-2 exact consumers.
            // They never own a striped dispatch row; otherwise changing the
            // scheduler stride below would duplicate or omit physical rows.
            if (blockIdx.x >= kExactRingW2PrefixCTAs)
                return;
        }
        DG_STATIC_ASSERT(
            kHidden % 128 == 0,
            "Pipelined grad-y requires complete 128-column groups");
        const uint32_t dispatch_warp_idx =
            warp_idx - kDispatchWarpStart;
        auto* pull_buffer =
            reinterpret_cast<cd_dtype_t*>(smem_buffer) +
            dispatch_warp_idx * kHidden;
        auto* pull_mbarrier =
            dispatch_barriers[dispatch_warp_idx];
        uint32_t pool_block_offset = active_pool_block_begin;

        #pragma unroll 1
        for (uint32_t expert_idx = 0;
             expert_idx < kNumExperts; ++expert_idx) {
            const uint32_t num_tokens = static_cast<uint32_t>(
                __ldg(active_expert_counts + expert_idx));
            const uint32_t num_blocks =
                math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t num_padded_tokens =
                num_blocks * BLOCK_M;

            for (uint32_t token_idx =
                     blockIdx.x * kNumDispatchWarps +
                     dispatch_warp_idx;
                 token_idx < num_padded_tokens;
                 token_idx +=
                     (kK3MxFp8ExactEpilogueRing
                          ? kExactRingW2PrefixCTAs
                          : kNumSMs) *
                     kNumDispatchWarps) {
                const bool valid_row = token_idx < num_tokens;
                const uint32_t pool_block_idx =
                    pool_block_offset + token_idx / BLOCK_M;
                const uint32_t local_m = token_idx % BLOCK_M;
                const uint32_t pool_row =
                    pool_block_idx * BLOCK_M + local_m;
                layout::TokenSrcMetadata metadata{};
                float effective_route_weight = 0.0f;

                if (valid_row) {
                    metadata = token_src_metadata[pool_row];
                    const auto* remote_grad_y =
                        backward_sym_buffer.map(
                            backward_grad_y +
                                static_cast<uint64_t>(
                                    (active_token_begin +
                                     metadata.token_idx)) *
                                    kHidden,
                            metadata.rank_idx);
                    if (lane_idx == 0) {
                        ptx::tma_load_1d(
                            pull_buffer, remote_grad_y,
                            pull_mbarrier,
                            kHidden * sizeof(cd_dtype_t));
                    }
                    if constexpr (
                        kComputeRouteGrad && !kInputsPrepared &&
                        kRouteWeightMode == RouteWeightMode::PostDown) {
                        // The remote dY TMA provides enough independent
                        // latency to stage the saved down row in L2 before the
                        // exact route-adjoint reduction consumes it.
                        for (uint32_t col = lane_idx * 64;
                             col < kHidden; col += 32 * 64) {
                            cute::prefetch(
                                down_unweighted_output +
                                static_cast<uint64_t>(pool_row) * kHidden +
                                col);
                        }
                    }
                    if (lane_idx == 0) {
                        const auto* remote_weight =
                            backward_sym_buffer.map(
                                backward_topk_weights +
                                    static_cast<uint64_t>(
                                        (active_token_begin +
                                         metadata.token_idx)) *
                                        num_topk +
                                    metadata.topk_idx,
                                metadata.rank_idx);
                        const float route_weight = *remote_weight;
                        if (route_weights_fp32 != nullptr) {
                            route_weights_fp32[pool_row] =
                                route_weight;
                            effective_route_weight = route_weight;
                        } else {
                            const cd_dtype_t rounded_weight(route_weight);
                            route_weights[pool_row] = rounded_weight;
                            effective_route_weight =
                                static_cast<float>(rounded_weight);
                        }
                        ptx::mbarrier_arrive_and_set_tx(
                            pull_mbarrier,
                            kHidden * sizeof(cd_dtype_t));
                        ptx::mbarrier_wait_and_flip_phase(
                            pull_mbarrier,
                            dispatch_pull_mbarrier_phase);
                        // The pipelined path consumes raw dY directly from this
                        // shared row for the route dot and weighted activation.
                        // Do not materialize it in grad_y_unweighted_output:
                        // direct-dX training intentionally aliases that BF16
                        // pool with the two compact FP8 W2 activation planes.
                        // A BF16 row store would overwrite compact FP8 rows
                        // 2*r and 2*r+1 after they have been published.
                    }
                    effective_route_weight = __shfl_sync(
                        0xffffffff, effective_route_weight, 0);
                } else {
                    for (uint32_t col = lane_idx;
                         col < kHidden; col += 32)
                        pull_buffer[col] = cd_dtype_t(0.0f);
                }
                __syncwarp();

                // Preserve the existing early POST_DOWN route-adjoint order:
                // four independent lane accumulators, then the same fixed warp
                // shuffle tree.  The raw BF16 operand is still resident in the
                // dispatch row and cannot be clobbered by its MXFP8 alias.
                if constexpr (
                    kComputeRouteGrad && !kInputsPrepared &&
                    kRouteWeightMode == RouteWeightMode::PostDown) {
                  if (valid_row) {
                    using RouteDotSavedDownVector =
                        cutlass::AlignedArray<cd_dtype_t, 4, 8>;
                    static_assert(
                        sizeof(RouteDotSavedDownVector) == 8 &&
                            alignof(RouteDotSavedDownVector) == 8,
                        "POST_DOWN route dot requires one aligned BF16x4 load");
                    float lane_sums[4] = {
                        0.0f, 0.0f, 0.0f, 0.0f};
                    const auto* const saved_down_row =
                        down_unweighted_output +
                        static_cast<uint64_t>(pool_row) * kHidden;
                    for (uint32_t col_base = lane_idx * 4;
                         col_base < kHidden;
                         col_base += 32 * 4) {
                        // The forward pool is allocation-aligned, every K3 row
                        // is a multiple of eight bytes, and col_base advances
                        // by four BF16 values.  Load the four saved-down
                        // operands once without changing their FP32 use order.
                        const auto saved_down =
                            *reinterpret_cast<
                                const RouteDotSavedDownVector*>(
                                saved_down_row + col_base);
                        #pragma unroll
                        for (uint32_t i = 0; i < 4; ++i) {
                            const uint32_t col = col_base + i;
                            lane_sums[i] = __fadd_rn(
                                lane_sums[i],
                                __fmul_rn(
                                    static_cast<float>(
                                        pull_buffer[col]),
                                    static_cast<float>(
                                        saved_down[i])));
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
                                0xffffffff, grad_route, offset));
                    }
                    if (lane_idx == 0) {
                        grad_route_output[pool_row] = grad_route;
                        if (backward_grad_route != nullptr) {
                            auto* remote_grad_route =
                                backward_sym_buffer.map(
                                    backward_grad_route +
                                        static_cast<uint64_t>(
                                            (active_token_begin +
                                             metadata.token_idx)) *
                                            num_topk +
                                        metadata.topk_idx,
                                    metadata.rank_idx);
                            *remote_grad_route = grad_route;
                        }
                    }
                  }
                }

                if constexpr (kExactBF16PipelinedGradYDispatch) {
                    // dW2 consumes the unweighted BF16 dY row.  Preserve the
                    // legacy TMA materialization exactly, including physical
                    // padding rows, before publishing the separately rounded
                    // route-weighted BF16 operand used by W2 dgrad.
                    // Invalid rows were zeroed through the generic shared
                    // proxy, so publish that view before TMA store reads the
                    // same pull buffer through the async proxy.
                    cutlass::arch::fence_view_async_shared();
                    if (lane_idx == 0) {
                        ptx::tma_store_1d(
                            grad_y_unweighted_output +
                                static_cast<uint64_t>(pool_row) *
                                    kHidden,
                            pull_buffer,
                            kHidden * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                    }
                    __syncwarp();
                }

                if constexpr (kResidualMXFP8Dgrad) {
                constexpr uint32_t kValuesPerLane = 4;
                const uint32_t group_idx = lane_idx / 8;
                const uint32_t lane_in_group = lane_idx % 8;
                #pragma unroll 1
                for (uint32_t k_block_idx = 0;
                     k_block_idx < kHidden / 128;
                     ++k_block_idx) {
                    const uint32_t global_k =
                        k_block_idx * 128 + group_idx * 32 +
                        lane_in_group * kValuesPerLane;
                    const uint64_t q_offset =
                        static_cast<uint64_t>(pool_row) * kHidden +
                        global_k;
                    const uint64_t residual_q_offset =
                        (static_cast<uint64_t>(active_num_acts_rows) +
                         pool_row) *
                            kHidden +
                        global_k;
                    const uint32_t sf_m =
                        pool_block_idx * SF_BLOCK_M +
                        (local_m & ~127u) +
                        (local_m & 31u) * 4 +
                        ((local_m >> 5) & 3u);
                    const uint64_t sf_offset = sf_m +
                        static_cast<uint64_t>(k_block_idx) *
                            residual_sf_rows;

                    if (!valid_row) {
                        *reinterpret_cast<uint32_t*>(
                            pipelined_w2_primary + q_offset) = 0;
                        *reinterpret_cast<uint32_t*>(
                            pipelined_w2_primary + residual_q_offset) = 0;
                        #pragma unroll
                        for (uint32_t i = 0;
                             i < kValuesPerLane; ++i) {
                            grad_ye_output[
                                static_cast<uint64_t>(pool_row) *
                                    kHidden +
                                global_k + i] = cd_dtype_t(0.0f);
                        }
                        if (lane_idx == 0) {
                            pipelined_w2_primary_sf[sf_offset] =
                                0x7f7f7f7fu;
                            pipelined_w2_residual_sf[sf_offset] =
                                0x7f7f7f7fu;
                        }
                        continue;
                    }

                    float values[kValuesPerLane];
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kValuesPerLane; ++i) {
                        const cd_dtype_t weighted(
                            static_cast<float>(
                                pull_buffer[global_k + i]) *
                            effective_route_weight);
                        grad_ye_output[
                            static_cast<uint64_t>(pool_row) *
                                kHidden +
                            global_k + i] = weighted;
                        values[i] = static_cast<float>(weighted);
                    }

                    float primary_amax = 0.0f;
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kValuesPerLane; ++i)
                        primary_amax = cute::max(
                            primary_amax, cute::abs(values[i]));
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
                        primary_sf_pair, primary_sf_inv_pair);
                    const auto primary_q = __nv_fp8x4_e4m3(
                        make_float4(
                            values[0] * primary_sf_inv_pair.x,
                            values[1] * primary_sf_inv_pair.x,
                            values[2] * primary_sf_inv_pair.x,
                            values[3] * primary_sf_inv_pair.x));
                    const float4 primary_f =
                        static_cast<float4>(primary_q);
                    // Keep the correction components scalar.  In this
                    // communication-heavy producer the array form spills and
                    // its packed FP8 bytes can be reloaded from clobbered local
                    // slots even though the primary bytes and both scales are
                    // correct.
                    const float residual_0 = values[0] -
                        primary_f.x * primary_sf_pair.x;
                    const float residual_1 = values[1] -
                        primary_f.y * primary_sf_pair.x;
                    const float residual_2 = values[2] -
                        primary_f.z * primary_sf_pair.x;
                    const float residual_3 = values[3] -
                        primary_f.w * primary_sf_pair.x;
                    float residual_amax = cute::max(
                        cute::max(
                            cute::abs(residual_0),
                            cute::abs(residual_1)),
                        cute::max(
                            cute::abs(residual_2),
                            cute::abs(residual_3)));
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
                        residual_sf_pair, residual_sf_inv_pair);
                    const float residual_scaled_0 = residual_0 *
                        residual_sf_inv_pair.x;
                    const float residual_scaled_1 = residual_1 *
                        residual_sf_inv_pair.x;
                    const float residual_scaled_2 = residual_2 *
                        residual_sf_inv_pair.x;
                    const float residual_scaled_3 = residual_3 *
                        residual_sf_inv_pair.x;
                    uint16_t residual_pair_01;
                    uint16_t residual_pair_23;
                    asm volatile(
                        "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
                        : "=h"(residual_pair_01)
                        : "f"(residual_scaled_0),
                          "f"(residual_scaled_1));
                    asm volatile(
                        "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
                        : "=h"(residual_pair_23)
                        : "f"(residual_scaled_2),
                          "f"(residual_scaled_3));
                    const uint32_t residual_q_bits =
                        static_cast<uint32_t>(residual_pair_01) |
                        (static_cast<uint32_t>(residual_pair_23) << 16);
                    *reinterpret_cast<uint32_t*>(
                        pipelined_w2_primary + q_offset) =
                        primary_q.__x;
                    auto* residual_q_ptr =
                        reinterpret_cast<uint32_t*>(
                            pipelined_w2_primary + residual_q_offset);
                    asm volatile(
                        "st.global.u32 [%0], %1;"
                        :: "l"(residual_q_ptr), "r"(residual_q_bits)
                        : "memory");
                    const uint32_t primary_scale_byte =
                        (*reinterpret_cast<const uint32_t*>(
                             &primary_sf_pair.x)) >> 23;
                    const uint32_t residual_scale_byte =
                        (*reinterpret_cast<const uint32_t*>(
                             &residual_sf_pair.x)) >> 23;
                    uint32_t primary_scale_word = 0;
                    uint32_t residual_scale_word = 0;
                    #pragma unroll
                    for (uint32_t group = 0; group < 4;
                         ++group) {
                        primary_scale_word |= __shfl_sync(
                            0xffffffff, primary_scale_byte,
                            group * 8) << (group * 8);
                        residual_scale_word |= __shfl_sync(
                            0xffffffff, residual_scale_byte,
                            group * 8) << (group * 8);
                    }
                    if (lane_idx == 0) {
                        pipelined_w2_primary_sf[sf_offset] =
                            primary_scale_word;
                        pipelined_w2_residual_sf[sf_offset] =
                            residual_scale_word;
                    }
                }

                // SF_BLOCK_M is 256 for K3 while each physical pool block has
                // 192 rows.  The legacy whole-pool builder initializes the 64
                // layout-only scale rows to the zero-scale encoding before it
                // publishes the grid.  Make row zero own the same disjoint
                // padding here so block readiness covers the complete TMA tile,
                // not only the physical value rows.
                if (local_m == 0) {
                    for (uint32_t padding_m = BLOCK_M + lane_idx;
                         padding_m < SF_BLOCK_M;
                         padding_m += 32) {
                        const uint32_t padding_sf_m =
                            pool_block_idx * SF_BLOCK_M +
                            (padding_m & ~127u) +
                            (padding_m & 31u) * 4 +
                            ((padding_m >> 5) & 3u);
                        #pragma unroll 1
                        for (uint32_t k_block_idx = 0;
                             k_block_idx < kHidden / 128;
                             ++k_block_idx) {
                            const uint64_t padding_sf_offset =
                                padding_sf_m +
                                static_cast<uint64_t>(k_block_idx) *
                                    residual_sf_rows;
                            pipelined_w2_primary_sf[
                                padding_sf_offset] = 0x7f7f7f7fu;
                            pipelined_w2_residual_sf[
                                padding_sf_offset] = 0x7f7f7f7fu;
                        }
                    }
                }
                } else {
                    // Native MXFP4's W2 operand is exactly one BF16 rounding
                    // of FP32(raw BF16 dY * FP32 route weight).  Publish that
                    // canonical row directly; no FP8 quantization, residual
                    // plane, or additional BF16 round trip is introduced.
                    for (uint32_t col = lane_idx;
                         col < kHidden; col += 32) {
                        grad_ye_output[
                            static_cast<uint64_t>(pool_row) *
                                kHidden +
                            col] = valid_row
                            ? cd_dtype_t(
                                  static_cast<float>(
                                      pull_buffer[col]) *
                                  effective_route_weight)
                            : cd_dtype_t(0.0f);
                    }
                }

                // The BF16 value row (and, for the residual specialization,
                // the value/scale planes) is written by every lane.
                // Proxy/global publication is per thread, so lane zero cannot
                // fence the other lanes' stores on their behalf. Make every
                // writer publish its own generic stores before lane zero
                // releases block readiness.
                __syncwarp();
                asm volatile(
                    "fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncwarp();
                if (lane_idx == 0)
                    ptx::atomic_add_rel(
                        grad_y_block_ready + pool_block_idx,
                        1u);
                __syncwarp();

                if constexpr (
                    kExactSourceX &&
                    !kOverlapExactSourceXPlaneOneWithW13) {
                  const bool publish_source_x =
                      kExactBF16PipelinedGradYDispatch ||
                      (valid_row && !late_exact_source_x);
                  if (publish_source_x) {
                    if (valid_row && lane_idx == 0) {
                        const auto* remote_x =
                            backward_sym_buffer.map(
                                backward_x +
                                    static_cast<uint64_t>(
                                        (active_token_begin +
                                         metadata.token_idx)) *
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
                    }
                    __syncwarp();
                    if (lane_idx == 0) {
                        // Exact native W13 wgrad rounds K over physical pool
                        // blocks. Invalid rows therefore need the same
                        // explicit zero source-X row as the serial path.
                        ptx::tma_store_1d(
                            x_pool_output +
                                static_cast<uint64_t>(pool_row) *
                                    kHidden,
                            pull_buffer,
                            kHidden * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                        ptx::tma_store_wait<0>();
                        if constexpr (kK3MxFp8ExactEpilogueRing) {
                            // The TMA wait retires the async write.  Bridge it
                            // back to generic visibility before the release
                            // atomic advertises this canonical BF16 row to a
                            // background A/B quantization engine.
                            asm volatile(
                                "fence.proxy.async.global;"
                                ::: "memory");
                            // Acq-rel is required on every contributor: the
                            // producer acquires only the final packed value,
                            // so each RMW must carry all preceding source-row
                            // writes forward through one release sequence.
                            const uint32_t previous = ptx::atomic_add_acq_rel(
                                k3_mxfp8_exact_ring_source_ready() +
                                    pool_block_idx,
                                kExactRingXReadyIncrement);
                            DG_DEVICE_ASSERT(
                                ((previous >> 8u) & 0xffu) <
                                    kExactRingXReadyTarget &&
                                (previous & 0xffff0000u) == 0u);
                        }
                    }
                    __syncwarp();
                  }
                }
            }
            pool_block_offset += num_blocks;
        }
      }
    };

    // Warp three is idle throughout this exact W13 specialization. Reuse its
    // first dispatch row and the untouched second dispatch-barrier epoch to
    // pull source X without removing any W13 epilogue threads. Once all 148
    // source warps on every rank retire their remote reads, the same warps
    // clear combine plane one and execute a second rank-wide publication edge.
    // W13 writes for that plane are staged elsewhere until this function and
    // the W13 roles join below.
    const auto run_w13_overlapped_source_x_and_reclaim_plane_one = [&]() {
      if constexpr (kOverlapExactSourceXPlaneOneWithW13) {
        constexpr uint32_t kExactSourceXWarp = 3u;
        if (warp_idx != kExactSourceXWarp)
            return;

        DG_DEVICE_ASSERT(
            active_token_begin == 0u && num_topk > 1u &&
            backward_x ==
                backward_grad_y +
                    static_cast<uint64_t>(
                        backward_workspace.num_max_tokens_per_rank) *
                        kHidden &&
            grad_y_unweighted_output != x_pool_output &&
            grad_y_unweighted_output != grad_ye_output &&
            grad_y_unweighted_output != h_weighted_output);

        k3_exact_source_x_pull_and_reclaim_plane_one<
            kHidden, kNumExperts, BLOCK_M, kNumSMs, kNumRanks>(
                active_expert_counts, token_src_metadata,
                backward_x, x_pool_output,
                backward_sym_buffer, backward_workspace,
                active_pool_block_begin,
                reinterpret_cast<cd_dtype_t*>(smem_buffer),
                dispatch_barriers[kNumDispatchWarps]);
      }
    };

    // Turn the completed reverse-dispatch warpgroup into a nonblocking W2
    // weight producer. The builder reuses only the dispatch prefix of shared
    // memory; the W2 UMMA/TMA pipeline starts at smem_gemm_base and remains
    // live concurrently. One CTA owns a complete (expert, N) group and walks
    // K in consumer order. Compared with scattering those K tiles over 28
    // CTAs, this publishes 148 useful first-wave groups and removes the
    // slowest-of-28 producer dependency from each W2 consumer.
    const auto run_dispatch_w2_residual_weight_cache = [&]() {
      if constexpr (kSplitResidualWeightCache) {
        constexpr uint32_t kBuilderWarps =
            kNumDispatchW2WeightBuilderWarps;
        constexpr uint32_t kBuilderThreads =
            kNumDispatchW2WeightBuilderThreads;
        constexpr uint32_t kBuilderWeightBytes =
            DGRAD_BLOCK_K * (LOAD_BLOCK_N / 2) * sizeof(int8_t);
        constexpr uint32_t kBuilderScaleBytes =
            DGRAD_BLOCK_K * (LOAD_BLOCK_N / kGranK) * sizeof(float);
        constexpr uint32_t kBuilderOutputBytes =
            DGRAD_BLOCK_K * LOAD_BLOCK_N *
            sizeof(residual_dgrad_dtype_t);
        constexpr uint32_t kBuilderOutputScaleBytes =
            LOAD_BLOCK_N * sizeof(uint32_t);
        constexpr uint32_t kBuilderScratchBytes =
            kBuilderWeightBytes + kBuilderScaleBytes +
            kBuilderOutputBytes + kBuilderOutputScaleBytes;
        static_assert(
            kBuilderScratchBytes + sizeof(Barrier) <=
                SMEM_DISPATCH_SIZE,
            "Dispatch-private shared memory is too small for one converter tile");
        static_assert(
            kBuilderScratchBytes % alignof(Barrier) == 0,
            "Dispatch builder barrier must remain naturally aligned");
        static_assert(
            kBuilderWeightBytes % 128 == 0 &&
            (kBuilderWeightBytes + kBuilderScaleBytes) % 128 == 0 &&
            (kBuilderWeightBytes + kBuilderScaleBytes +
             kBuilderOutputBytes) % 128 == 0,
            "Every dispatch builder TMA operand must stay 128-byte aligned");

        const uint32_t builder_warp_idx =
            warp_idx - kDispatchWarpStart;
        const uint32_t builder_thread_idx =
            builder_warp_idx * 32 + lane_idx;
        auto* builder_weight_source =
            reinterpret_cast<int8_t*>(smem_buffer);
        auto* builder_scale_source = reinterpret_cast<float*>(
            smem_buffer + kBuilderWeightBytes);
        auto* builder_output =
            reinterpret_cast<uint8_t*>(
                smem_buffer + kBuilderWeightBytes +
                kBuilderScaleBytes);
        auto* builder_output_scales =
            reinterpret_cast<uint32_t*>(
                smem_buffer + kBuilderWeightBytes +
                kBuilderScaleBytes + kBuilderOutputBytes);
        // The dispatch pull barriers remain live until early dW2 can reuse
        // them for late exact-X. Place this producer-only barrier after the
        // packed tile instead of changing any dispatch-barrier lifetime.
        auto* builder_load_barrier = reinterpret_cast<Barrier*>(
            smem_buffer + kBuilderScratchBytes);

        // Warps 4..7 have drained their private pull buffers before entering;
        // borrowed warps 8..11 wait here without touching the dispatch prefix.
        // Start a distinct barrier lifetime only after all eight builder warps
        // join, so no W2 role can still reference a dispatch-prefix byte.
        cutlass::arch::NamedBarrier::sync(
            kBuilderThreads, kDispatchWeightBuilderBarrier);
        if (builder_warp_idx == 0 && cute::elect_one_sync()) {
            builder_load_barrier->init(1);
            cutlass::arch::fence_barrier_init();
        }
        cutlass::arch::NamedBarrier::sync(
            kBuilderThreads, kDispatchWeightBuilderBarrier);

        uint32_t builder_load_phase = 0;
        const auto transform_w2 = [&]() {
            constexpr uint32_t output_mn = kIntermediateHidden;
            constexpr uint32_t output_k = kHidden;
            const uint32_t num_n_blocks = output_mn / LOAD_BLOCK_N;
            const uint32_t num_k_blocks = output_k / DGRAD_BLOCK_K;
            const uint64_t num_groups =
                static_cast<uint64_t>(kNumExperts) *
                num_n_blocks;

            // Consumers hold (expert, N) fixed and walk K in their inner
            // pipeline. Keep all K tiles of that group on one CTA and publish
            // them in exactly that order. Empty experts are skipped by all
            // roles and therefore require no epoch.
            #pragma unroll 1
            for (uint64_t group_idx = blockIdx.x;
                 group_idx < num_groups;
                 group_idx += kNumSMs) {
                const uint32_t n_block_idx =
                    group_idx % num_n_blocks;
                const uint32_t expert_idx =
                    group_idx / num_n_blocks;
                if (__ldg(expert_counts + expert_idx) == 0)
                    continue;

                // Prime K=0 before entering the consumer-ordered walk. The
                // one-stage input buffer is dead as soon as all converter
                // warps reach the post-conversion named barrier, while the
                // disjoint output buffer remains owned by the TMA store. That
                // lets the control lane issue load(K+1) after store(K) and
                // overlap both engines without another shared-memory stage.
                if (builder_warp_idx == 0 &&
                    cute::elect_one_sync()) {
                    tma::copy<
                        LOAD_BLOCK_N / 2, DGRAD_BLOCK_K, 0,
                        int8_t>(
                        &tensor_map_w2_weights,
                        builder_load_barrier,
                        builder_weight_source,
                        n_block_idx * (LOAD_BLOCK_N / 2),
                        expert_idx * output_k);
                    tma::copy<
                        LOAD_BLOCK_N / kGranK,
                        DGRAD_BLOCK_K, 0, float>(
                        &tensor_map_w2_scales,
                        builder_load_barrier,
                        builder_scale_source,
                        n_block_idx * (LOAD_BLOCK_N / kGranK),
                        expert_idx * output_k);
                    builder_load_barrier->arrive_and_expect_tx(
                        kBuilderWeightBytes +
                        kBuilderScaleBytes);
                }
                #pragma unroll 1
                for (uint32_t k_block_idx = 0;
                     k_block_idx < num_k_blocks;
                     ++k_block_idx) {
                    const uint64_t state_idx =
                        (static_cast<uint64_t>(expert_idx) *
                             num_k_blocks +
                         k_block_idx) *
                            num_n_blocks +
                        n_block_idx;

                    builder_load_barrier->wait(builder_load_phase);

                    k3_mxfp4_to_mxfp8_transposed_tile_one_pass<
                        kBuilderWarps>(
                        builder_weight_source,
                        builder_scale_source,
                        builder_output,
                        builder_output_scales,
                        n_block_idx,
                        builder_thread_idx);
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::sync(
                        kBuilderThreads,
                        kDispatchWeightBuilderBarrier);

                    if (builder_warp_idx == 0 &&
                        cute::elect_one_sync()) {
                        // These are ordinary one-CTA TMA stores. The later
                        // dgrad load is paired-CTA, but both peers acquire the
                        // same completed global tile before issuing it.
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_w2_dgrad_weights,
                            builder_output,
                            k_block_idx * DGRAD_BLOCK_K,
                            expert_idx * output_mn +
                                n_block_idx * LOAD_BLOCK_N);
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_w2_dgrad_weights_sf,
                            builder_output_scales,
                            n_block_idx * LOAD_BLOCK_N,
                            expert_idx *
                                    (output_k / (kGranK * 4)) +
                                k_block_idx);
                        cute::tma_store_arrive();

                        // The converter input and output occupy disjoint
                        // shared-memory ranges. Start the next packed-weight
                        // load before draining the current output stores; the
                        // second named barrier below still prevents converter
                        // reuse until both the store wait and release publish
                        // have completed.
                        if (k_block_idx + 1 < num_k_blocks) {
                            const uint32_t next_k_block_idx =
                                k_block_idx + 1;
                            tma::copy<
                                LOAD_BLOCK_N / 2, DGRAD_BLOCK_K,
                                0, int8_t>(
                                &tensor_map_w2_weights,
                                builder_load_barrier,
                                builder_weight_source,
                                n_block_idx *
                                    (LOAD_BLOCK_N / 2),
                                expert_idx * output_k +
                                    next_k_block_idx *
                                        DGRAD_BLOCK_K);
                            tma::copy<
                                LOAD_BLOCK_N / kGranK,
                                DGRAD_BLOCK_K, 0, float>(
                                &tensor_map_w2_scales,
                                builder_load_barrier,
                                builder_scale_source,
                                n_block_idx *
                                    (LOAD_BLOCK_N / kGranK),
                                expert_idx * output_k +
                                    next_k_block_idx *
                                        DGRAD_BLOCK_K);
                            builder_load_barrier
                                ->arrive_and_expect_tx(
                                    kBuilderWeightBytes +
                                    kBuilderScaleBytes);
                        }
                        ptx::tma_store_wait<0>();
                        asm volatile(
                            "fence.proxy.async.global;" :::
                            "memory");
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(weight_tile_states + state_idx),
                               "r"(launch_epoch)
                            : "memory");
                    }
                    cutlass::arch::NamedBarrier::sync(
                        kBuilderThreads,
                        kDispatchWeightBuilderBarrier);
                    builder_load_phase ^= 1;
                }
            }
        };

        transform_w2();
        cutlass::arch::NamedBarrier::sync(
            kBuilderThreads, kDispatchWeightBuilderBarrier);
        if (builder_warp_idx == 0 && cute::elect_one_sync()) {
            Barrier::invalidate(
                reinterpret_cast<Barrier::ValueType const*>(
                    builder_load_barrier));
        }
        cutlass::arch::NamedBarrier::sync(
            kBuilderThreads, kDispatchWeightBuilderBarrier);
      }
    };

    // dW2 retires the W2 parent pipeline and its shared-memory/barrier
    // lifetimes. Reuse that phase boundary to publish W13 with the exact
    // packed v376 converter, instead of putting W13 conversion on W2's
    // critical path. Sixteen high-register epilogue warps each own one of the
    // converter's 16 independent (group, stripe) tasks; warp zero only issues
    // TMA. The compact output aliases the future BF16 dW13 destination and is
    // consumed before dW13 overwrites it, so this adds no allocation.
    const auto run_post_dw2_w13_residual_weight_cache = [&]() {
      if constexpr (kSplitResidualWeightCache) {
        constexpr uint32_t kBuilderWarps = 16;
        constexpr uint32_t kFirstBuilderWarp = 4;
        constexpr uint32_t kBuilderWeightBytes =
            DGRAD_BLOCK_K * (LOAD_BLOCK_N / 2) * sizeof(int8_t);
        constexpr uint32_t kBuilderScaleBytes =
            DGRAD_BLOCK_K * (LOAD_BLOCK_N / kGranK) * sizeof(float);
        constexpr uint32_t kBuilderOutputBytes =
            DGRAD_BLOCK_K * LOAD_BLOCK_N *
            sizeof(residual_dgrad_dtype_t);
        constexpr uint32_t kBuilderOutputScaleBytes =
            LOAD_BLOCK_N * sizeof(uint32_t);
        constexpr uint32_t kBuilderScratchBytes =
            kBuilderWeightBytes + kBuilderScaleBytes +
            kBuilderOutputBytes + kBuilderOutputScaleBytes;
        static_assert(
            kFirstBuilderWarp + kBuilderWarps <= kNumThreads / 32,
            "W13 packed builder exceeds the resident warp set");
        static_assert(
            kBuilderScratchBytes <= SMEM_DISPATCH_SIZE,
            "Dispatch-prefix shared memory is too small for W13 builder");
        static_assert(
            kBuilderWeightBytes % 128 == 0 &&
            (kBuilderWeightBytes + kBuilderScaleBytes) % 128 == 0 &&
            (kBuilderWeightBytes + kBuilderScaleBytes +
             kBuilderOutputBytes) % 128 == 0,
            "Every W13 builder TMA operand must stay 128-byte aligned");

        auto* builder_weight_source =
            reinterpret_cast<int8_t*>(smem_buffer);
        auto* builder_scale_source = reinterpret_cast<float*>(
            smem_buffer + kBuilderWeightBytes);
        auto* builder_output = reinterpret_cast<uint8_t*>(
            smem_buffer + kBuilderWeightBytes + kBuilderScaleBytes);
        auto* builder_output_scales = reinterpret_cast<uint32_t*>(
            smem_buffer + kBuilderWeightBytes + kBuilderScaleBytes +
            kBuilderOutputBytes);

        // early dW2 invalidated the parent object before aliasing all shared
        // memory into the grouped BF16 body. Begin and end a fresh, explicit
        // weight-load-barrier lifetime without touching dispatch barriers,
        // which late exact-X may still consume.
        if constexpr (!kCoSchedulePostDW2W13Builders) {
            __syncthreads();
            if (warp_idx == 0 && cute::elect_one_sync()) {
                weight_load_barrier->init(1);
                cutlass::arch::fence_barrier_init();
            }
            __syncthreads();
        }

        constexpr uint32_t output_mn = kHidden;
        constexpr uint32_t output_k = 2 * kIntermediateHidden;
        constexpr uint32_t num_n_blocks = output_mn / LOAD_BLOCK_N;
        constexpr uint32_t num_k_blocks = output_k / DGRAD_BLOCK_K;
        constexpr uint64_t num_groups =
            static_cast<uint64_t>(kNumExperts) * num_n_blocks;
        constexpr uint32_t w13_ready_epoch_mask = 0x80000000u;
        const uint32_t ready_epoch =
            launch_epoch ^ w13_ready_epoch_mask;
        uint32_t builder_load_phase = 0;
        constexpr uint32_t kBuilderParticipantThreads =
            (kBuilderWarps + 1) * 32;
        const bool is_converter_warp =
            warp_idx >= kFirstBuilderWarp &&
            warp_idx < kFirstBuilderWarp + kBuilderWarps;
        const bool is_builder_warp =
            warp_idx == 0 || is_converter_warp;

        // Match W13's (expert, N, K-inner) consumer order and keep each group
        // on one CTA. The state index remains the established K-major storage
        // order expected by the dgrad producer.
        const uint32_t first_unpublished_group =
            kBackgroundW13WeightCache
            ? ptx::ld_acq(
                  w13_background_group_cursors + blockIdx.x)
            : 0;
        if (is_builder_warp) {
          #pragma unroll 1
          for (uint64_t group_idx =
                   blockIdx.x +
                   static_cast<uint64_t>(first_unpublished_group) *
                       kNumSMs;
               group_idx < num_groups;
               group_idx += kNumSMs) {
            const uint32_t n_block_idx = group_idx % num_n_blocks;
            const uint32_t expert_idx = group_idx / num_n_blocks;
            if (__ldg(expert_counts + expert_idx) == 0)
                continue;

            // Prime K=0 before the consumer-ordered walk. The converter input
            // and output occupy disjoint shared-memory ranges, so once all
            // converter warps reach the post-conversion named barrier the
            // control lane may refill the input for K+1 while the current
            // output is still draining through TMA stores.
            if (warp_idx == 0 && cute::elect_one_sync()) {
                tma::copy<
                    LOAD_BLOCK_N / 2, DGRAD_BLOCK_K, 0, int8_t>(
                    &tensor_map_w13_weights,
                    weight_load_barrier,
                    builder_weight_source,
                    n_block_idx * (LOAD_BLOCK_N / 2),
                    expert_idx * output_k);
                tma::copy<
                    LOAD_BLOCK_N / kGranK,
                    DGRAD_BLOCK_K, 0, float>(
                    &tensor_map_w13_scales,
                    weight_load_barrier,
                    builder_scale_source,
                    n_block_idx * (LOAD_BLOCK_N / kGranK),
                    expert_idx * output_k);
                weight_load_barrier->arrive_and_expect_tx(
                    kBuilderWeightBytes + kBuilderScaleBytes);
            }

            #pragma unroll 1
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks;
                 ++k_block_idx) {
                const uint64_t state_idx =
                    (static_cast<uint64_t>(expert_idx) *
                         num_k_blocks +
                     k_block_idx) *
                        num_n_blocks +
                    n_block_idx;
                if (is_converter_warp) {
                    weight_load_barrier->wait(builder_load_phase);
                    k3_mxfp4_to_mxfp8_transposed_tile_one_pass<
                        kBuilderWarps>(
                        builder_weight_source,
                        builder_scale_source,
                        builder_output,
                        builder_output_scales,
                        n_block_idx,
                        (warp_idx - kFirstBuilderWarp) * 32 +
                            lane_idx);
                    cutlass::arch::fence_view_async_shared();
                }
                cutlass::arch::NamedBarrier::sync(
                    kBuilderParticipantThreads,
                    kPostDW2WeightBuilderBarrier);

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_w13_dgrad_weights,
                        builder_output,
                        k_block_idx * DGRAD_BLOCK_K,
                        expert_idx * output_mn +
                            n_block_idx * LOAD_BLOCK_N);
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_w13_dgrad_weights_sf,
                        builder_output_scales,
                        n_block_idx * LOAD_BLOCK_N,
                        expert_idx *
                                (output_k / (kGranK * 4)) +
                            k_block_idx);
                    cute::tma_store_arrive();

                    // The participant barrier above proves that converter
                    // warps no longer reference the input tile. Refill that
                    // disjoint range before waiting for the current output
                    // stores, so W13 weight loads and stores progress
                    // concurrently.
                    if (k_block_idx + 1 < num_k_blocks) {
                        const uint32_t next_k_block_idx =
                            k_block_idx + 1;
                        tma::copy<
                            LOAD_BLOCK_N / 2, DGRAD_BLOCK_K, 0,
                            int8_t>(
                            &tensor_map_w13_weights,
                            weight_load_barrier,
                            builder_weight_source,
                            n_block_idx * (LOAD_BLOCK_N / 2),
                            expert_idx * output_k +
                                next_k_block_idx * DGRAD_BLOCK_K);
                        tma::copy<
                            LOAD_BLOCK_N / kGranK,
                            DGRAD_BLOCK_K, 0, float>(
                            &tensor_map_w13_scales,
                            weight_load_barrier,
                            builder_scale_source,
                            n_block_idx *
                                (LOAD_BLOCK_N / kGranK),
                            expert_idx * output_k +
                                next_k_block_idx * DGRAD_BLOCK_K);
                        weight_load_barrier->arrive_and_expect_tx(
                            kBuilderWeightBytes +
                            kBuilderScaleBytes);
                    }
                    ptx::tma_store_wait<0>();
                    asm volatile(
                        "fence.proxy.async.global;" ::: "memory");
                    asm volatile(
                        "st.release.gpu.global.u32 [%0], %1;"
                        :: "l"(weight_tile_states +
                               kNumW2WeightTileStates + state_idx),
                           "r"(ready_epoch)
                        : "memory");
                }
                cutlass::arch::NamedBarrier::sync(
                    kBuilderParticipantThreads,
                    kPostDW2WeightBuilderBarrier);
                builder_load_phase ^= 1;
            }
          }
        }

        if constexpr (!kCoSchedulePostDW2W13Builders) {
            __syncthreads();
            if (warp_idx == 0 && cute::elect_one_sync()) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        weight_load_barrier));
            }
            __syncthreads();
        }
      }
    };

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
    constexpr uint32_t kEarlyDW2WgradStages = 5;
    constexpr uint32_t kWgradRoleThreads = 128;
    constexpr uint32_t kWgradTensorUtil = 100;
    constexpr uint32_t kReadyDW2TasksPerExpert =
        math::constexpr_ceil_div(kHidden, kWgradBlockM) / 2u *
        math::constexpr_ceil_div(kIntermediateHidden, kWgradBlockN);
    constexpr uint32_t kReadyDW13TasksPerExpert =
        math::constexpr_ceil_div(
            2u * kIntermediateHidden, kWgradBlockM) / 2u *
        math::constexpr_ceil_div(kHidden, kWgradBlockN);
    // The architecture-valid paired-N tail retains cta_group::2 and folds two
    // adjacent 256-wide N tiles into one scheduler task / two FP32 TMEM halves.
    constexpr uint32_t kPairedDW2TasksPerExpert =
        kReadyDW2TasksPerExpert / 2u;
    constexpr uint32_t kPairedDW13TasksPerExpert =
        kReadyDW13TasksPerExpert / 2u;
    DG_STATIC_ASSERT(
        kReadyDW2TasksPerExpert % 2u == 0u &&
            kReadyDW13TasksPerExpert % 2u == 0u &&
            kPairedDW2TasksPerExpert == 84u &&
            kPairedDW13TasksPerExpert == 168u,
        "K3 paired-N terminal task geometry changed");
    // The retained BF16 body aliases the complete parent shared allocation.
    // Its A/B/CD data intentionally overwrite parent control objects only
    // after every live parent mbarrier is invalidated. Its own mbarrier/TMEM
    // control starts above the entire parent control range, so a retained
    // resource lifetime cannot alias a still-addressable parent object.
    constexpr uint32_t kReadyBf16SmemDataBytes =
        2u * kWgradBlockM *
            (kWgradSwizzle / sizeof(cd_dtype_t)) *
            sizeof(cd_dtype_t) +
        kWgradStages * 2u * kWgradBlockM *
            kWgradBlockK * sizeof(cd_dtype_t);
    // The hybrid dW2 uses five BF16 stages.  Its body control objects end
    // before one 32x128 dW13 quantization engine, all inside the unchanged
    // 230400-byte launch allocation.  The background engine therefore adds
    // no persistent or dynamic memory and cannot alias a live BF16 mbarrier.
    constexpr uint32_t kHybridDW2Bf16SmemDataBytes =
        2u * kWgradBlockM *
            (kWgradSwizzle / sizeof(cd_dtype_t)) *
            sizeof(cd_dtype_t) +
        kEarlyDW2WgradStages * 2u * kWgradBlockM *
            kWgradBlockK * sizeof(cd_dtype_t);
    constexpr uint32_t kHybridDW2Bf16ControlBytes =
        (3u * kEarlyDW2WgradStages + 2u * 2u + 1u) *
            sizeof(Barrier) +
        sizeof(uint32_t);
    constexpr uint32_t kHybridDW13QuantScratchBegin =
        math::constexpr_align(
            kHybridDW2Bf16SmemDataBytes +
                kHybridDW2Bf16ControlBytes,
            128u);
    constexpr uint32_t kHybridDW13QuantScratchEnd =
        kHybridDW13QuantScratchBegin +
        kK3MxFp8DW13QuantEngineStride;
    constexpr uint32_t kReadyParentBarrierOffset =
        SMEM_DISPATCH_SIZE + SMEM_CD_SIZE +
        kNumStages *
            (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
             SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE +
             SMEM_RESIDUAL_A_SIZE_PER_STAGE +
             SMEM_RESIDUAL_SFA_SIZE_PER_STAGE) +
        SMEM_DGRAD_WEIGHT_SOURCE_SIZE +
        SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE;
    constexpr uint32_t kReadyParentControlEnd =
        kReadyParentBarrierOffset +
        (2u * kNumStages + kNumWeightLoadBarriers +
         2u * kNumEpilogueStages + kNumDispatchBarriers + 2u) *
            sizeof(Barrier) +
        sizeof(uint32_t);
    DG_STATIC_ASSERT(
        !(kReadyWgradSchedule || kK3BranchMajorBF16DynamicTail) ||
            (kReadyDW2TasksPerExpert % 4u == 0u &&
             kReadyDW13TasksPerExpert % 4u == 0u),
        "K3 dynamic expert ranges must preserve BF16 epilogue parity");
    DG_STATIC_ASSERT(
        !(kReadyWgradSchedule || kK3BranchMajorBF16DynamicTail) ||
            (4u * (BLOCK_M / kWgradBlockK)) %
                    (2u * kWgradStages) ==
                0u,
        "K3 dynamic four-task batches must restore BF16 mainloop phases");
    DG_STATIC_ASSERT(
        !(kReadyWgradSchedule || kK3BranchMajorBF16DynamicTail) ||
            4u % (2u * 2u) == 0u,
        "K3 dynamic four-task batches must restore BF16 epilogue phases");
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule ||
            kReadyParentControlEnd <= kReadyBf16SmemDataBytes,
        "K3 ready BF16 barriers overlap the parent control range");
    // The dW13 suffix arena beginning at 153856 is still live during W13
    // dgrad and cannot be lent to elastic dW2 production. Instead, place one
    // compact 32x64 engine after every parent data/control object. This tail
    // already belongs to the 230400-byte inline-wgrad launch, so the overlap
    // adds no dynamic shared memory.
    constexpr uint32_t kElasticDW2QuantScratchBegin =
        math::constexpr_align(kReadyParentControlEnd, 1024u);
    constexpr uint32_t kElasticDW2QuantScratchEnd =
        kElasticDW2QuantScratchBegin +
        kK3MxFp8DW2W13QuantScratchBytes;
    DG_STATIC_ASSERT(
        !kK3MxFp8WgradOverlap ||
            (kElasticDW2QuantScratchBegin == 219136u &&
             kElasticDW2QuantScratchEnd == 229376u &&
             kElasticDW2QuantScratchEnd <= kReadyBf16SmemDataBytes),
        "Elastic dW2 scratch must stay beyond live W13 control and inside the existing launch");
    DG_STATIC_ASSERT(
        !kReadyWgradSchedule ||
            (kReadyParentBarrierOffset == 218112u &&
             kReadyParentControlEnd == 218324u &&
             kReadyBf16SmemDataBytes == 229376u),
        "K3 ready shared-memory proof changed with the selected shape");
    DG_STATIC_ASSERT(
        !kK3MxFp8DW13Hybrid ||
            (kHybridDW13QuantScratchBegin == 196864u &&
             kHybridDW13QuantScratchEnd == 214400u &&
             kHybridDW13QuantScratchEnd <=
                 kReadyParentBarrierOffset &&
             kHybridDW13QuantScratchEnd <=
                 kReadyBf16SmemDataBytes),
        "Hybrid dW13 quantizer must fit after released five-stage BF16 data");
    constexpr bool kPublishRemoteGradients =
        kNumRanks > 1 &&
        (kDirectRemoteGradX || kComputeRouteGrad);
    constexpr bool kStreamingDirectGradXCombine =
        kReadyWgradSchedule && !kMultiRangeBackward && kNumRanks > 1 &&
        kDirectRemoteGradX &&
        kCombineOrderMode == CombineOrderMode::FixedTopK;
    auto* const direct_grad_x_pool_completions = [&]() -> uint32_t* {
        if constexpr (kStreamingDirectGradXCombine)
            return weight_tile_states + kDirectGradXPoolCompletionWord;
        return nullptr;
    }();
    auto* const direct_grad_x_ready_counts = [&]() -> int* {
        if constexpr (kStreamingDirectGradXCombine) {
            // Backward intentionally constructs a one-block Workspace view, so
            // its ring counters cannot index all source tokens. The large
            // forward-only source-index table immediately precedes metadata;
            // reuse its final max_tokens words after dispatch has retired.
            // Derive the address from the symmetric workspace itself: training
            // may pass a standalone clone as token_src_metadata, which is valid
            // for local reads but cannot be mapped to a peer rank.
            return reinterpret_cast<int*>(
                       backward_workspace.get_token_src_metadata_ptr(0)) -
                backward_workspace.num_max_tokens_per_rank;
        }
        return nullptr;
    }();
    // Ready-wgrad retains per-range schedulers plus one completion word for
    // every physical pool block.  The MXFP8 scheduler has a larger, unrelated
    // ABI and therefore starts at the next 128-byte boundary after that live
    // runtime tail.  All hybrid producers and the terminal consumer derive
    // their pointer through this single function; counter units can never be
    // confused by accidentally falling back to the ready-state base.
    const auto k3_mxfp8_dw13_hybrid_state = [&]() -> uint32_t* {
        if constexpr (kK3MxFp8ExactEpilogueRing) {
            // Option-A must publish source readiness before W2 divergence,
            // while the ordinary retired-weight state is still live.  Its
            // exact metadata therefore occupies the reserved plane-two
            // control slot for the complete early-consumer lifetime.
            return k3_mxfp8_exact_ring_control_state();
        }
        if constexpr (kK3MxFp8DW13Hybrid) {
            DG_DEVICE_ASSERT(
                backward_ranges.total_pool_rows % BLOCK_M == 0u);
            const uint32_t num_pool_blocks =
                backward_ranges.total_pool_rows / BLOCK_M;
            DG_DEVICE_ASSERT(
                num_pool_blocks <=
                kK3MxFp8DW13HybridMaxPoolCompletionWords);
            const uint32_t state_word = math::constexpr_align(
                kDirectGradXPoolCompletionWord + num_pool_blocks,
                kK3MxFp8DW13HybridStateAlignmentWords);
            DG_DEVICE_ASSERT(
                state_word >=
                    kDirectGradXPoolCompletionWord + num_pool_blocks &&
                state_word + K3MxFp8OverlapState::kNumWords <=
                    kNumW2WeightTileStates);
            return weight_tile_states + state_word;
        }
        return nullptr;
    };
    if constexpr (kK3MxFp8ExactEpilogueRing) {
        // Plane two belonged to the preceding fixed-top-k generation.  Hoist
        // its rank-wide reclamation edge before the grid divides into the
        // 132-CTA W2 prefix and 16-CTA exact suffix; neither subset can safely
        // execute this full-rank rendezvous after divergence.
        constexpr uint32_t kBeforeExactRingGridSyncIndex = 2u;
        constexpr uint32_t kBeforeExactRingBarrierTag = 7u;
        comm::nvlink_barrier<
            kNumRanks, kNumSMs, kNumThreads,
            kBeforeExactRingGridSyncIndex,
            kBeforeExactRingBarrierTag>(
                backward_workspace, backward_sym_buffer,
                blockIdx.x, threadIdx.x,
                []() { __syncthreads(); });

        auto* const exact_state =
            k3_mxfp8_exact_ring_control_state();
        auto* const source_ready =
            k3_mxfp8_exact_ring_source_ready();
        const uint32_t total_pool_blocks =
            backward_ranges.total_pool_rows / BLOCK_M;
        DG_DEVICE_ASSERT(
            backward_ranges.total_pool_rows % BLOCK_M == 0u &&
            (static_cast<uint64_t>(K3MxFp8OverlapState::kNumWords) +
             total_pool_blocks) * sizeof(uint32_t) <=
                static_cast<uint64_t>(kExactRingControlRows) *
                    kHidden * sizeof(cutlass::bfloat16_t));
        const uint32_t global_thread =
            static_cast<uint32_t>(blockIdx.x) * kNumThreads +
            threadIdx.x;
        constexpr uint32_t kGlobalThreads = kNumSMs * kNumThreads;
        for (uint32_t physical_block = global_thread;
             physical_block < total_pool_blocks;
             physical_block += kGlobalThreads) {
            source_ready[physical_block] = 0u;
        }
        // The prior launch's direct-dX phase leaves arbitrary BF16 in planes
        // two through five.  Invalidate every atomic publication key before
        // this launch's grid barrier and epoch release; otherwise those bytes
        // could impersonate a valid {epoch, sequence} on a later launch.
        auto* const combine_plane_zero =
            reinterpret_cast<uint8_t*>(
                const_cast<cd_dtype_t*>(backward_grad_y));
        k3_mxfp8_clear_dw13_ab_ring_publication_keys<
            kHidden, kIntermediateHidden, BLOCK_M>(
                combine_plane_zero,
                backward_workspace.num_max_tokens_per_rank,
                total_pool_blocks, launch_epoch,
                global_thread, kGlobalThreads);
        prepare_k3_mxfp8_dw2_overlap_state<
            kNumExperts, BLOCK_M, kNumSMs, kNumThreads>(
                expert_counts, backward_ranges, exact_state,
                K3MxFp8WgradGridBarrier<kNumSMs>{
                    phase_count, phase_sense});
        if (blockIdx.x == 0u && threadIdx.x == 0u) {
            const uint32_t dw13_epoch = launch_epoch ^ 0x80000000u;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(exact_state + K3MxFp8OverlapState::kDW13Epoch),
                   "r"(dw13_epoch)
                : "memory");
        }
    }
    const auto no_input_tile_retired =
        [] (uint32_t) {};
    const auto no_background_work =
        [] (uint32_t, uint32_t) {};
    const auto run_wgrad = [&]<
        bool kFuseWgradCombine,
        uint32_t kExtraCombineThreads = 0,
        bool kPublishBeforeCombineReduce = false,
        uint32_t kRunWgradStages = kWgradStages,
        typename InputTileRetiredCallback,
        typename BackgroundWorkCallback>(
        const uint32_t shape_m,
        const uint32_t shape_n,
        const cute::TmaDescriptor& tensor_map_a,
        const cute::TmaDescriptor& tensor_map_b,
        const cute::TmaDescriptor& tensor_map_d,
        const bool combine_reduce,
        InputTileRetiredCallback input_tile_retired,
        BackgroundWorkCallback background_work) {
        sm100_bf16_gemm_body<
            cute::UMMA::Major::MN,
            cute::UMMA::Major::MN,
            0, 0, 0,
            kWgradBlockM, kWgradBlockN, kWgradBlockK,
            kNumExperts,
            kWgradSwizzle, kWgradSwizzle, kWgradSwizzle,
            kRunWgradStages,
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
                direct_grad_x_ready_counts,
                num_backward_tokens,
                combine_first_range_tokens,
                combine_second_range_begin,
                backward_workspace.num_max_tokens_per_rank,
                num_topk, kHidden, combine_reduce,
                smem_buffer, false,
                input_tile_retired, background_work);
    };

#if DG_EXPERIMENTAL_K3_READY_WGRAD || \
    DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL
    // Execute one externally claimed range while allowing the enclosing K3
    // scheduler to retain the BF16 mbarrier/TMEM allocation across dW2/dW13
    // descriptor switches.  Every range is a multiple of four cluster tasks;
    // K3's BLOCK_M=192 makes its K work a multiple of 2*kWgradStages, so both
    // the epilogue and mainloop phases return to zero before the next call.
    const auto run_ready_wgrad_range = [&]<
        typename TaskProvider,
        typename BatchResourceHooks,
        bool kFuseWgradCombine,
        uint32_t kExtraCombineThreads = 0,
        bool kPublishBeforeCombineReduce = false,
        bool kRangeAccumulateWgrad = kAccumulateWgrad,
        uint32_t kRunWgradStages = kWgradStages,
        typename TaskStream = sched::ExternalKGroupedRangeStream,
        typename InputTileRetiredCallback,
        typename BackgroundWorkCallback>(
        const uint32_t shape_m,
        const uint32_t shape_n,
        const cute::TmaDescriptor& tensor_map_a,
        const cute::TmaDescriptor& tensor_map_b,
        const cute::TmaDescriptor& tensor_map_d,
        const TaskStream& task_stream,
        const bool combine_reduce,
        InputTileRetiredCallback input_tile_retired,
        BackgroundWorkCallback background_work,
        const cute::TmaDescriptor* phase_one_tensor_map_a = nullptr,
        const cute::TmaDescriptor* phase_one_tensor_map_b = nullptr,
        const cute::TmaDescriptor* phase_one_tensor_map_d = nullptr) {
        sm100_bf16_gemm_body<
            cute::UMMA::Major::MN,
            cute::UMMA::Major::MN,
            TaskProvider::kTaskShapeM,
            TaskProvider::kTaskShapeN,
            0,
            kWgradBlockM, kWgradBlockN, kWgradBlockK,
            kNumExperts,
            kWgradSwizzle, kWgradSwizzle, kWgradSwizzle,
            kRunWgradStages,
            kWgradRoleThreads, kWgradRoleThreads,
            2, false,
            kNumSMs,
            kWgradBlockK,
            false, true,
            GemmType::KGroupedContiguous,
            kRangeAccumulateWgrad,
            cd_dtype_t,
            kWgradTensorUtil,
            kNumRanks,
            kFuseWgradCombine,
            CombineOrderMode::FixedTopK,
            kExtraCombineThreads,
            kPublishBeforeCombineReduce,
            TaskProvider,
            BatchResourceHooks>(
                reinterpret_cast<int*>(
                    const_cast<TaskStream*>(
                        &task_stream)),
                shape_m, shape_n,
                kMultiRangeBackward
                    ? backward_ranges.total_pool_rows
                    : num_pool_rows,
                tensor_map_a, tensor_map_b, tensor_map_d,
                backward_sym_buffer, backward_workspace,
                backward_grad_x_output,
                const_cast<cd_dtype_t*>(backward_grad_y),
                nullptr,
                direct_grad_x_ready_counts,
                num_backward_tokens,
                combine_first_range_tokens,
                combine_second_range_begin,
                backward_workspace.num_max_tokens_per_rank,
                num_topk, kHidden, combine_reduce,
                smem_buffer, false,
                input_tile_retired, background_work,
                phase_one_tensor_map_a,
                phase_one_tensor_map_b,
                phase_one_tensor_map_d);
    };

#if DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL
    const auto run_branch_major_early_dw2 = [&]<bool kEnabled>() {
        if constexpr (kEnabled) {
            static_assert(
                !kEnabled ||
                    (kK3BranchMajorBF16EarlyDW2Overlap &&
                     !kEarlyW2Wgrad && kPublishRemoteGradients),
                "early branch-major dW2 requires exact two-range publication");
            DG_DEVICE_ASSERT(backward_ranges.num_ranges == 2u);

            // The parent W13 resources are cluster-local.  A cluster may
            // retire and replace them as soon as both peer CTAs have exhausted
            // their static W13 streams; no device-wide handoff is required.
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u && cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0u; i < kNumStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u; i < kNumEpilogueStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u; i < kNumDispatchBarriers; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dispatch_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u; i < 2u; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dequant_barriers + i));
                }
            }
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u)
                Allocator().free(0, kNumTmemCols);
            __syncthreads();

            static_assert(
                !kUseReducedW2ProducerSet,
                "early branch-major dW2 register entry state changed");
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();

            constexpr uint32_t kDynamicBatchTasks = 4u;
            using DynamicTwoSegmentDW2Provider = sched::
                ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    kHidden, kIntermediateHidden,
                    kDynamicBatchTasks,
                    kReadyDW2TasksPerExpert,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord>;
            using DynamicDW2RetainedResources =
                Sm100Bf16GemmBatchResourceHooks<true, false>;
            static_assert(
                DynamicTwoSegmentDW2Provider::kCompleteAcquireMask ==
                        kReadyCompleteRoleMask &&
                    kReadyDW2TasksPerExpert % kDynamicBatchTasks == 0u,
                "early two-segment dW2 scheduler contract changed");
            static_assert(
                DynamicDW2RetainedResources::kInitializeBatchResources &&
                    !DynamicDW2RetainedResources::kReleaseBatchResources,
                "early dW2 must retain BF16 resources for terminal dW13");

            const uint32_t second_range_index =
                backward_ranges.reverse_range_index(1u);
            auto* const union_state =
                weight_tile_states + kReadyTerminalUnionStateWord;
            const auto* const second_state =
                weight_tile_states +
                second_range_index * kReadyRangeStateStride;
            const uint32_t dynamic_cluster_idx = blockIdx.x / 2u;
            auto* const dw2_mailbox =
                union_state + kReadyDW2ClusterSlotWord +
                dynamic_cluster_idx * kReadyClusterSlotWords;
            const sched::ExternalKGroupedTerminalTwoSegmentRangeStream
                dw2_stream{
                    union_state,
                    second_state,
                    union_state + kReadyDW2CursorWord,
                    union_state[kReadyDW2TasksWord],
                    dw2_mailbox,
                    kDynamicBatchTasks,
                    kReadyDW2TasksPerExpert,
                };
            trace_begin(17);
            run_ready_wgrad_range.template operator()<
                DynamicTwoSegmentDW2Provider,
                DynamicDW2RetainedResources,
                kPublishRemoteGradients>(
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    dw2_stream, false,
                    no_input_tile_retired,
                    no_background_work);
            trace_end(17);
        }
    };
#endif
#endif

    // Co-schedule a bounded prefix of the exact W13 packed-weight converter on
    // dW2's otherwise-idle warps. Five BF16 mainloop stages free one 32-KiB
    // stage; the background TMA tile, converter output, and two private
    // mbarriers fit wholly inside that retired sixth-stage tail. Warp 2 is a
    // valid callback role when fused-combine is disabled; warps 8..23 are the
    // sixteen independent converters. Private shared barriers avoid consuming
    // a user named-barrier ID while dW2's epilogue is live.
    constexpr uint32_t kBackgroundControlWarp = 2;
    constexpr uint32_t kBackgroundFirstConverterWarp = 8;
    constexpr uint32_t kBackgroundConverterWarps = 16;
    constexpr uint32_t kBackgroundParticipantThreads =
        (kBackgroundConverterWarps + 1) * 32;
    constexpr uint32_t kBackgroundBuilderWeightBytes =
        DGRAD_BLOCK_K * (LOAD_BLOCK_N / 2) * sizeof(int8_t);
    constexpr uint32_t kBackgroundBuilderScaleBytes =
        DGRAD_BLOCK_K * (LOAD_BLOCK_N / kGranK) * sizeof(float);
    constexpr uint32_t kBackgroundBuilderOutputBytes =
        DGRAD_BLOCK_K * LOAD_BLOCK_N *
        sizeof(residual_dgrad_dtype_t);
    constexpr uint32_t kBackgroundBuilderOutputScaleBytes =
        LOAD_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t kBackgroundBuilderScratchBytes =
        kBackgroundBuilderWeightBytes + kBackgroundBuilderScaleBytes +
        kBackgroundBuilderOutputBytes +
        kBackgroundBuilderOutputScaleBytes;

    // Mirror sm100_bf16_gemm_body's non-swap, 2-CTA shared layout exactly.
    // Keeping these equations beside the callback makes overlap and capacity
    // failures compile-time errors if the embedded wgrad geometry changes.
    constexpr uint32_t kWgradLoadBlockM = kWgradBlockM;
    constexpr uint32_t kWgradLoadBlockN = kWgradBlockN / 2;
    constexpr uint32_t kWgradStoreBlockM = 128;
    constexpr uint32_t kWgradStoreBlockN =
        kWgradSwizzle / sizeof(cd_dtype_t);
    constexpr uint32_t kWgradCDBytes =
        2 * kWgradStoreBlockM * kWgradStoreBlockN *
        sizeof(cd_dtype_t);
    constexpr uint32_t kWgradMainloopBytesPerStage =
        (kWgradLoadBlockM + kWgradLoadBlockN) *
        kWgradBlockK * sizeof(cd_dtype_t);
    constexpr uint32_t kWgradEpilogueStages = 2;
    constexpr uint32_t kWgradControlBytes = []<uint32_t kStages>() {
        // Three stage-indexed barrier arrays, two epilogue arrays, the
        // tensor-core barrier slot, and the shared TMEM pointer.
        return (3 * kStages + 2 * kWgradEpilogueStages + 1) *
                   sizeof(Barrier) +
               sizeof(uint32_t);
    }.template operator()<kEarlyDW2WgradStages>();
    constexpr uint32_t kWgradOriginalControlBytes =
        (3 * kWgradStages + 2 * kWgradEpilogueStages + 1) *
            sizeof(Barrier) +
        sizeof(uint32_t);
    constexpr uint32_t kBackgroundScratchOffset =
        math::constexpr_align(
            kWgradCDBytes +
                kEarlyDW2WgradStages *
                    kWgradMainloopBytesPerStage +
                kWgradControlBytes,
            1024u);
    constexpr uint32_t kBackgroundLoadBarrierOffset =
        kBackgroundScratchOffset + kBackgroundBuilderScratchBytes;
    constexpr uint32_t kBackgroundParticipantBarrierOffset =
        kBackgroundLoadBarrierOffset + sizeof(Barrier);
    constexpr uint32_t kBackgroundScratchEnd =
        kBackgroundParticipantBarrierOffset + sizeof(Barrier);
    constexpr uint32_t kOriginalWgradSharedEnd =
        kWgradCDBytes +
        kWgradStages * kWgradMainloopBytesPerStage +
        kWgradOriginalControlBytes;
    DG_STATIC_ASSERT(
        !kBackgroundW13WeightCache ||
            kBackgroundFirstConverterWarp +
                    kBackgroundConverterWarps <=
                kNumThreads / 32,
        "W13 background roles exceed the resident warp set");
    DG_STATIC_ASSERT(
        !kBackgroundW13WeightCache ||
            kBackgroundBuilderWeightBytes % 128 == 0 &&
            (kBackgroundBuilderWeightBytes +
             kBackgroundBuilderScaleBytes) % 128 == 0 &&
            (kBackgroundBuilderWeightBytes +
             kBackgroundBuilderScaleBytes +
             kBackgroundBuilderOutputBytes) % 128 == 0,
        "W13 background TMA operands must remain 128-byte aligned");
    DG_STATIC_ASSERT(
        !kBackgroundW13WeightCache ||
            kBackgroundScratchEnd <= kOriginalWgradSharedEnd,
        "W13 background scratch exceeds the retired sixth BF16 stage");

    const auto run_early_dw2_w13_background = [&] (
        const uint32_t background_warp_idx,
        const uint32_t background_lane_idx) {
      if constexpr (kBackgroundW13WeightCache) {
        const bool is_control_warp =
            background_warp_idx == kBackgroundControlWarp;
        const bool is_converter_warp =
            background_warp_idx >= kBackgroundFirstConverterWarp &&
            background_warp_idx <
                kBackgroundFirstConverterWarp +
                    kBackgroundConverterWarps;
        if (!is_control_warp && !is_converter_warp)
            return;

        auto* builder_scratch =
            smem_buffer + kBackgroundScratchOffset;
        auto* builder_weight_source =
            reinterpret_cast<int8_t*>(builder_scratch);
        auto* builder_scale_source = reinterpret_cast<float*>(
            builder_scratch + kBackgroundBuilderWeightBytes);
        auto* builder_output = reinterpret_cast<uint8_t*>(
            builder_scratch + kBackgroundBuilderWeightBytes +
            kBackgroundBuilderScaleBytes);
        auto* builder_output_scales = reinterpret_cast<uint32_t*>(
            builder_scratch + kBackgroundBuilderWeightBytes +
            kBackgroundBuilderScaleBytes +
            kBackgroundBuilderOutputBytes);
        auto* builder_load_barrier = reinterpret_cast<Barrier*>(
            smem_buffer + kBackgroundLoadBarrierOffset);
        auto* builder_participant_barrier = reinterpret_cast<Barrier*>(
            smem_buffer + kBackgroundParticipantBarrierOffset);

        if (is_control_warp && cute::elect_one_sync()) {
            cute::prefetch_tma_descriptor(&tensor_map_w13_weights);
            cute::prefetch_tma_descriptor(&tensor_map_w13_scales);
            cute::prefetch_tma_descriptor(
                &tensor_map_w13_dgrad_weights);
            cute::prefetch_tma_descriptor(
                &tensor_map_w13_dgrad_weights_sf);
        }
        uint32_t participant_phase = 0;
        const auto participant_sync = [&]() {
            builder_participant_barrier->arrive();
            builder_participant_barrier->wait(participant_phase);
            participant_phase ^= 1;
        };
        participant_sync();

        constexpr uint32_t output_mn = kHidden;
        constexpr uint32_t output_k = 2 * kIntermediateHidden;
        constexpr uint32_t num_n_blocks =
            output_mn / LOAD_BLOCK_N;
        constexpr uint32_t num_k_blocks =
            output_k / DGRAD_BLOCK_K;
        constexpr uint64_t num_groups =
            static_cast<uint64_t>(kNumExperts) * num_n_blocks;
        constexpr uint32_t w13_ready_epoch_mask = 0x80000000u;
        const uint32_t ready_epoch =
            launch_epoch ^ w13_ready_epoch_mask;
        uint32_t builder_load_phase = 0;

        #pragma unroll 1
        for (uint32_t group_sequence = 0;
             group_sequence < kBackgroundW13GroupsPerCTA;
             ++group_sequence) {
            const uint64_t group_idx =
                blockIdx.x +
                static_cast<uint64_t>(group_sequence) * kNumSMs;
            if (group_idx >= num_groups)
                break;
            const uint32_t n_block_idx =
                group_idx % num_n_blocks;
            const uint32_t expert_idx =
                group_idx / num_n_blocks;

            if (__ldg(expert_counts + expert_idx) != 0) {
              #pragma unroll 1
              for (uint32_t k_block_idx = 0;
                   k_block_idx < num_k_blocks;
                   ++k_block_idx) {
                const uint64_t state_idx =
                    (static_cast<uint64_t>(expert_idx) *
                         num_k_blocks +
                     k_block_idx) *
                        num_n_blocks +
                    n_block_idx;
                if (is_control_warp && cute::elect_one_sync()) {
                    tma::copy<
                        LOAD_BLOCK_N / 2, DGRAD_BLOCK_K, 0, int8_t>(
                        &tensor_map_w13_weights,
                        builder_load_barrier,
                        builder_weight_source,
                        n_block_idx * (LOAD_BLOCK_N / 2),
                        expert_idx * output_k +
                            k_block_idx * DGRAD_BLOCK_K);
                    tma::copy<
                        LOAD_BLOCK_N / kGranK,
                        DGRAD_BLOCK_K, 0, float>(
                        &tensor_map_w13_scales,
                        builder_load_barrier,
                        builder_scale_source,
                        n_block_idx * (LOAD_BLOCK_N / kGranK),
                        expert_idx * output_k +
                            k_block_idx * DGRAD_BLOCK_K);
                    builder_load_barrier->arrive_and_expect_tx(
                        kBackgroundBuilderWeightBytes +
                        kBackgroundBuilderScaleBytes);
                }

                if (is_converter_warp) {
                    builder_load_barrier->wait(
                        builder_load_phase);
                    k3_mxfp4_to_mxfp8_transposed_tile_one_pass<
                        kBackgroundConverterWarps>(
                        builder_weight_source,
                        builder_scale_source,
                        builder_output,
                        builder_output_scales,
                        n_block_idx,
                        (background_warp_idx -
                         kBackgroundFirstConverterWarp) * 32 +
                            background_lane_idx);
                    cutlass::arch::fence_view_async_shared();
                }
                participant_sync();

                if (is_control_warp && cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_w13_dgrad_weights,
                        builder_output,
                        k_block_idx * DGRAD_BLOCK_K,
                        expert_idx * output_mn +
                            n_block_idx * LOAD_BLOCK_N);
                    cute::SM90_TMA_STORE_2D::copy(
                        &tensor_map_w13_dgrad_weights_sf,
                        builder_output_scales,
                        n_block_idx * LOAD_BLOCK_N,
                        expert_idx *
                                (output_k / (kGranK * 4)) +
                            k_block_idx);
                    cute::tma_store_arrive();
                    ptx::tma_store_wait<0>();
                    asm volatile(
                        "fence.proxy.async.global;" ::: "memory");
                    asm volatile(
                        "st.release.gpu.global.u32 [%0], %1;"
                        :: "l"(weight_tile_states +
                               kNumW2WeightTileStates +
                               state_idx),
                           "r"(ready_epoch)
                        : "memory");
                }
                participant_sync();
                builder_load_phase ^= 1;
              }
            }

            // Cursor publication is strictly group-granular: it follows the
            // store wait and tile-epoch release for every K tile in this group.
            if (is_control_warp && cute::elect_one_sync()) {
                asm volatile(
                    "st.release.gpu.global.u32 [%0], %1;"
                    :: "l"(w13_background_group_cursors + blockIdx.x),
                       "r"(group_sequence + 1)
                    : "memory");
            }
        }

        // Do not end either private barrier lifetime inside the callback. A
        // control warp may leave this final mbarrier wait while another
        // participant is still retiring its wait instruction. The converged
        // all-CTA join after run_wgrad owns both invalidations instead.
        participant_sync();
      }
    };

#if DG_EXPERIMENTAL_K3_RANGE_WGRAD
    // Range entry point retaining the exact BF16 TMA/UMMA/epilogue and
    // communication roles while consuming a caller-owned, complete-cluster
    // logical range rather than blockIdx-derived persistent tasks.
    const auto run_wgrad_range = [&]<
        typename TaskProvider,
        bool kFuseWgradCombine,
        uint32_t kExtraCombineThreads = 0,
        bool kPublishBeforeCombineReduce = false>(
        const uint32_t shape_m,
        const uint32_t shape_n,
        const cute::TmaDescriptor& tensor_map_a,
        const cute::TmaDescriptor& tensor_map_b,
        const cute::TmaDescriptor& tensor_map_d,
        const sched::ExternalKGroupedRangeStream& task_stream,
        const bool combine_reduce) {
        sm100_bf16_gemm_body<
            cute::UMMA::Major::MN,
            cute::UMMA::Major::MN,
            TaskProvider::kTaskShapeM,
            TaskProvider::kTaskShapeN,
            0,
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
            kPublishBeforeCombineReduce,
            TaskProvider,
            Sm100Bf16GemmDefaultBatchResourceHooks>(
                reinterpret_cast<int*>(
                    const_cast<sched::ExternalKGroupedRangeStream*>(
                        &task_stream)),
                shape_m, shape_n, num_pool_rows,
                tensor_map_a, tensor_map_b, tensor_map_d,
                backward_sym_buffer, backward_workspace,
                backward_grad_x_output,
                const_cast<cd_dtype_t*>(backward_grad_y),
                nullptr,
                direct_grad_x_ready_counts,
                num_backward_tokens,
                combine_first_range_tokens,
                combine_second_range_begin,
                backward_workspace.num_max_tokens_per_rank,
                num_topk, kHidden, combine_reduce,
                smem_buffer, false,
                no_input_tile_retired, no_background_work);
    };
#endif

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

#if DG_EXPERIMENTAL_K3_RANGE_WGRAD
    constexpr uint32_t kOverlapStateWords = 256;
    constexpr uint32_t kOverlapPoolPrefixWord = 31;
    constexpr uint32_t kOverlapActiveExpertWord = 144;
    constexpr uint32_t kOverlapDW2TasksWord = 10;
    constexpr uint32_t kOverlapDW13TasksWord = 11;
    constexpr uint32_t kW13DgradRangeBatchTasks =
        DG_EXPERIMENTAL_K3_RANGE_W13_DGRAD_BATCH_TASKS;
    constexpr uint32_t kDW2RangeBatchTasks =
        DG_EXPERIMENTAL_K3_RANGE_DW2_BATCH_TASKS;
    constexpr uint32_t kDW13RangeBatchTasks =
        DG_EXPERIMENTAL_K3_RANGE_DW13_BATCH_TASKS;
    DG_STATIC_ASSERT(
        kW13DgradRangeBatchTasks != 0u &&
            kDW2RangeBatchTasks != 0u &&
            kDW13RangeBatchTasks != 0u,
        "K3 range batch sizes must be nonzero");
    DG_STATIC_ASSERT(
        kDW2RangeBatchTasks % 4u == 0u &&
            kDW13RangeBatchTasks % 4u == 0u,
        "K3 wgrad range batches must reset both epilogue phases");
    DG_STATIC_ASSERT(
        kOverlapStateWords <= kNumW2WeightTileStates,
        "K3 overlap state must fit in the retired W2 readiness prefix");

    // Initialize the compact state only after the caller has retired every W2
    // readiness consumer. The call site below is after both dgrad phases, so
    // the first 256 words are dead and need no additional allocation.
    const auto initialize_overlap_state = [&]() {
        if (blockIdx.x == 0) {
            auto* state = weight_tile_states;
            if (threadIdx.x < kOverlapStateWords)
                state[threadIdx.x] = 0;
            __syncthreads();

            if (threadIdx.x == 0) {

                auto* pool_prefix = state + kOverlapPoolPrefixWord;
                auto* active_experts = state + kOverlapActiveExpertWord;
                uint32_t pool_blocks = 0;
                uint32_t active_count = 0;
                pool_prefix[0] = 0;
                for (uint32_t expert = 0; expert < kNumExperts; ++expert) {
                    const auto count = static_cast<uint32_t>(
                        __ldg(expert_counts + expert));
                    if (count != 0)
                        active_experts[active_count++] = expert;
                    pool_blocks += math::ceil_div(count, BLOCK_M);
                    pool_prefix[expert + 1] = pool_blocks;
                }

                constexpr uint32_t kW13DgradTasksPerPoolBlock =
                    math::constexpr_ceil_div(kHidden / BLOCK_N, 2u);
                constexpr uint32_t kDW2TasksPerExpert =
                    math::constexpr_ceil_div(kHidden, kWgradBlockM) / 2u *
                    math::constexpr_ceil_div(
                        kIntermediateHidden, kWgradBlockN);
                constexpr uint32_t kDW13TasksPerExpert =
                    math::constexpr_ceil_div(
                        2u * kIntermediateHidden, kWgradBlockM) / 2u *
                    math::constexpr_ceil_div(kHidden, kWgradBlockN);
                const auto w13_dgrad_tasks =
                    pool_blocks * kW13DgradTasksPerPoolBlock;
                const auto dw2_tasks = active_count * kDW2TasksPerExpert;
                const auto dw13_tasks = active_count * kDW13TasksPerExpert;
                const auto w13_dgrad_batches = math::ceil_div(
                    w13_dgrad_tasks, kW13DgradRangeBatchTasks);
                const auto dw2_batches = math::ceil_div(
                    dw2_tasks, kDW2RangeBatchTasks);
                const auto dw13_batches = math::ceil_div(
                    dw13_tasks, kDW13RangeBatchTasks);
                const auto max_batches = cute::max(
                    w13_dgrad_batches,
                    cute::max(dw2_batches, dw13_batches));

                state[0] = 0x4b334f56u;
                state[1] = 1u;
                state[2] = launch_epoch;
                state[3] = launch_epoch;
                state[4] = kNumExperts;
                state[5] = kNumRanks;
                state[6] = kNumSMs / 2u;
                state[7] = active_count;
                state[8] = pool_blocks;
                state[9] = w13_dgrad_tasks;
                state[kOverlapDW2TasksWord] = dw2_tasks;
                state[kOverlapDW13TasksWord] = dw13_tasks;
                state[12] = max_batches;
                state[13] = 3u * max_batches;
                state[19] = kW13DgradRangeBatchTasks;
                state[20] = kDW2RangeBatchTasks;
                state[21] = kDW13RangeBatchTasks;
                state[22] = w13_dgrad_batches;
                state[23] = dw2_batches;
                state[24] = dw13_batches;
                __threadfence();
            }
        }
        full_grid_phase_barrier(kTraceSiteCount, 1);
    };
#endif

#if DG_EXPERIMENTAL_K3_READY_WGRAD
    // Prepare the exact BF16 operands and compact scheduler state before
    // v409's already-required route/streaming-dispatch grid join. This moves
    // no arithmetic: it is the legacy padding/empty-expert work executed at
    // the last producer boundary, then published to the same TMA proxy used by
    // grouped wgrad.
    const auto prepare_ready_wgrad_state_and_operands = [&] (
        const bool prepare_operands = true,
        const bool prepare_state = true,
        uint32_t* ready_state = nullptr) {
      if constexpr (kReadyWgradSchedule) {
        if constexpr (kClearWgradPadding) {
          if (prepare_operands) {
            constexpr uint32_t kPaddingColumns =
                kHidden + 3u * kIntermediateHidden;
            uint32_t pad_pool_block_offset =
                active_pool_block_begin;
            uint32_t pad_global_block = 0;
            #pragma unroll 1
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens = static_cast<uint32_t>(
                    __ldg(active_expert_counts + expert_idx));
                const uint32_t num_blocks =
                    math::ceil_div(num_tokens, BLOCK_M);
                if (num_blocks != 0) {
                    const uint32_t last_valid =
                        num_tokens - (num_blocks - 1u) * BLOCK_M;
                    const uint32_t pool_block =
                        pad_pool_block_offset + num_blocks - 1u;
                    if (pad_global_block % kNumSMs == blockIdx.x) {
                        for (uint32_t linear = threadIdx.x;
                             linear <
                                 (BLOCK_M - last_valid) * kPaddingColumns;
                             linear += kNumThreads) {
                            const uint32_t row_delta =
                                linear / kPaddingColumns;
                            const uint32_t col =
                                linear - row_delta * kPaddingColumns;
                            const uint32_t pool_row =
                                pool_block * BLOCK_M + last_valid + row_delta;
                            if (col < kHidden) {
                                grad_ye_output[
                                    static_cast<uint64_t>(pool_row) *
                                        kHidden + col] = cd_dtype_t(0.0f);
                            } else if (
                                col < kHidden + kIntermediateHidden) {
                                h_weighted_output[
                                    static_cast<uint64_t>(pool_row) *
                                        kIntermediateHidden +
                                    col - kHidden] = cd_dtype_t(0.0f);
                            } else {
                                grad_gate_up_output[
                                    static_cast<uint64_t>(pool_row) *
                                        (2u * kIntermediateHidden) +
                                    col - kHidden - kIntermediateHidden] =
                                        cd_dtype_t(0.0f);
                            }
                        }
                    }
                    ++pad_global_block;
                }
                pad_pool_block_offset += num_blocks;
            }
          }
        }

        if constexpr (!kAccumulateWgrad) {
            if (prepare_operands &&
                (!kMultiRangeBackward || !active_accumulate_wgrad)) {
                constexpr uint64_t kBF16PerVector =
                    sizeof(uint4) / sizeof(cd_dtype_t);
                constexpr uint64_t kW2VectorsPerExpert =
                    static_cast<uint64_t>(kHidden) *
                    kIntermediateHidden / kBF16PerVector;
                constexpr uint64_t kW13VectorsPerExpert =
                    static_cast<uint64_t>(2u * kIntermediateHidden) *
                    kHidden / kBF16PerVector;
                constexpr uint64_t kVectorsPerExpert =
                    kMultiRangeBackward
                    ? kW2VectorsPerExpert
                    : kW2VectorsPerExpert + kW13VectorsPerExpert;
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
                    if (__ldg(active_expert_counts + expert_idx) != 0)
                        continue;
                    bool needed_by_later_range = false;
                    if constexpr (kMultiRangeBackward) {
                        #pragma unroll 1
                        for (uint32_t later_iteration =
                                 active_range_iteration + 1u;
                             later_iteration < backward_ranges.num_ranges;
                             ++later_iteration) {
                            const uint32_t later_range_index =
                                backward_ranges.reverse_range_index(
                                    later_iteration);
                            needed_by_later_range |= __ldg(
                                expert_counts +
                                backward_ranges.expert_counts_begin(
                                    later_range_index, kNumExperts) +
                                expert_idx) != 0;
                        }
                    }
                    if (!clear_empty_wgrad_expert_outputs &&
                        !needed_by_later_range)
                        continue;
                    for (uint64_t linear = global_thread;
                         linear < kVectorsPerExpert;
                         linear += kGlobalThreads) {
                        if (linear < kW2VectorsPerExpert) {
                            w2_vectors[
                                static_cast<uint64_t>(expert_idx) *
                                    kW2VectorsPerExpert + linear] = zero;
                        } else if constexpr (!kMultiRangeBackward) {
                            w13_vectors[
                                static_cast<uint64_t>(expert_idx) *
                                    kW13VectorsPerExpert +
                                linear - kW2VectorsPerExpert] = zero;
                        }
                    }
                }
            }
        }

        if (prepare_state && blockIdx.x == 0 && threadIdx.x == 0) {
            auto* state = ready_state != nullptr
                ? ready_state
                : weight_tile_states;
            for (uint32_t word = 0; word < kReadyStateWords; ++word)
                state[word] = 0u;

            auto* pool_prefix = state + kReadyPoolPrefixWord;
            auto* active_experts = state + kReadyActiveExpertWord;
            uint32_t pool_blocks = active_pool_block_begin;
            uint32_t active_count = 0u;
            pool_prefix[0] = active_pool_block_begin;
            for (uint32_t expert = 0;
                 expert < kNumExperts; ++expert) {
                const uint32_t count = static_cast<uint32_t>(
                    __ldg(active_expert_counts + expert));
                if (count != 0u)
                    active_experts[active_count++] = expert;
                pool_blocks += math::ceil_div(count, BLOCK_M);
                pool_prefix[expert + 1u] = pool_blocks;
            }
            const uint32_t dw2_tasks =
                active_count * kReadyDW2TasksPerExpert;
            const uint32_t dw13_tasks =
                active_count * kReadyDW13TasksPerExpert;
            state[kReadyMagicWord] = 0x4b335257u; // "K3RW"
            state[kReadyDW2CursorWord] = 0u;
            state[kReadyDW13CursorWord] = 0u;
            state[kReadyActiveCountWord] = active_count;
            state[kReadyPoolBlocksWord] = pool_blocks;
            state[kReadyDW2TasksWord] = dw2_tasks;
            state[kReadyDW13TasksWord] = dw13_tasks;
            for (uint32_t cluster = 0u;
                 cluster < kReadyNumClusters; ++cluster) {
                state[kReadyDW2ClusterSlotWord +
                      cluster * kReadyClusterSlotWords] =
                    kReadyCompleteRoleMask;
                state[kReadyDW13ClusterSlotWord +
                      cluster * kReadyClusterSlotWords] =
                    kReadyCompleteRoleMask;
            }
            // Publish the epoch last. Dynamic range publishers acquire this
            // generation before touching the cursor, prefix, or mailboxes.
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(state + kReadyEpochWord),
                   "r"(active_range_epoch)
                : "memory");
        }

        // Each writer publishes its generic BF16 stores to the async/TMA proxy
        // before thread zero enters the existing full-grid phase barrier.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
      }
    };

    // Up to three physical range arenas remain immutable until every late W13
    // producer has joined the wgrad suffix. Build the concatenated-K schedule
    // in a separate retired-W2 arena instead of borrowing any live range's
    // active-expert list. The dedicated high generation bit distinguishes the
    // union publication from every range-local W2Wgrad generation.
    const auto ready_terminal_wgrad_epoch = [&]() {
        if constexpr (kReadyWgradSchedule && kMultiRangeBackward) {
            const uint32_t first_range_index =
                backward_ranges.reverse_range_index(0u);
            return k3_multirange_epoch(
                       K3MultiRangeBackwardPhase::W2Wgrad,
                       backward_ranges.epoch_seed(first_range_index)) |
                0x80000000u;
        }
        return 0u;
    };

    const auto prepare_ready_terminal_wgrad_state = [&]() {
      if constexpr (kReadyWgradSchedule && kMultiRangeBackward) {
        DG_DEVICE_ASSERT(
            backward_ranges.num_ranges > 0u &&
            backward_ranges.num_ranges <= kK3MaxBackwardRanges);
        const uint32_t first_range_index =
            backward_ranges.reverse_range_index(0u);
        const uint32_t second_range_index =
            backward_ranges.num_ranges > 1u
            ? backward_ranges.reverse_range_index(1u)
            : (first_range_index + 1u) % kK3MaxBackwardRanges;
        auto* const first_state =
            weight_tile_states +
            first_range_index * kReadyRangeStateStride;
        auto* const second_state =
            weight_tile_states +
            second_range_index * kReadyRangeStateStride;
        const uint32_t third_range_index =
            backward_ranges.num_ranges > 2u
            ? backward_ranges.reverse_range_index(2u)
            : (second_range_index + 1u) % kK3MaxBackwardRanges;
        auto* const third_state =
            weight_tile_states +
            third_range_index * kReadyRangeStateStride;
        auto* const union_state =
            weight_tile_states + kReadyTerminalUnionStateWord;

        if (blockIdx.x == 0u && threadIdx.x == 0u) {
            const uint32_t first_epoch = k3_multirange_epoch(
                K3MultiRangeBackwardPhase::W2Wgrad,
                backward_ranges.epoch_seed(first_range_index));
            while (ptx::ld_acq(first_state + kReadyEpochWord) !=
                   first_epoch) {
                __nanosleep(64);
            }
            if (backward_ranges.num_ranges > 1u) {
                const uint32_t second_epoch = k3_multirange_epoch(
                    K3MultiRangeBackwardPhase::W2Wgrad,
                    backward_ranges.epoch_seed(second_range_index));
                while (ptx::ld_acq(second_state + kReadyEpochWord) !=
                       second_epoch) {
                    __nanosleep(64);
                }
            } else {
                // Preserve one provider/body specialization for one- and
                // two-range launches. The unused physical arena supplies an
                // immutable empty second prefix; no live range state aliases it.
                for (uint32_t word = 0u;
                     word < kReadyStateWords; ++word) {
                    second_state[word] = 0u;
                }
                const uint32_t empty_pool_block =
                    first_state[kReadyPoolPrefixWord + kNumExperts];
                for (uint32_t expert = 0u;
                     expert <= kNumExperts; ++expert) {
                    second_state[kReadyPoolPrefixWord + expert] =
                        empty_pool_block;
                }
            }
            if (backward_ranges.num_ranges > 2u) {
                const uint32_t third_epoch = k3_multirange_epoch(
                    K3MultiRangeBackwardPhase::W2Wgrad,
                    backward_ranges.epoch_seed(third_range_index));
                while (ptx::ld_acq(third_state + kReadyEpochWord) !=
                       third_epoch) {
                    __nanosleep(64);
                }
            }

            for (uint32_t word = 0u; word < kReadyStateWords; ++word)
                union_state[word] = 0u;
            for (uint32_t expert = 0u;
                 expert <= kNumExperts; ++expert) {
                union_state[kReadyPoolPrefixWord + expert] =
                    first_state[kReadyPoolPrefixWord + expert];
            }
            const uint32_t active_count =
                backward_ranges.num_ranges == 3u
                ? sched::external_k_grouped_build_three_segment_union<
                      kNumExperts,
                      kReadyPoolPrefixWord,
                      kReadyActiveExpertWord>(
                          first_state, second_state, third_state,
                          union_state)
                : sched::external_k_grouped_build_two_segment_union<
                      kNumExperts,
                      kReadyPoolPrefixWord,
                      kReadyActiveExpertWord>(
                          first_state, second_state, union_state);

            union_state[kReadyMagicWord] = 0x4b335455u; // "K3TU"
            union_state[kReadyDW2CursorWord] = 0u;
            union_state[kReadyDW13CursorWord] = 0u;
            union_state[kReadyActiveCountWord] = active_count;
            union_state[kReadyPoolBlocksWord] =
                backward_ranges.total_pool_rows / BLOCK_M;
            union_state[kReadyDW2TasksWord] =
                active_count * kReadyDW2TasksPerExpert;
            union_state[kReadyDW13TasksWord] =
                active_count * kReadyDW13TasksPerExpert;
            for (uint32_t cluster = 0u;
                 cluster < kReadyNumClusters; ++cluster) {
                union_state[kReadyDW2ClusterSlotWord +
                            cluster * kReadyClusterSlotWords] =
                    kReadyCompleteRoleMask;
                union_state[kReadyDW13ClusterSlotWord +
                            cluster * kReadyClusterSlotWords] =
                    kReadyCompleteRoleMask;
            }

            // Publish the immutable prefixes and union state before the
            // following full-grid edge. That edge prevents any consumer from
            // accepting a same-valued stale epoch from an earlier launch and
            // separately publishes every CTA's BF16 operand stores to TMA.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
            const uint32_t union_epoch =
                ready_terminal_wgrad_epoch();
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(union_state + kReadyEpochWord),
                   "r"(union_epoch)
                : "memory");
        }

        constexpr bool kClearTwoSegmentEmptyExpertsEarly =
            kK3TwoSegmentBF16ProgressiveWgrad ||
            (kK3BranchMajorBF16DynamicTail && kMultiRangeBackward);
        if (backward_ranges.num_ranges == 3u ||
            kClearTwoSegmentEmptyExpertsEarly) {
            // Empty union experts have no W13 dgrad reader in either segment,
            // so their aliased weight-cache slice is writable before W13
            // starts. Clear it here, under the existing state-publication grid
            // edge, instead of inserting two terminal full-grid fences between
            // the persistent dW2 and dW13 bodies.
            DG_DEVICE_ASSERT(backward_ranges.num_ranges >= 2u);
            if (clear_empty_wgrad_expert_outputs) {
                constexpr uint64_t kBF16PerVector =
                    sizeof(uint4) / sizeof(cd_dtype_t);
                constexpr uint64_t kW13VectorsPerExpert =
                    static_cast<uint64_t>(2u * kIntermediateHidden) *
                    kHidden / kBF16PerVector;
                auto* const w13_vectors =
                    reinterpret_cast<uint4*>(w13_dequant_scratch);
                const uint4 zero = {0u, 0u, 0u, 0u};
                const uint64_t global_thread =
                    static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                    threadIdx.x;
                constexpr uint64_t kGlobalThreads =
                    static_cast<uint64_t>(kNumSMs) * kNumThreads;
                #pragma unroll 1
                for (uint32_t expert = 0u;
                     expert < kNumExperts; ++expert) {
                    bool active_in_union = false;
                    #pragma unroll
                    for (uint32_t range_idx = 0u;
                         range_idx < backward_ranges.num_ranges;
                         ++range_idx) {
                        active_in_union |= __ldg(
                            expert_counts +
                            backward_ranges.expert_counts_begin(
                                range_idx, kNumExperts) + expert) != 0;
                    }
                    if (active_in_union)
                        continue;
                    for (uint64_t linear = global_thread;
                         linear < kW13VectorsPerExpert;
                         linear += kGlobalThreads) {
                        w13_vectors[
                            static_cast<uint64_t>(expert) *
                                kW13VectorsPerExpert + linear] = zero;
                    }
                }
            }
            // Every generic writer publishes to the async/TMA proxy before
            // phase 14 makes the union state and early empty-expert stores
            // visible. Progressive consumers may arrive immediately; the
            // terminal consumer additionally waits for the post-dgrad edge.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
            __syncthreads();
        }
      }
    };

#endif

    // This walk normally clears the two W2-wgrad operands. Once W2 has
    // retired it can also clear an aliased late exact-X row at no extra
    // traversal cost.  The exact overlap specialization additionally clears
    // the future dW13 A operand here, before suffix clusters are allowed to
    // leave the W13 schedule; replaying that clear after dW2 quantization would
    // corrupt the aliased FP8 value planes.
    const auto clear_w2_and_late_x_padding_rows = [&]() {
        constexpr uint32_t kPaddingColumns =
            kHidden + kIntermediateHidden +
            (kK3MxFp8ThreeTermWgrad
                 ? 2u * kIntermediateHidden
                 : 0u);
        uint32_t pad_global_block = 0;
        #pragma unroll 1
        for (uint32_t range_idx = 0u;
             range_idx < (kMultiRangeBackward
                 ? backward_ranges.num_ranges : 1u);
             ++range_idx) {
            const auto& range = backward_ranges.ranges[range_idx];
            const int* const range_expert_counts = kMultiRangeBackward
                ? expert_counts + backward_ranges.expert_counts_begin(
                    range_idx, kNumExperts)
                : expert_counts;
            uint32_t pad_pool_block_offset = kMultiRangeBackward
                ? range.pool_row_begin / BLOCK_M : 0u;
            #pragma unroll 1
            for (uint32_t expert_idx = 0;
                 expert_idx < kNumExperts; ++expert_idx) {
                const uint32_t num_tokens = static_cast<uint32_t>(
                    __ldg(range_expert_counts + expert_idx));
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
                                     kPaddingColumns;
                             linear += kNumThreads) {
                            const uint32_t row_delta =
                                linear / kPaddingColumns;
                            const uint32_t col =
                                linear - row_delta * kPaddingColumns;
                            const uint32_t pool_row =
                                pool_block * BLOCK_M +
                                last_valid + row_delta;
                            if (col < kHidden) {
                                grad_ye_output[
                                    static_cast<uint64_t>(pool_row) *
                                        kHidden + col] = cd_dtype_t(0.0f);
                                if (late_exact_source_x) {
                                    // W2 has retired globally before this
                                    // helper runs, so exact-X padding may
                                    // safely replace its compact alias.
                                    x_pool_output[
                                        static_cast<uint64_t>(pool_row) *
                                            kHidden + col] =
                                        cd_dtype_t(0.0f);
                                }
                            } else if (
                                    col < kHidden + kIntermediateHidden) {
                                h_weighted_output[
                                    static_cast<uint64_t>(pool_row) *
                                        kIntermediateHidden +
                                    col - kHidden] = cd_dtype_t(0.0f);
                            } else {
                                grad_gate_up_output[
                                    static_cast<uint64_t>(pool_row) *
                                        (2u * kIntermediateHidden) +
                                    col - kHidden - kIntermediateHidden] =
                                    cd_dtype_t(0.0f);
                            }
                        }
                    }
                    ++pad_global_block;
                }
                pad_pool_block_offset += num_blocks;
            }
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
        full_grid_phase_barrier(kTraceSiteCount, 2);
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
        if constexpr (!kConcurrentResidualWeightCache) {
            if (warp_idx == 0 && cute::elect_one_sync()) {
                weight_load_barrier->init(1);
                cutlass::arch::fence_barrier_init();
            }
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

            constexpr uint32_t kBuilderCTAs =
                kConcurrentResidualWeightCache
                    ? kNumWeightProducerCTAs
                    : kNumSMs;
            for (uint64_t tile_idx = blockIdx.x;
                 tile_idx < num_tiles;
                 tile_idx += kBuilderCTAs) {
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
                    uint16_t weight_bf16_bits[kValuesPerLane];
                    uint32_t weight_amax_bits = 0;
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
                        const uint32_t source_scale_bits =
                            *reinterpret_cast<const uint32_t*>(
                                &builder_scale_source[
                                source_chunk *
                                    (kBuilderScaleChunkBytes /
                                     sizeof(float)) +
                                (local_k + i) *
                                    (kBuilderTmaN / kGranK) +
                                source_n / kGranK]);
                        const uint8_t nibble =
                            (packed >> ((global_n & 1u) * 4)) & 0xf;
                        weight_bf16_bits[i] =
                            k3_mxfp4_bf16_bits(
                                nibble, source_scale_bits);
                        weight_amax_bits = cute::max(
                            weight_amax_bits,
                            static_cast<uint32_t>(
                                weight_bf16_bits[i] & 0x7fff));
                    }
                    #pragma unroll
                    for (uint32_t offset = 4;
                         offset > 0; offset >>= 1)
                        weight_amax_bits = cute::max(
                            weight_amax_bits,
                            __shfl_xor_sync(
                                0xffffffff,
                                weight_amax_bits, offset, 8));
                    const uint32_t scale_byte = cute::max(
                        88,
                        static_cast<int>(
                            (weight_amax_bits >> 7) & 0xff) - 8);
                    const uint32_t quantized =
                        k3_quantize_bf16x4_e4m3(
                            weight_bf16_bits, scale_byte);
                    const uint32_t global_k =
                        k_block_idx * DGRAD_BLOCK_K + local_k;
                    const uint64_t row =
                        static_cast<uint64_t>(expert_idx) * output_mn +
                        global_n;
                    *reinterpret_cast<uint32_t*>(
                        q + row * output_k + global_k) = quantized;

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
                // Every thread above owns disjoint compact-cache words. Make
                // all normal stores visible before a single thread publishes
                // the tile epoch consumed by the dgrad TMA loader.
                asm volatile(
                    "fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                if constexpr (kConcurrentResidualWeightCache) {
                    if (threadIdx.x == 0) {
                        const uint32_t weight_tile_idx =
                            (expert_idx * num_k_blocks + k_block_idx) *
                                num_n_blocks +
                            n_block_idx;
                        const uint32_t state_offset =
                            w13 ? kNumW2WeightTileStates : 0;
                        const uint32_t ready_epoch =
                            w13
                                ? (launch_epoch ^ 0x80000000u)
                                : launch_epoch;
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(weight_tile_states + state_offset +
                                   weight_tile_idx),
                               "r"(ready_epoch)
                            : "memory");
                    }
                    __syncthreads();
                }
                weight_source_phase ^= 1;
            }
        };

        transform_operand(false);
        transform_operand(true);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        if constexpr (!kConcurrentResidualWeightCache) {
            if (warp_idx == 0 && cute::elect_one_sync()) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        weight_load_barrier));
            }
        }
        __syncthreads();
        if constexpr (!kConcurrentResidualWeightCache)
            full_grid_phase_barrier(kTraceSiteCount, 3);
      }
    };

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
        full_grid_phase_barrier(kTraceSiteCount, 4);

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
        full_grid_phase_barrier(kTraceSiteCount, 5);
      }
    };

    // One idle warp owns a complete row. It first reads every interleaved SiTU
    // derivative needed by the primary+residual group-32 encodings, then (only
    // for the allocation-free first chunk) performs the in-place gate/up
    // deinterleave as disjoint eight-BF16 permutation cycles. No shared-memory
    // staging or extra allocation is required, and no source element is
    // overwritten before this warp has finished all quantization reads.
    const auto quantize_w13_residual_row = [&] (
        const uint32_t pool_row,
        const bool valid_row) {
        constexpr uint32_t kWidth = 2 * kIntermediateHidden;
        constexpr uint32_t kKBlocks = kWidth / 128;
        constexpr uint32_t kValuesPerLane = 4;
        constexpr uint32_t kPrimaryRowStrideBytes = 2 * kHidden;
        constexpr uint64_t kW13WeightAliasValues =
            static_cast<uint64_t>(kNumExperts) * kHidden * kWidth;
        constexpr uint64_t kW13WeightAliasBytes =
            kW13WeightAliasValues + kW13WeightAliasValues / kGranK;
        auto* primary = reinterpret_cast<uint8_t*>(grad_ye_output);
        auto* residual = reinterpret_cast<uint8_t*>(grad_h_output);
        auto* primary_sf = reinterpret_cast<uint32_t*>(
            reinterpret_cast<uint8_t*>(w13_dequant_scratch) +
            kW13WeightAliasBytes);
        auto* residual_sf = primary_sf +
            static_cast<uint64_t>(residual_sf_rows) * kKBlocks;
        const uint32_t group_idx = lane_idx / 8;
        const uint32_t lane_in_group = lane_idx % 8;
        const uint64_t source_row_base =
            static_cast<uint64_t>(pool_row) * kWidth;

        // SF_BLOCK_M rounds each 192-row K3 pool block to 256 scale rows.
        // Initialize the 64 layout-only rows once per block; unlike value
        // padding, no row task naturally owns these permuted SF positions.
        if (pool_row % BLOCK_M == 0) {
            for (uint32_t padding_m = BLOCK_M + lane_idx;
                 padding_m < SF_BLOCK_M; padding_m += 32) {
                const uint32_t sf_m =
                    (pool_row / BLOCK_M) * SF_BLOCK_M +
                    (padding_m & ~127u) +
                    (padding_m & 31u) * 4 +
                    ((padding_m >> 5) & 3u);
                #pragma unroll 1
                for (uint32_t k_block_idx = 0;
                     k_block_idx < kKBlocks; ++k_block_idx) {
                    const uint64_t sf_offset = sf_m +
                        static_cast<uint64_t>(k_block_idx) *
                            residual_sf_rows;
                    primary_sf[sf_offset] = 0x7f7f7f7fu;
                    residual_sf[sf_offset] = 0x7f7f7f7fu;
                }
            }
        }

        #pragma unroll 1
        for (uint32_t k_block_idx = 0;
             k_block_idx < kKBlocks; ++k_block_idx) {
            const uint32_t global_k = k_block_idx * 128 +
                group_idx * 32 + lane_in_group * kValuesPerLane;
            const uint64_t primary_q_offset =
                static_cast<uint64_t>(pool_row) *
                    kPrimaryRowStrideBytes + global_k;
            const uint64_t residual_q_offset =
                static_cast<uint64_t>(pool_row) * kWidth + global_k;
            if (!valid_row) {
                *reinterpret_cast<uint32_t*>(
                    primary + primary_q_offset) = 0;
                *reinterpret_cast<uint32_t*>(
                    residual + residual_q_offset) = 0;
                if (lane_idx == 0) {
                    const uint32_t sf_m =
                        (pool_row / BLOCK_M) * SF_BLOCK_M +
                        ((pool_row % BLOCK_M) & ~127u) +
                        ((pool_row % BLOCK_M) & 31u) * 4 +
                        (((pool_row % BLOCK_M) >> 5) & 3u);
                    const uint64_t sf_offset = sf_m +
                        static_cast<uint64_t>(k_block_idx) *
                            residual_sf_rows;
                    primary_sf[sf_offset] = 0x7f7f7f7fu;
                    residual_sf[sf_offset] = 0x7f7f7f7fu;
                }
                continue;
            }

            float values[kValuesPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerLane; ++i) {
                const uint32_t conventional_col = global_k + i;
                if (inplace_interleaved_gate_up_grad) {
                    const uint32_t hidden_col =
                        conventional_col < kIntermediateHidden
                        ? conventional_col
                        : conventional_col - kIntermediateHidden;
                    const uint32_t source_col =
                        (hidden_col / 8) * 16 +
                        (conventional_col >= kIntermediateHidden ? 8 : 0) +
                        (hidden_col & 7);
                    values[i] = static_cast<float>(
                        grad_gate_up_output[
                            source_row_base + source_col]);
                } else {
                    values[i] = static_cast<float>(
                        grad_gate_up_output[
                            source_row_base + conventional_col]);
                }
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
            *reinterpret_cast<uint32_t*>(
                primary + primary_q_offset) = primary_q.__x;
            *reinterpret_cast<uint32_t*>(
                residual + residual_q_offset) = residual_q.__x;

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
                const uint32_t local_m = pool_row % BLOCK_M;
                const uint32_t sf_m =
                    (pool_row / BLOCK_M) * SF_BLOCK_M +
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
        __syncwarp();

        if (inplace_interleaved_gate_up_grad) {
            constexpr uint32_t kValuesPerPermutationBlock = 8;
            constexpr uint32_t kBranchBlocks =
                kIntermediateHidden / kValuesPerPermutationBlock;
            constexpr uint32_t kNumPermutationBlocks = 2 * kBranchBlocks;
            const auto destination_block = [] (const uint32_t source_block) {
                return (source_block & 1u) * kBranchBlocks +
                    (source_block >> 1);
            };
            // Enumerating candidate leaders is cheap for K3 (the early-exit
            // walk averages under eight integer steps at I=3072) and avoids a
            // visited bitmap. Each 8-lane group rotates disjoint cycles.
            for (uint32_t candidate = group_idx;
                 candidate < kNumPermutationBlocks;
                 candidate += 4) {
                uint32_t cursor = destination_block(candidate);
                bool is_cycle_leader = true;
                while (cursor != candidate) {
                    if (cursor < candidate) {
                        is_cycle_leader = false;
                        break;
                    }
                    cursor = destination_block(cursor);
                }
                if (!is_cycle_leader)
                    continue;
                const uint64_t candidate_offset =
                    source_row_base +
                    static_cast<uint64_t>(candidate) *
                        kValuesPerPermutationBlock + lane_in_group;
                cd_dtype_t carry =
                    grad_gate_up_output[candidate_offset];
                cursor = candidate;
                do {
                    const uint32_t dst_block =
                        destination_block(cursor);
                    const uint64_t dst_offset =
                        source_row_base +
                        static_cast<uint64_t>(dst_block) *
                            kValuesPerPermutationBlock + lane_in_group;
                    const cd_dtype_t next =
                        grad_gate_up_output[dst_offset];
                    grad_gate_up_output[dst_offset] = carry;
                    carry = next;
                    cursor = dst_block;
                } while (cursor != candidate);
            }
            __syncwarp();
        }
    };

    // Initialize the sparse scale planes before either post-dW2 producer
    // starts.  Keeping this grid-wide publication outside the split roles is
    // what lets the weight and activation builders run independently: no
    // partial-warp role ever enters a CTA-wide barrier.
    const auto prepare_w13_residual_act_scales_once = [&]() {
      if constexpr (kBuildW13ResidualActsOnce) {
        constexpr uint32_t kWidth = 2 * kIntermediateHidden;
        constexpr uint32_t kKBlocks = kWidth / 128;
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
        // The co-scheduled phase assigns some of these writers to the W13
        // weight role, so they never execute the activation producer's tail
        // fence. Publish every writer's generic global stores to the async
        // proxy before either role can run and before W13 TMA readers start.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        full_grid_phase_barrier(kTraceSiteCount, 6);
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
        // Warp 0 drives weight TMA and warps 4..19 convert weights. Compact
        // the complementary warp IDs so activation tasks remain dense and
        // exactly-once despite the non-contiguous physical role assignment.
        constexpr uint32_t kActivationWarpsPerCTA =
            kCoSchedulePostDW2W13Builders ? 15 : kWarpsPerCTA;
        constexpr uint32_t kActivationThreads =
            kActivationWarpsPerCTA * 32;
        constexpr uint32_t kGlobalWarps =
            kNumSMs * kActivationWarpsPerCTA;
        const uint32_t activation_warp_idx =
            !kCoSchedulePostDW2W13Builders
            ? warp_idx
            : (warp_idx >= 1 && warp_idx <= 3)
                  ? warp_idx - 1
                  : (warp_idx >= 20 ? warp_idx - 17 : 0);
        const uint32_t activation_thread_idx =
            activation_warp_idx * 32 + lane_idx;
        const auto activation_role_sync = [&]() {
            if constexpr (kCoSchedulePostDW2W13Builders) {
                cutlass::arch::NamedBarrier::sync(
                    kActivationThreads,
                    kPostDW2ActivationBuilderBarrier);
            } else {
                __syncthreads();
            }
        };
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
        if constexpr (!kCoSchedulePostDW2W13Builders)
            prepare_w13_residual_act_scales_once();

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

        if (inplace_interleaved_gate_up_grad) {
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
                    for (uint32_t col = activation_thread_idx;
                         col < kWidth;
                         col += kActivationThreads)
                        row_staging[col] =
                            grad_gate_up_output[row_base + col];
                }
                activation_role_sync();

                for (uint32_t k_block_idx = activation_warp_idx;
                     k_block_idx < kKBlocks;
                     k_block_idx += kActivationWarpsPerCTA) {
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
                activation_role_sync();
            }
        } else {
            const uint32_t global_warp_idx =
                blockIdx.x * kActivationWarpsPerCTA +
                activation_warp_idx;
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
        if constexpr (!kCoSchedulePostDW2W13Builders) {
            __syncthreads();
            full_grid_phase_barrier(kTraceSiteCount, 7);
        }
      }
    };

    const auto restore_w13_bf16_acts_once = [&]() {
      if constexpr (kBuildW13ResidualActsOnce) {
        if (!inplace_interleaved_gate_up_grad)
            return;
        full_grid_phase_barrier(kTraceSiteCount, 8);
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
        full_grid_phase_barrier(kTraceSiteCount, 9);
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
            kOverlapInitialBF16WeightDequant
            ? dequant_barriers + static_cast<uint32_t>(dequant_w13)
            : phase_ordered_dequant
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
        constexpr uint32_t kDequantScratchBytes =
            kDequantWeightBytes + 2 * kDequantScaleBytes;
        constexpr uint32_t kGemmScratchBytes =
            SMEM_CD_SIZE +
            kNumStages *
                (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE +
                 SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE) +
            kNumStages *
                (SMEM_RESIDUAL_A_SIZE_PER_STAGE +
                 SMEM_RESIDUAL_SFA_SIZE_PER_STAGE) +
            SMEM_DGRAD_WEIGHT_SOURCE_SIZE +
            SMEM_DGRAD_WEIGHT_SCALE_SOURCE_SIZE;
        DG_STATIC_ASSERT(
            !kOverlapInitialBF16WeightDequant ||
                kGemmScratchBytes >= 2 * kDequantScratchBytes,
            "Initial W2/W13 dequant staging exceeds the dead replay arena");
        constexpr uint32_t kNumDequantKTiles =
            kHidden / kDequantTileK;
        constexpr uint32_t kNumDequantNTiles =
            kIntermediateHidden / kDequantTileN;
        constexpr uint32_t kNumDequantTiles =
            kNumExperts * kNumDequantKTiles *
            kNumDequantNTiles;
        const uint32_t dequant_thread_idx =
            kOverlapInitialBF16WeightDequant
            ? dequant_w13
                ? (warp_idx - kInitialW13DequantWarpStart) * 32 +
                    lane_idx
                : (warp_idx < kInitialW2DequantFirstEnd
                       ? warp_idx
                       : warp_idx -
                             (kInitialW2DequantSecondStart -
                              kInitialW2DequantFirstEnd)) *
                          32 +
                          lane_idx
            : threadIdx.x;
        const uint32_t dequant_num_threads =
            kOverlapInitialBF16WeightDequant
            ? dequant_w13
                ? kInitialW13DequantThreads
                : kInitialW2DequantThreads
            : kNumThreads;
        const uint32_t dequant_named_barrier =
            dequant_w13
            ? kInitialW13DequantNamedBarrier
            : kInitialW2DequantNamedBarrier;
        auto* dequant_smem_base =
            kOverlapInitialBF16WeightDequant
            ? smem_gemm_base +
                static_cast<uint32_t>(dequant_w13) *
                    kDequantScratchBytes
            : smem_buffer;
        const auto dequant_sync = [&]() {
            if constexpr (kOverlapInitialBF16WeightDequant) {
                cutlass::arch::NamedBarrier::sync(
                    dequant_num_threads,
                    dequant_named_barrier);
            } else {
                __syncthreads();
            }
        };
        auto* dequant_weights =
            reinterpret_cast<int8_t*>(dequant_smem_base);
        auto* dequant_scales =
            reinterpret_cast<float*>(
                dequant_smem_base + kDequantWeightBytes);
        auto* dequant_scale_half2 =
            reinterpret_cast<uint32_t*>(
                dequant_smem_base + kDequantWeightBytes +
                kDequantScaleBytes);
        if (initialize_barriers) {
            if constexpr (kOverlapInitialBF16WeightDequant) {
                dequant_sync();
                if (dequant_thread_idx == 0) {
                    dequant_barrier->init(1);
                    cutlass::arch::fence_barrier_init();
                }
                dequant_sync();
            } else {
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

            if (dequant_thread_idx == 0) {
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
            dequant_sync();

            for (uint32_t scale_idx = dequant_thread_idx;
                 scale_idx < kDequantSFsPerTile;
                 scale_idx += dequant_num_threads) {
                const auto scale_half2 =
                    __float2half2_rn(
                        dequant_scales[scale_idx]);
                dequant_scale_half2[scale_idx] =
                    *reinterpret_cast<const uint32_t*>(
                        &scale_half2);
            }
            dequant_sync();

            for (uint32_t pair_idx = dequant_thread_idx;
                 pair_idx < kDequantPairsPerTile;
                 pair_idx += dequant_num_threads) {
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
            dequant_sync();
            if (dequant_thread_idx <
                kDequantTileK / DGRAD_BLOCK_K) {
                const uint32_t dgrad_k_block_idx =
                    k_tile_idx *
                        (kDequantTileK /
                         DGRAD_BLOCK_K) +
                    dequant_thread_idx;
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
            dequant_sync();
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

                if (dequant_thread_idx == 0) {
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
                dequant_sync();

                for (uint32_t scale_idx = dequant_thread_idx;
                     scale_idx < kW13DequantSFsPerTile;
                     scale_idx += dequant_num_threads) {
                    const auto scale_half2 =
                        __float2half2_rn(
                            dequant_scales[scale_idx]);
                    dequant_scale_half2[scale_idx] =
                        *reinterpret_cast<
                            const uint32_t*>(
                            &scale_half2);
                }
                dequant_sync();

                for (uint32_t pair_idx = dequant_thread_idx;
                     pair_idx <
                         kW13DequantPairsPerTile;
                     pair_idx += dequant_num_threads) {
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
                dequant_sync();

                if (dequant_thread_idx <
                    kW13DequantTileK /
                        DGRAD_BLOCK_K) {
                    const uint32_t
                        dgrad_k_block_idx =
                            k_tile_idx *
                                (kW13DequantTileK /
                                 DGRAD_BLOCK_K) +
                            dequant_thread_idx;
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
                dequant_sync();
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
    if constexpr (!kOverlapInitialBF16WeightDequant) {
        if constexpr (!phase_ordered_w2_dequant) {
            dequant_noninline_weights_once(false, false, true);
            if constexpr (
                !phase_ordered_w13_dequant &&
                !kInlineW13WeightDequant)
                dequant_noninline_weights_once(true, false, false);
        } else if constexpr (
            !phase_ordered_w13_dequant &&
            !kInlineW13WeightDequant) {
            dequant_noninline_weights_once(true, false, true);
        }
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
            primary_mma_barrier->init(1);
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

    if constexpr (kOverlapInitialBF16WeightDequant) {
        // Materializers and v409's streaming dispatch leave the same cluster
        // boundary. Their first common join is the existing CTA phase barrier.
        const bool is_w2_dequant_warp =
            warp_idx < kInitialW2DequantFirstEnd ||
            (warp_idx >= kInitialW2DequantSecondStart &&
             warp_idx < kInitialW2DequantSecondEnd);
        if (is_w2_dequant_warp) {
            dequant_noninline_weights_once(false, false, true);
        } else if (warp_idx >= kInitialW13DequantWarpStart) {
            if constexpr (!kInlineW13WeightDequant)
                dequant_noninline_weights_once(true, false, true);
        }
    }

    // Every role walks this deterministic schedule independently.  Pool offsets
    // are prefixes of ceil(count/BLOCK_M), matching the forward MegaMoE layout.
    const auto for_each_block = [&](const auto& func) {
        uint32_t next_assigned_block = blockIdx.x;
        uint32_t global_block = 0;
        uint32_t pool_block_offset = active_pool_block_begin;
        #pragma unroll 1
        for (uint32_t expert_idx = 0; expert_idx < kNumExperts; ++expert_idx) {
            const uint32_t num_tokens =
                static_cast<uint32_t>(
                    __ldg(active_expert_counts + expert_idx));
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
        // Prepared K3 compiles this replay schedule to zero iterations.  Do
        // not execute its legacy per-warp register transition: setmaxnreg is
        // warpgroup-aligned and every warp in the group must execute the same
        // static instruction.  Non-prepared specializations retain the
        // original producer budget for their live replay work.
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumProducerRegisters>();
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
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumProducerRegisters>();
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
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumProducerRegisters>();
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
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumProducerRegisters>();
    } else if (warp_idx < kReplayEpilogueWarpEnd) {
        // The prepared replay epilogue is empty and this specialization uses
        // no register above the kernel's ordinary budget.  Skipping the
        // otherwise empty 208-register residency avoids an unsynchronized
        // alloc/dealloc pair while leaving all live replay paths unchanged.
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_alloc<
                kNumEpilogueRegisters>();
        const uint32_t allocated_tmem =
            ptx::ld_shared(tmem_ptr_in_smem);
        if (allocated_tmem != 0) {
            // Keep this device-side assertion independent of printf: the CUDA
            // diagnostic path itself reports an illegal instruction under
            // compute-sanitizer on SM103 and hides the original source site.
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
    // Reverse dispatch owns the per-range symmetric-memory pull and the
    // one-time register transition into W2.  Multi-range calls invoke the
    // arithmetic twice, but every warp may execute a setmaxnreg transition
    // only on the first iteration.
    const auto prepare_w2_range_roles = [&]() {
    if (
        warp_idx >= kDispatchWarpStart &&
        warp_idx < kDispatchWarpStart + kNumDispatchWarps) {
        // The 1024-thread dgrad launch already reserves extra warps. Reuse one
        // warpgroup as the third role instead of increasing the launch size.
        constexpr uint32_t kNumDispatchRegisters =
            kBF16Mode ? 56 :
            (kPipelinedGradYDispatch ? 64 : 48);
        // Prepared K3 never acquires the compile-empty replay allocation. Its
        // exact non-residual specialization also compiles to a 40-register
        // maximum, so a legacy `dealloc<48>` would itself be outside the legal
        // temporal-register range. Non-prepared paths retain their transition.
        if constexpr (!kGateUpPrepared || kResidualMXFP8Dgrad)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumDispatchRegisters>();
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

            if constexpr (kPipelinedGradYDispatch) {
                const uint32_t num_counter_blocks =
                    active_num_acts_rows / BLOCK_M;
                const uint32_t counter_block_end =
                    active_pool_block_begin + num_counter_blocks;
                // acts_sf is laid out [scale-group, sf-row], so stride(1)
                // is the number of rows backing each hidden scale group.
                const uint64_t acts_sf_capacity_words =
                    static_cast<uint64_t>(acts_sf_stride) *
                    (kHidden / 128);
                const bool invalid_counter_storage =
                    kExactBF16PipelinedGradYDispatch
                    ? (acts_sf_ptr == nullptr ||
                       counter_block_end > acts_sf_capacity_words)
                    : (counter_block_end >
                       kNumW13WeightTileStates);
                if (active_num_acts_rows % BLOCK_M != 0 ||
                    invalid_counter_storage) {
                    if (dispatch_thread_idx == 0)
                        asm volatile("trap;");
                }
                for (uint32_t local_block =
                         blockIdx.x * kNumDispatchThreads +
                         dispatch_thread_idx;
                     local_block < num_counter_blocks;
                     local_block += kNumSMs * kNumDispatchThreads) {
                    grad_y_block_ready[
                        active_pool_block_begin + local_block] =
                        active_grad_y_counter_base;
                }
                __threadfence();
            }

            // All ranks stage their local BF16 grad-y before launch. This
            // system-scope barrier publishes those stores before remote TMA.
            if ((!kMultiRangeBackward || active_range_iteration == 0u) &&
                !kK3MxFp8ExactEpilogueRing) {
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
            }

          if constexpr (!kPipelinedGradYDispatch) {
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
        }
    } else if (warp_idx >= 12) {
        // W13 wgrad needs the exact BF16 value represented by the forward
        // FP8+UE8M0 pool.  Produce it while the recompute MMA is running, using
        // otherwise-idle warps.  Padding rows are explicitly zeroed so the
        // k-grouped wgrad mainloop can round K up to 64 without reading the
        // following expert.
        constexpr uint32_t kNumXPoolRegisters =
            kBF16Mode ? 56 : 40;
        if (!kMultiRangeBackward || active_range_iteration == 0u)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumXPoolRegisters>();
        constexpr uint32_t kFirstXPoolWarp = 12;
        constexpr uint32_t kNumXPoolThreads =
            kNumThreads - kFirstXPoolWarp * 32;
        const uint32_t x_thread_idx =
            (warp_idx - kFirstXPoolWarp) * 32 + lane_idx;
        if constexpr (
            !(kBF16Mode && kDispatchInputsPrepared) &&
            !kExactBF16PipelinedGradYDispatch) {
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
                    if constexpr (kExactSourceX) {
                        // A CTA-local early clear can race another CTA's W2
                        // producer when exact-X aliases the compact dgrad
                        // planes. The post-W2 padding helper owns this row.
                        if (late_exact_source_x)
                            continue;
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
        if (!kMultiRangeBackward || active_range_iteration == 0u)
            cutlass::arch::warpgroup_reg_dealloc<
                kNumIdleRegisters>();
    }
    };

    const auto clear_and_publish_combine_planes = [&]() {
      if constexpr (
          kDirectRemoteGradX &&
          kCombineOrderMode == CombineOrderMode::FixedTopK) {
        if constexpr (kStreamingDirectGradXCombine) {
            // The retired source-index tail reserves one word per physical
            // source token. Use that full granularity so one slow route cannot
            // hold seven unrelated tokens behind the same readiness counter.
            const uint32_t num_source_tokens =
                K3DirectGradXReadyContract::num_counters(
                    backward_workspace.num_max_tokens_per_rank);
            for (uint32_t source_token =
                     blockIdx.x * kNumThreads + threadIdx.x;
                 source_token < num_source_tokens;
                 source_token += kNumSMs * kNumThreads) {
                direct_grad_x_ready_counts[source_token] = 0;
            }

            const uint32_t num_pool_blocks =
                kMultiRangeBackward
                ? backward_ranges.total_pool_rows / BLOCK_M
                : math::ceil_div(num_pool_rows, BLOCK_M);
            DG_DEVICE_ASSERT(
                kDirectGradXPoolCompletionWord + num_pool_blocks <=
                kNumW2WeightTileStates);
            for (uint32_t pool_block =
                     blockIdx.x * kNumThreads + threadIdx.x;
                 pool_block < num_pool_blocks;
                 pool_block += kNumSMs * kNumThreads) {
                direct_grad_x_pool_completions[pool_block] = 0u;
            }
        }

        // FixedTopK consumes every physical slot, including invalid routes.
        // The W13-overlap specialization preserves source-X plane one until
        // its dedicated warp retires every remote read. All other planes are
        // cleared and published here, so W13 may scatter those slots
        // immediately while slot one is staged in a dead local pool.
        auto* combine_buffer =
            const_cast<cd_dtype_t*>(backward_grad_y);
        const uint64_t values_per_plane =
            static_cast<uint64_t>(
                backward_workspace.num_max_tokens_per_rank) *
            kHidden;
        if constexpr (kOverlapExactSourceXPlaneOneWithW13) {
            DG_DEVICE_ASSERT(
                num_topk > 1u &&
                backward_x == combine_buffer + values_per_plane);
            for (uint32_t plane = 0u; plane < num_topk; ++plane) {
                if (plane == 1u)
                    continue;
                auto* const plane_buffer =
                    combine_buffer +
                    static_cast<uint64_t>(plane) * values_per_plane;
                for (uint64_t offset =
                         static_cast<uint64_t>(blockIdx.x) *
                             kNumThreads +
                         threadIdx.x;
                     offset < values_per_plane;
                     offset +=
                         static_cast<uint64_t>(kNumSMs) *
                         kNumThreads) {
                    plane_buffer[offset] = cd_dtype_t(0.0f);
                }
            }
        } else {
            const uint64_t num_plane_values =
                static_cast<uint64_t>(num_topk) * values_per_plane;
            for (uint64_t linear =
                     static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                     threadIdx.x;
                 linear < num_plane_values;
                 linear +=
                     static_cast<uint64_t>(kNumSMs) * kNumThreads) {
                combine_buffer[linear] = cd_dtype_t(0.0f);
            }
        }
        if constexpr (kNumRanks > 1) {
            constexpr uint32_t kAfterGradYClearGridSyncIndex = 3;
            constexpr uint32_t kAfterGradYClearBarrierTag = 8;
            trace_begin(8);
            comm::nvlink_barrier<
                kNumRanks, kNumSMs, kNumThreads,
                kAfterGradYClearGridSyncIndex,
                kAfterGradYClearBarrierTag>(
                    backward_workspace, backward_sym_buffer,
                    blockIdx.x, threadIdx.x,
                    []() { __syncthreads(); });
            if constexpr (kStreamingDirectGradXCombine) {
                // Initialize each source token to minus its exact number of
                // valid routes. Metadata contains only routed rows, so masked
                // top-k slots need no special case. A second rank barrier keeps
                // these remote decrements ahead of every W13 publication.
                const uint32_t num_ranges =
                    kMultiRangeBackward
                    ? backward_ranges.num_ranges
                    : 1u;
                const uint32_t global_thread =
                    blockIdx.x * kNumThreads + threadIdx.x;
                constexpr uint32_t kGlobalThreads =
                    kNumSMs * kNumThreads;
                #pragma unroll 1
                for (uint32_t range_idx = 0u;
                     range_idx < num_ranges; ++range_idx) {
                    const uint32_t token_begin =
                        kMultiRangeBackward
                        ? backward_ranges.token_begin(range_idx)
                        : 0u;
                    uint32_t pool_block_offset =
                        kMultiRangeBackward
                        ? backward_ranges.ranges[range_idx]
                                  .pool_row_begin /
                              BLOCK_M
                        : 0u;
                    const auto* range_expert_counts =
                        kMultiRangeBackward
                        ? expert_counts +
                              backward_ranges.expert_counts_begin(
                                  range_idx, kNumExperts)
                        : expert_counts;
                    #pragma unroll 1
                    for (uint32_t expert_idx = 0u;
                         expert_idx < kNumExperts; ++expert_idx) {
                        const uint32_t num_tokens =
                            static_cast<uint32_t>(__ldg(
                                range_expert_counts + expert_idx));
                        for (uint32_t local_m = global_thread;
                             local_m < num_tokens;
                             local_m += kGlobalThreads) {
                            const uint32_t pool_row =
                                pool_block_offset * BLOCK_M + local_m;
                            const auto metadata =
                                token_src_metadata[pool_row];
                            const uint32_t source_token =
                                token_begin + metadata.token_idx;
                            DG_DEVICE_ASSERT(
                                source_token <
                                backward_workspace
                                    .num_max_tokens_per_rank);
                            auto* remote_ready =
                                backward_sym_buffer.map(
                                    direct_grad_x_ready_counts +
                                        K3DirectGradXReadyContract::
                                            counter_index(source_token),
                                    metadata.rank_idx);
                            ptx::red_add_rel_sys(remote_ready, -1);
                        }
                        pool_block_offset +=
                            math::ceil_div(num_tokens, BLOCK_M);
                    }
                }
                constexpr uint32_t kAfterReadyInitBarrierTag = 10;
                comm::nvlink_barrier<
                    kNumRanks, kNumSMs, kNumThreads,
                    kAfterGradYClearGridSyncIndex,
                    kAfterReadyInitBarrierTag>(
                        backward_workspace, backward_sym_buffer,
                        blockIdx.x, threadIdx.x,
                        []() { __syncthreads(); });
            }
            trace_end(8);
        }
      }
    };

    const auto prepare_direct_grad_x_planes = [&]() {
      if constexpr (kDirectRemoteGradX) {
        if constexpr (kNumRanks > 1) {
            // backward_grad_y aliases combine plane zero. All ranks must
            // finish pulling it before W13 publishes direct grad-x there.
            if constexpr (!(kBF16Mode && kDispatchInputsPrepared)) {
                constexpr uint32_t kBeforeDirectGradXGridSyncIndex = 2;
                constexpr uint32_t kBeforeDirectGradXBarrierTag = 7;
                trace_begin(7);
                comm::nvlink_barrier<
                    kNumRanks, kNumSMs, kNumThreads,
                    kBeforeDirectGradXGridSyncIndex,
                    kBeforeDirectGradXBarrierTag>(
                        backward_workspace, backward_sym_buffer,
                        blockIdx.x, threadIdx.x,
                        []() { __syncthreads(); });
                trace_end(7);
            }
        }
        clear_and_publish_combine_planes();
      }
    };

    {
        const uint32_t num_w2_ranges = kMultiRangeBackward
            ? backward_ranges.num_ranges
            : 1u;
        if constexpr (kMultiRangeBackward) {
            static_assert(
                !kMultiRangeBackward ||
                    DG_EXPERIMENTAL_K3_READY_WGRAD ||
                    kK3MxFp8WgradOverlap,
                "multi-range requires ready BF16 or exact MXFP8 wgrad");
            bool invalid_range_set =
                num_w2_ranges == 0u ||
                num_w2_ranges > kK3MaxBackwardRanges ||
                backward_ranges.total_backward_tokens !=
                    num_backward_tokens ||
                backward_ranges.total_pool_rows > num_pool_rows ||
                backward_ranges.total_acts_rows > num_acts_rows ||
                gate_up_output != grad_gate_up_output ||
                h_act_output != grad_h_output ||
                h_weighted_output != h_act_output ||
                x_pool_output !=
                    const_cast<cd_dtype_t*>(down_unweighted_output);
            uint32_t token_capacity_end = 0u;
            uint32_t token_count = 0u;
            uint32_t pool_end = 0u;
            uint32_t sf_pool_end = 0u;
            uint32_t previous_epoch = 0u;
            for (uint32_t range_idx = 0u;
                 range_idx < num_w2_ranges; ++range_idx) {
                const auto& range = backward_ranges.ranges[range_idx];
                const uint32_t epoch =
                    backward_ranges.epoch_seed(range_idx) & 0xffffu;
                invalid_range_set |=
                    range.pool_row_begin != pool_end ||
                    range.acts_row_begin != range.pool_row_begin ||
                    range.num_acts_rows != range.num_pool_rows ||
                    range.sf_pool_row_begin != sf_pool_end ||
                    range.num_tokens > range.max_tokens_per_rank ||
                    range.pool_row_begin % BLOCK_M != 0u ||
                    range.num_pool_rows % BLOCK_M != 0u ||
                    epoch == 0u || epoch == previous_epoch;
                token_count += range.num_tokens;
                token_capacity_end += range.max_tokens_per_rank;
                pool_end += range.num_pool_rows;
                sf_pool_end += range.num_sf_pool_rows;
                previous_epoch = epoch;
            }
            invalid_range_set |=
                token_count != backward_ranges.total_backward_tokens ||
                token_capacity_end >
                    backward_workspace.num_max_tokens_per_rank ||
                pool_end != backward_ranges.total_pool_rows ||
                pool_end != backward_ranges.total_acts_rows;
            if (invalid_range_set)
                asm volatile("trap;");
        }

        // Option-A assigns complete cluster pairs to disjoint persistent
        // roles.  The 132-CTA prefix continues the native W2 schedule while
        // the final 16 CTAs enter exact grouped dW13 immediately.  Keep the
        // branch outside the range loop: a tail CTA must never execute even
        // the first W2 role/barrier before becoming a ring consumer.
        const bool exact_ring_tail =
            kK3MxFp8ExactEpilogueRing &&
            static_cast<uint32_t>(blockIdx.x) >=
                kExactRingW2PrefixCTAs;
        if (!exact_ring_tail) {
          #pragma unroll 1
          for (active_range_iteration = 0u;
               active_range_iteration < num_w2_ranges;
               ++active_range_iteration) {
        if constexpr (kMultiRangeBackward) {
            active_range_index = backward_ranges.reverse_range_index(
                active_range_iteration);
            const auto& range = backward_ranges.ranges[active_range_index];
            active_token_begin = backward_ranges.token_begin(
                active_range_index);
            active_pool_row_begin = range.pool_row_begin;
            active_pool_block_begin = range.pool_row_begin / BLOCK_M;
            active_num_pool_rows = range.num_pool_rows;
            active_num_acts_rows = range.num_acts_rows;
            active_range_epoch = k3_multirange_epoch(
                K3MultiRangeBackwardPhase::W2Dgrad,
                backward_ranges.epoch_seed(active_range_index));
            active_accumulate_wgrad = active_range_iteration != 0u;
            active_expert_counts =
                expert_counts + backward_ranges.expert_counts_begin(
                    active_range_index, kNumExperts);
            active_grad_y_counter_base =
                active_range_epoch ^ 0x80000000u;
            active_grad_y_ready_value =
                active_grad_y_counter_base + BLOCK_M;
        }
        stage_idx = 0u;
        phase = 0u;
        prepare_w2_range_roles();
        __syncthreads();
        if constexpr (kMultiRangeBackward) {
            // The first range's NVLink barrier publishes the distributed
            // readiness-counter initialization. Later ranges skip that
            // network barrier, so join all CTAs before any producer can
            // increment a counter that a delayed initializer could overwrite.
            if (active_range_iteration != 0u) {
                if constexpr (kK3MxFp8ExactEpilogueRing) {
                    if (blockIdx.x < kExactRingW2PrefixCTAs) {
                        K3MxFp8WgradSubsetBarrier<
                            kExactRingW2PrefixCTAs>{
                                k3_mxfp8_dw13_hybrid_state() +
                                    K3MxFp8OverlapState::
                                        kDW2SubsetBarrierCount,
                                k3_mxfp8_dw13_hybrid_state() +
                                    K3MxFp8OverlapState::
                                        kDW2SubsetBarrierSense}();
                    }
                } else {
                    full_grid_phase_barrier(kTraceSiteCount, 29);
                }
            }
        }
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
            !kDispatchInputsPrepared &&
            !kPipelinedGradYDispatch) {
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
                full_grid_phase_barrier(6, 10);
            }
        }
        // Exact source X is staged in combine plane one. When x_pool aliases
        // grad-y, retain that plane until the W2/route readers release grad-y;
        // the late TMA pull below drains it immediately before W13 dgrad starts
        // publishing direct grad-x into the same symmetric planes.
        if (!late_exact_source_x && !kPipelinedGradYDispatch)
            prepare_direct_grad_x_planes();
        if constexpr (!kBF16Mode) {
            if (warp_idx >= kDispatchWarpStart &&
                warp_idx <
                    kDispatchWarpStart +
                        kNumDispatchWarps) {
                if constexpr (!kPipelinedGradYDispatch) {
                    // Legacy dispatch used 48 registers; transition down to
                    // the common dgrad epilogue budget with dealloc, not
                    // reg_alloc (allocating a lower count is illegal on SM100).
                    if (!kMultiRangeBackward ||
                        active_range_iteration == 0u)
                        cutlass::arch::warpgroup_reg_dealloc<40>();
                }
            } else if (warp_idx >= kDispatchWarpStart) {
                if (!kMultiRangeBackward ||
                    active_range_iteration == 0u)
                    cutlass::arch::warpgroup_reg_alloc<40>();
            }
        }
        const auto for_each_dgrad_block = [&](const auto& func) {
            if constexpr (kK3MxFp8ExactEpilogueRing) {
                // Option-A reserves the final eight complete clusters for the
                // early exact dW13 body.  Every W2 role must use the same
                // 132-CTA domain; remapping only dispatch or only dgrad would
                // silently leave physical rows or (M,N) tasks unowned.
                if (blockIdx.x >= kExactRingW2PrefixCTAs)
                    return;
            }
            if constexpr (kConcurrentResidualWeightCache) {
                // Keep this diagnostic control on v352's proven role split:
                // dedicated conversion CTAs do not enter the W2 UMMA schedule.
                if (is_weight_producer_cta)
                    return;
            }
            constexpr uint32_t kNumSchedulePasses =
                kFirstMWaveBuildsResidualWeightCache ? 2 : 1;
            // MoK publishes every weight producer before any dependent CLC
            // consumer. Preserve that dependency order here: pass zero owns
            // M=0 and converts each (expert,N,K) tile exactly once; pass one
            // schedules every remaining M tile against the released cache.
            #pragma unroll 1
            for (uint32_t schedule_pass = 0;
                 schedule_pass < kNumSchedulePasses;
                 ++schedule_pass) {
                uint32_t next_assigned_block =
                    kConcurrentResidualWeightCache
                        ? w2_consumer_block_idx
                        : blockIdx.x;
                uint32_t global_block = 0;
                uint32_t pool_block_offset =
                    active_pool_block_begin;
                // Cloning this full scheduler body for all 112 local K3
                // experts produces a multi-megabyte instruction image and
                // trashes L0/L1 I$.
                #pragma unroll 1
                for (uint32_t expert_idx = 0;
                     expert_idx < kNumExperts; ++expert_idx) {
                    const uint32_t num_tokens =
                        static_cast<uint32_t>(
                            __ldg(active_expert_counts + expert_idx));
                    const uint32_t num_m_blocks =
                        math::ceil_div(num_tokens, BLOCK_M);
                    const uint32_t first_m_block =
                        kFirstMWaveBuildsResidualWeightCache
                            ? schedule_pass
                            : 0;
                    const uint32_t scheduled_m_blocks =
                        schedule_pass == 0
                            ? cute::min(num_m_blocks, 1u)
                            : num_m_blocks -
                                  cute::min(num_m_blocks, 1u);
                    const uint32_t expert_blocks =
                        (kFirstMWaveBuildsResidualWeightCache
                             ? scheduled_m_blocks
                             : num_m_blocks) *
                        kNumDgradBlockNs;
                    const uint32_t expert_end =
                        global_block + expert_blocks;

                    while (next_assigned_block < global_block)
                        next_assigned_block +=
                            kConcurrentResidualWeightCache
                                ? kNumW2ConsumerCTAs
                                : (kK3MxFp8ExactEpilogueRing
                                       ? kExactRingW2PrefixCTAs
                                       : kNumSMs);
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
                        next_assigned_block +=
                            kConcurrentResidualWeightCache
                                ? kNumW2ConsumerCTAs
                                : (kK3MxFp8ExactEpilogueRing
                                       ? kExactRingW2PrefixCTAs
                                       : kNumSMs);
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
          if constexpr (
              kRouteGradBeforeW2 &&
              !kPipelinedGradYDispatch) {
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
            full_grid_phase_barrier(kTraceSiteCount, 11);
          }
        }

        // The route-gradient dot is complete. A multichunk caller may now
        // reuse the saved down pool as W2's BF16 dequant workspace;
        // single-chunk calls keep using dW2 storage.
        if constexpr (phase_ordered_w2_dequant) {
            dequant_noninline_weights_once(false, true, true);
            full_grid_phase_barrier(kTraceSiteCount, 12);
        }
        if constexpr (!kPipelinedGradYDispatch)
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
                        primary_mma_barrier));
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
                primary_mma_barrier->init(1);
                residual_mma_barrier->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(
                    2 * kNumW2DgradEpilogueThreads);
            }
            cutlass::arch::fence_barrier_init();
        }
        trace_begin(11);
        comm::cluster_sync_with_relaxed_arrive();
        trace_end(11);
        stage_idx = 0;
        phase = 0;
        if constexpr (kConcurrentResidualWeightCache) {
            if (is_weight_producer_cta) {
                // All launch-wide replay/NVLink handshakes are complete at
                // this point. Drain this producer CTA's share of the
                // readiness-driven grad-y dispatch before the CTA-wide
                // converter aliases its shared staging region. Consumer CTAs
                // retain the original overlap between dispatch and W2 UMMA.
                if (warp_idx >= kDispatchWarpStart &&
                    warp_idx <
                        kDispatchWarpStart + kNumDispatchWarps) {
                    run_pipelined_grad_y_dispatch();
                }
                __syncthreads();
                build_residual_weights_tiled_once();
            }
        }
        if (
            warp_idx == 0 ||
            (kResidualMXFP8Dgrad &&
             (!kBuildW2ResidualActsOnce ||
              kOnDemandResidualWeightCache) &&
             (warp_idx == 1 ||
              (warp_idx >= 3 &&
               warp_idx < kNumW2ResidualProducerWarps + 1)))) {
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
                    if constexpr (kPipelinedGradYDispatch) {
                      if (warp_idx == 0) {
                        if (lane_idx == 0) {
                            const uint64_t wait_start = clock64();
                            while (ptx::ld_acq(
                                       grad_y_block_ready +
                                       pool_block_idx) !=
                                   active_grad_y_ready_value) {
                                if (clock64() - wait_start >
                                    4000000000ull) {
                                    // The source-distinct trap preserves the
                                    // actual readiness timeout in line-info
                                    // builds without entering device vfprintf.
                                    asm volatile("trap;");
                                }
                            }
                        }
                        __syncwarp();
                      }
                    }
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
                                kNumW2ResidualProducerWarps;
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
                                build_weight_tile =
                                    kFirstMWaveBuildsResidualWeightCache &&
                                    m_block_idx == 0;
                                if (producer_warp_idx == 0 && lane_idx == 0 &&
                                    !build_weight_tile) {
                                    auto* state =
                                        weight_tile_states + weight_tile_idx;
                                    const uint64_t cache_wait_start = clock64();
                                    while (ptx::ld_acq(state) != launch_epoch) {
                                        if (clock64() - cache_wait_start >
                                            4000000000ull) {
                                            asm volatile("trap;");
                                        }
                                    }
                                }
                                cutlass::arch::NamedBarrier::sync(
                                    kNumW2ResidualProducerThreads,
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
                                     kNumW2ResidualProducerThreads) {
                                smem_sfa[stage_idx][sf_idx] =
                                    0x7f7f7f7fu;
                                smem_dgrad_sfa_residual[
                                    stage_idx][sf_idx] =
                                    0x7f7f7f7fu;
                            }
                            cutlass::arch::NamedBarrier::sync(
                                kNumW2ResidualProducerThreads,
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
                                kNumW2ResidualProducerThreads,
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
                                k3_mxfp4_to_mxfp8_transposed_tile_one_pass<
                                    kNumW2ResidualProducerWarps>(
                                    smem_dgrad_weight_source,
                                    smem_dgrad_weight_scale_source,
                                    weight_bytes,
                                    smem_sfb[stage_idx],
                                    n_block_idx,
                                    producer_warp_idx * 32 + lane_idx);
                            } else {
                            for (uint32_t local_n = producer_warp_idx;
                                 local_n < LOAD_BLOCK_N;
                                 local_n += kNumW2ResidualProducerWarps) {
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
                                }
                            }
                            }
                            cutlass::arch::fence_view_async_shared();
                            cutlass::arch::NamedBarrier::sync(
                                kNumW2ResidualProducerThreads,
                                kResidualWeightProducerBarrier);
                              if (producer_warp_idx == 0 &&
                                  cute::elect_one_sync()) {
                                  if constexpr (
                                      kOnDemandResidualWeightCache) {
                                      // The TMA store is the sole cache
                                      // publisher. Scalar lanes populate only
                                      // shared B/SFB above; duplicating their
                                      // writes in global memory wastes the
                                      // full transposed-weight bandwidth and
                                      // is unnecessary because the release
                                      // epoch follows tma_store_wait<0>().
                                      cute::tma_store_fence();
                                      cute::SM90_TMA_STORE_2D::copy(
                                          &tensor_map_w2_dgrad_weights,
                                          smem_dgrad_b[stage_idx],
                                          k_block_idx * DGRAD_BLOCK_K,
                                          expert_idx * kIntermediateHidden +
                                              n_block_idx * BLOCK_N);
                                      cute::SM90_TMA_STORE_2D::copy(
                                          &tensor_map_w2_dgrad_weights_sf,
                                          smem_sfb[stage_idx],
                                          n_block_idx * BLOCK_N,
                                          expert_idx *
                                                  (kHidden / (kGranK * 4)) +
                                              k_block_idx);
                                      cute::tma_store_arrive();
                                  }
                                  full_barriers[stage_idx]->arrive(0u);
                                  if constexpr (
                                      kOnDemandResidualWeightCache) {
                                      ptx::tma_store_wait<0>();
                                      asm volatile(
                                          "fence.proxy.async.global;"
                                          ::: "memory");
                                      asm volatile(
                                          "st.release.gpu.global.u32 [%0], %1;"
                                          :: "l"(weight_tile_states +
                                                 weight_tile_idx),
                                             "r"(launch_epoch)
                                          : "memory");
                                  }
                              }
                              if constexpr (
                                  kOnDemandResidualWeightCache) {
                                  cutlass::arch::NamedBarrier::sync(
                                      kNumW2ResidualProducerThreads,
                                      kResidualWeightCacheStoreDoneBarrier);
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
                        if constexpr (kConcurrentResidualWeightCache) {
                            if (lane_idx == 0) {
                                while (ptx::ld_acq(
                                           weight_tile_states +
                                           weight_tile_idx) !=
                                       launch_epoch) {
                                }
                            }
                            __syncwarp();
                        } else if constexpr (
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
                uint32_t primary_mma_phase = 0;
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
                            // Keep primary and correction completion on
                            // distinct barriers. Reusing one barrier lets two
                            // multicast arrivals advance the follower through
                            // two parity epochs before its first wait observes
                            // either completion.
                            cutlass::arch::umma_arrive_multicast_2x1SM(
                                reinterpret_cast<uint64_t*>(
                                    primary_mma_barrier),
                                kCTAMask);
                            primary_mma_barrier->wait(primary_mma_phase);
                            primary_mma_phase ^= 1;
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
                uint32_t follower_primary_mma_phase = 0;
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
                            primary_mma_barrier->wait(
                                follower_primary_mma_phase);
                            follower_primary_mma_phase ^= 1;
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
        } else if (
            kPipelinedGradYDispatch &&
            (!kConcurrentResidualWeightCache ||
             !is_weight_producer_cta) &&
            warp_idx >= kDispatchWarpStart &&
            warp_idx <
                kDispatchWarpStart +
                    kNumDispatchW2WeightBuilderWarps) {
            const bool is_dispatch_warp =
                warp_idx < kDispatchWarpStart + kNumDispatchWarps;
            if (is_dispatch_warp) {
                run_pipelined_grad_y_dispatch();
            } else if constexpr (kWideDispatchW2WeightBuilder) {
                // The borrowed epilogue warpgroup entered W2 at the common
                // 40-register budget. Raise it to the dispatch group's 64
                // registers before the retained-pair converter, then return
                // the registers before the W13 role map is restored.
                cutlass::arch::warpgroup_reg_alloc<64>();
            }
            // Trace site four is compile-time unused by the pipelined K3
            // dispatch path; reuse it to isolate the concurrent W2 builder.
            trace_begin(4);
            run_dispatch_w2_residual_weight_cache();
            trace_end(4);
            if constexpr (kWideDispatchW2WeightBuilder) {
                if (!is_dispatch_warp)
                    cutlass::arch::warpgroup_reg_dealloc<40>();
            }
        } else if (warp_idx >= kW2DgradEpilogueWarpStart) {
            const uint32_t epilogue_warp_idx =
                warp_idx - kW2DgradEpilogueWarpStart;
            const uint32_t epilogue_thread_idx =
                epilogue_warp_idx * 32 + lane_idx;
            uint32_t current_iter = 0;

            for_each_dgrad_block(
                [&](const uint32_t& expert_idx,
                    const uint32_t& pool_block_offset,
                    const uint32_t& m_block_idx,
                    const uint32_t& n_block_idx,
                    const uint32_t& valid_m) {
                    const uint32_t accum_stage =
                        current_iter % kNumEpilogueStages;
                    const uint32_t accum_phase =
                        (current_iter++ /
                         kNumEpilogueStages) &
                        1;
                    const uint32_t effective_m =
                        math::align(valid_m, 16u);
                    const auto prefetch_gate_up_segment = [&] (
                            const uint32_t segment) {
                        if constexpr (kExactBF16PipelinedGradYDispatch) {
                            if (lane_idx == 0u) {
                                for (uint32_t linear =
                                         epilogue_warp_idx * 32u;
                                     linear <
                                         STORE_BLOCK_M * BLOCK_N;
                                     linear +=
                                         kNumW2DgradEpilogueThreads) {
                                    const uint32_t row =
                                        linear / BLOCK_N;
                                    const uint32_t hidden_col =
                                        n_block_idx * BLOCK_N +
                                        linear - row * BLOCK_N;
                                    const uint32_t local_m =
                                        segment * STORE_BLOCK_M + row;
                                    if (local_m >= valid_m)
                                        continue;
                                    const uint32_t pool_row =
                                        (pool_block_offset + m_block_idx) *
                                            BLOCK_M +
                                        local_m;
                                    const uint64_t row_base =
                                        static_cast<uint64_t>(pool_row) *
                                        (2 * kIntermediateHidden);
                                    cute::prefetch(
                                        gate_up_output + row_base +
                                        hidden_col);
                                    cute::prefetch(
                                        gate_up_output + row_base +
                                        kIntermediateHidden + hidden_col);
                                }
                            }
                        }
                    };
                    prefetch_gate_up_segment(0u);
                    // Only the four warps that copy TMEM into SMEM need the
                    // completion wait. The all-epilogue named barrier below
                    // broadcasts their readiness before any SMEM consumer
                    // can advance.
                    if (epilogue_warp_idx <
                        kNumEpilogueThreads / 32) {
                        tmem_full_barriers[accum_stage]->wait(
                            accum_phase);
                        ptx::tcgen05_after_thread_sync();
                    }

                    for (uint32_t s = 0;
                         s < effective_m / STORE_BLOCK_M; ++s) {
                        cutlass::arch::NamedBarrier::sync(
                            kNumW2DgradEpilogueThreads, 0);
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
                                        smem_cd[s & 1u]) +
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
                        if (s + 1u < effective_m / STORE_BLOCK_M)
                            prefetch_gate_up_segment(s + 1u);
                        cutlass::arch::NamedBarrier::sync(
                            kNumW2DgradEpilogueThreads, 0);

                        #pragma unroll
                        for (uint32_t linear =
                                 epilogue_thread_idx;
                             linear <
                                 STORE_BLOCK_M * BLOCK_N;
                             linear +=
                                 kNumW2DgradEpilogueThreads) {
                            const uint32_t row =
                                linear / BLOCK_N;
                            const uint32_t n =
                                linear - row * BLOCK_N;
                            const uint32_t local_m =
                                s * STORE_BLOCK_M + row;
                            if (local_m >= valid_m) {
                                continue;
                            }

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
                                        smem_cd[s & 1u]) +
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
                            // POST_DOWN K3 reuses this pool for the SiTU
                            // activation that this same lane publishes below.
                            // Avoid issuing a global store that would be
                            // overwritten before any consumer can observe it.
                            // Keep the store for callers that provide distinct
                            // grad-h and activation pools.
                            if (kRouteWeightMode !=
                                    RouteWeightMode::PostDown ||
                                grad_h_output != h_act_output) {
                                grad_h_output[
                                    static_cast<uint64_t>(
                                        pool_row) *
                                        kIntermediateHidden +
                                    hidden_col] =
                                    grad_h_w2;
                            }
                            const uint32_t chunk =
                                hidden_col / 8;
                            const uint32_t in_chunk =
                                hidden_col & 7;
                            const uint32_t gate_col =
                                (kBF16Mode || kGateUpPrepared)
                                ? hidden_col
                                : chunk * 16 + in_chunk;
                            const uint32_t up_col =
                                (kBF16Mode || kGateUpPrepared)
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
                            const uint64_t activation_output_idx =
                                static_cast<uint64_t>(pool_row) *
                                    kIntermediateHidden +
                                hidden_col;
                            h_act_output[activation_output_idx] =
                                h_act_bf16;
                            // PRE_DOWN may phase-alias h_act and h_weighted.
                            // Preserve unweighted h until its route-gradient
                            // reduction, then overwrite it in a later phase.
                            if (h_act_output != h_weighted_output ||
                                (!kBF16Mode &&
                                 kRouteWeightMode !=
                                     RouteWeightMode::PostDown)) {
                                h_weighted_output[
                                    activation_output_idx] =
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
                            if (inplace_interleaved_gate_up_grad) {
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
                    if constexpr (kK3MxFp8ExactEpilogueRing) {
                        // Every canonical gate/up writer publishes its own
                        // generic stores before the complete epilogue role
                        // joins.  One release atomic then advertises this
                        // (physical block, N128) tile; 24 arrivals make the
                        // entire 6144-wide BF16 A source safe to reload.
                        // `bar.sync` is the CTA memory edge for every generic
                        // BF16 store above.  The elected release atomic below
                        // carries that joined write set to the ring producer;
                        // fencing all 768 writers would add one device fence
                        // per thread and destroy the intended overlap.  The
                        // acquiring producer performs the generic-to-async
                        // proxy fence before issuing its source TMA.
                        cutlass::arch::NamedBarrier::sync(
                            kNumW2DgradEpilogueThreads, 0);
                        if (epilogue_thread_idx == 0u) {
                            const uint32_t physical_pool_block =
                                pool_block_offset + m_block_idx;
                            // Chain all 24 independent tile publications; a
                            // release-only RMW would not acquire the preceding
                            // contributor's generic BF16 writes.
                            const uint32_t previous = ptx::atomic_add_acq_rel(
                                k3_mxfp8_exact_ring_source_ready() +
                                    physical_pool_block,
                                1u);
                            DG_DEVICE_ASSERT(
                                (previous & 0xffu) <
                                    kExactRingGUReadyTarget &&
                                (previous & 0xffff0000u) == 0u);
                        }
                    }
                    ptx::tcgen05_before_thread_sync();
                    tmem_empty_barriers[accum_stage]->arrive(0u);
                });

        }

            __syncthreads();
          }
        }

        if constexpr (kK3MxFp8ExactEpilogueRing) {
            // Both cohorts enter one readiness-driven grouped stream.  The
            // suffix reaches this point immediately; prefix clusters arrive as
            // they retire W2.  Cluster-local mailboxes make that late join
            // legal, while one global task cursor prevents duplicate dW13.
            // Retire the parent's drained mbarriers first, but preserve its
            // base-zero TMEM allocation for the later W13/dW2 continuation.
            __syncthreads();
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u && cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0u; i < kNumStages; ++i) {
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
                            primary_mma_barrier));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            residual_mma_barrier));
                }
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumEpilogueStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumDispatchBarriers; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dispatch_barriers[i]));
                }
            }
            comm::cluster_sync_with_relaxed_arrive();

            const uint32_t dw13_epoch = launch_epoch ^ 0x80000000u;
            k3_mxfp8_run_early_dw13_ab_ring<
                kHidden, kIntermediateHidden, kNumExperts, BLOCK_M,
                kNumSMs, kNumThreads,
                false, kK3MxFp8WgradBatchTasks>(
                    k3_mxfp8_dw13_hybrid_state(), dw13_epoch,
                    &tensor_map_mxfp8_wgrad_pack,
                    tensor_map_w13_wgrad_slot_d.exact_output_map,
                    smem_buffer,
                    expert_counts, backward_ranges,
                    num_acts_rows, acts_ptr,
                    w13_dequant_scratch,
                    clear_empty_wgrad_expert_outputs,
                    const_cast<cutlass::bfloat16_t*>(backward_grad_y),
                    backward_workspace.num_max_tokens_per_rank);
        }

        if constexpr (kMultiRangeBackward) {
            // Ranges occupy disjoint packed arenas, so no grid join is needed
            // between them.  Join once after every W2 dgrad before repurposing
            // the shared weight-state prefix or the immutable W2 cache as dW2.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
            full_grid_phase_barrier(kTraceSiteCount, 13);
            if constexpr (kK3MxFp8ExactEpilogueRing) {
                // Every grouped body has returned and the common ring drain is
                // complete.  Only now may fixed-top-k planes two through five
                // be cleared for W13's direct remote dX publication.
                prepare_direct_grad_x_planes();
            }
        }

#if DG_EXPERIMENTAL_K3_READY_WGRAD
        if constexpr (kMultiRangeBackward && !kK3MxFp8WgradOverlap) {
            static_assert(
                kReadyWgradSchedule && !kAccumulateWgrad,
                "multi-range owns overwrite/accumulate selection internally");

            // Materialize every disjoint BF16 operand before W13 starts and
            // publish one immutable scheduler arena per physical range. A
            // third arena copies the first reverse range's prefix and owns the
            // active-expert union. All three states and every operand store are
            // carried through the grid publication edge below before either
            // concatenated-K wgrad can consume them.
            for (active_range_iteration = 0u;
                 active_range_iteration < num_w2_ranges;
                 ++active_range_iteration) {
                active_range_index = backward_ranges.reverse_range_index(
                    active_range_iteration);
                const auto& range =
                    backward_ranges.ranges[active_range_index];
                active_pool_row_begin = range.pool_row_begin;
                active_pool_block_begin =
                    range.pool_row_begin / BLOCK_M;
                active_num_pool_rows = range.num_pool_rows;
                active_range_epoch = k3_multirange_epoch(
                    K3MultiRangeBackwardPhase::W2Wgrad,
                    backward_ranges.epoch_seed(active_range_index));
                active_accumulate_wgrad =
                    active_range_iteration != 0u;
                active_expert_counts =
                    expert_counts + backward_ranges.expert_counts_begin(
                        active_range_index, kNumExperts);
                prepare_ready_wgrad_state_and_operands(
                    /* prepare_operands = */ true,
                    /* prepare_state = */ true,
                    weight_tile_states +
                        active_range_index *
                            kReadyRangeStateStride);
            }
            prepare_ready_terminal_wgrad_state();
#if DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
            if constexpr (
                kK3MxFp8DW13Hybrid &&
                !kK3MxFp8ExactEpilogueRing) {
                // The ready ranges and pool-completion tail stay live until
                // dW2 returns.  Build the unrelated MXFP8 metadata in the
                // disjoint runtime-shifted arena before W13 begins so every
                // dgrad retirement has a valid composite destination.
                if constexpr (!kK3MxFp8ExactEpilogueRing) {
                    prepare_k3_mxfp8_dw2_overlap_state<
                        kNumExperts, BLOCK_M, kNumSMs, kNumThreads>(
                            expert_counts, backward_ranges,
                            k3_mxfp8_dw13_hybrid_state(),
                            K3MxFp8WgradGridBarrier<kNumSMs>{
                                phase_count, phase_sense});
                }
                // Open dW13 before dW2 clusters diverge. Composite
                // per-expert readiness—not this generation—guards every
                // aliased operand and output, so a locally retired dW2
                // cluster can transition directly into exact UMMA/TMA.
                if constexpr (!kK3MxFp8ExactEpilogueRing) {
                  if (blockIdx.x == 0u && threadIdx.x == 0u) {
                    const uint32_t dw13_epoch =
                        launch_epoch ^ 0x80000000u;
                    asm volatile(
                        "st.release.gpu.global.u32 [%0], %1;"
                        :: "l"(
                               k3_mxfp8_dw13_hybrid_state() +
                               K3MxFp8OverlapState::kDW13Epoch),
                           "r"(dw13_epoch)
                        : "memory");
                  }
                }
            }
#endif
            full_grid_phase_barrier(kTraceSiteCount, 14);
        }
#else
        static_assert(
            !kMultiRangeBackward || kK3MxFp8WgradOverlap,
            "multi-range requires ready BF16 or exact MXFP8 wgrad");
#endif

        if constexpr (kUseReducedW2ProducerSet) {
            if (warp_idx >= 4 && warp_idx < 8)
                cutlass::arch::warpgroup_reg_alloc<
                    kNumEpilogueRegisters>();
            // Restore the original seven-producer W13 register map before
            // any route/exact-X handoff or packed-weight conversion begins.
            __syncthreads();
        }
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
            constexpr uint32_t kW13EpilogueWarpStart =
                kK3MxFp8WgradOverlap
                    ? kK3MxFp8DW2W13QuantFirstWarp +
                          kK3MxFp8DW2W13QuantWarps
                    : kInlineW13WeightDequant
                        ? 8u
                        : kDgradEpilogueWarpStart;
            constexpr uint32_t kNumW13EpilogueThreads =
                kNumThreads - kW13EpilogueWarpStart * 32u -
                (kOverlapRouteGradWithW13
                     ? kNumDispatchThreads
                     : 0);
            DG_STATIC_ASSERT(
                kNumW13EpilogueThreads >= kNumEpilogueThreads,
                "W13 needs the four proven TMEM loader warps");
            DG_STATIC_ASSERT(
                !kK3MxFp8WgradOverlap ||
                    (kW13EpilogueWarpStart == 8u &&
                     kNumW13EpilogueThreads == 24u * 32u &&
                     kElasticDW2QuantScratchBegin >=
                         kReadyParentControlEnd &&
                     kElasticDW2QuantScratchEnd <=
                         kReadyBf16SmemDataBytes),
                "Elastic W13 must retain 24 epilogue warps and disjoint scratch");
            if constexpr (
                kK3MxFp8WgradOverlap &&
                !kK3MxFp8SuffixPanelStream) {
                detail::initialize_k3_mxfp8_dw2_w13_quant_scratch<
                    kHidden, kIntermediateHidden, BLOCK_M>(
                    smem_buffer + kElasticDW2QuantScratchBegin,
                    expert_counts,
                    &tensor_map_w2_wgrad_slot_d.exact_ranges,
                    num_acts_rows,
                    acts_ptr, weight_tile_states,
                    &tensor_map_mxfp8_wgrad_pack);
            }

            // W13 dgrad consumes grad_gate_up rows produced by every CTA in
            // the preceding L2-dgrad/SwiGLU phase. Cluster synchronization is
            // insufficient here: an early cluster can otherwise read rows
            // whose owning cluster has not stored them yet.
            full_grid_phase_barrier(12, 16);
            // Exact MXFP8 dW2/dW13 consumes the full BLOCK_M extent of both
            // three-term operand pairs.  The parent epilogues write only
            // logical rows, so clear every final partial expert block before
            // the suffix producers are allowed to quantize those arenas.
            if (late_exact_source_x || kK3MxFp8ThreeTermWgrad) {
                clear_w2_and_late_x_padding_rows();
                // The clear is distributed by final expert block across the
                // persistent grid.  Suffix CTAs may leave the W13 schedule
                // before the prefix and begin quantizing any expert, so a
                // CTA-local completion is not sufficient.  Publish every
                // writer's generic stores to the async TMA proxy and join the
                // full grid before producer/consumer roles can diverge.
                asm volatile(
                    "fence.proxy.async.global;" ::: "memory");
                __threadfence();
                full_grid_phase_barrier(kTraceSiteCount, 30);
            }

            if constexpr (kEarlyW2Wgrad) {
                if constexpr (!kBF16Mode)
                    trace_begin(13);
                // W2's dgrad and SiTU epilogue have retired globally, so its
                // two exact BF16 operands are final. Pad only their last
                // expert blocks, publish the grouped layout, and compute dW2
                // before either buffer is repurposed for W13 MXFP8 planes.
                if constexpr (kClearWgradPadding) {
                    if (!late_exact_source_x)
                        clear_w2_and_late_x_padding_rows();
                }
                initialize_wgrad_grouped_layout();
                asm volatile(
                    "fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount, 17);

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
                                primary_mma_barrier));
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

                if constexpr (kBackgroundW13WeightCache) {
                    if (threadIdx.x == 0) {
                        const uint32_t cursor_reset = 0;
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(w13_background_group_cursors +
                                   blockIdx.x),
                               "r"(cursor_reset)
                            : "memory");
                        auto* background_load_barrier =
                            reinterpret_cast<Barrier*>(
                                smem_buffer +
                                kBackgroundLoadBarrierOffset);
                        auto* background_participant_barrier =
                            reinterpret_cast<Barrier*>(
                                smem_buffer +
                                kBackgroundParticipantBarrierOffset);
                        background_load_barrier->init(1);
                        background_participant_barrier->init(
                            kBackgroundParticipantThreads);
                        cutlass::arch::fence_barrier_init();
                    }
                    // Publish both private mbarriers before the embedded BF16
                    // body dispatches only the background participant warps.
                    __syncthreads();
                }

                run_wgrad.template operator()<
                    false, 0, false, kEarlyDW2WgradStages>(
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    false,
                    no_input_tile_retired,
                    run_early_dw2_w13_background);

                // This join is deliberately separate from the callback's
                // participant mbarrier. Every callback waiter has returned
                // before it completes, so warp zero can now invalidate both
                // private objects without racing a merely-awakened waiter.
                if constexpr (kBackgroundW13WeightCache) {
                    __syncthreads();
                    if (warp_idx == 0 && cute::elect_one_sync()) {
                        auto* background_load_barrier =
                            reinterpret_cast<Barrier*>(
                                smem_buffer +
                                kBackgroundLoadBarrierOffset);
                        auto* background_participant_barrier =
                            reinterpret_cast<Barrier*>(
                                smem_buffer +
                                kBackgroundParticipantBarrierOffset);
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                background_load_barrier));
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                background_participant_barrier));
                    }
                    __syncthreads();
                }
                comm::cluster_sync_with_relaxed_arrive();

                if (warp_idx == 0 && cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < kNumDispatchBarriers; ++i)
                        dispatch_barriers[i]->init(1);
                    cutlass::arch::fence_barrier_init();
                }
                comm::cluster_sync_with_relaxed_arrive();
                if constexpr (!kBF16Mode)
                    trace_end(13);
            }

            if constexpr (kSplitResidualWeightCache) {
                // Trace site five is compile-time absent in pipelined K3.
                // Isolate the post-dW2 W13 producer phase from both dW2
                // (site 13) and W13 dgrad (site 21).  On K3, weight TMA plus
                // 16 converter warps run beside the complementary 15-warp
                // activation quantizer; one CTA/grid join publishes both.
                trace_begin(5);
                if constexpr (kCoSchedulePostDW2W13Builders) {
                    prepare_w13_residual_act_scales_once();

                    // early dW2 invalidated this parent barrier before the
                    // grouped BF16 body aliased shared memory.  Recreate it
                    // while every warp is still converged, then split roles.
                    if (warp_idx == 0 && cute::elect_one_sync()) {
                        weight_load_barrier->init(1);
                        cutlass::arch::fence_barrier_init();
                    }
                    __syncthreads();

                    const bool is_post_dw2_weight_builder_warp =
                        warp_idx == 0 ||
                        (warp_idx >= 4 && warp_idx < 20);
                    if (is_post_dw2_weight_builder_warp)
                        run_post_dw2_w13_residual_weight_cache();
                    else
                        build_w13_residual_acts_once();

                    // Both shared-memory partitions and all global stores are
                    // retired here.  End the mbarrier lifetime before W13
                    // reuses the parent barrier array, then publish once.
                    __syncthreads();
                    if (warp_idx == 0 && cute::elect_one_sync()) {
                        Barrier::invalidate(
                            reinterpret_cast<
                                Barrier::ValueType const*>(
                                weight_load_barrier));
                    }
                    __syncthreads();
                    full_grid_phase_barrier(kTraceSiteCount, 18);
                } else {
                    run_post_dw2_w13_residual_weight_cache();
                }
                trace_end(5);
            }

            if (inplace_interleaved_gate_up_grad &&
                !kBuildW13ResidualActsOnce) {
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
                    // The saved K3 gate/up layout is interleaved in exact
                    // eight-BF16 (16-byte) blocks. Snapshot one whole block
                    // per thread so the deinterleave preserves every BF16 bit
                    // while replacing eight scalar global/shared operations
                    // with one naturally aligned vector transaction.
                    constexpr uint32_t kGateUpVectorValues = 8;
                    constexpr uint32_t kGateUpVectorBlocks =
                        kGateUpColumns / kGateUpVectorValues;
                    static_assert(
                        kGateUpColumns % kGateUpVectorValues == 0,
                        "K3 gate/up width must be uint4 aligned");
                    auto* gate_up_row_vectors =
                        reinterpret_cast<uint4*>(
                            grad_gate_up_output + row_base);
                    auto* gate_up_scratch_vectors =
                        reinterpret_cast<uint4*>(
                            gate_up_row_scratch);
                    for (uint32_t block = threadIdx.x;
                         block < kGateUpVectorBlocks;
                         block += kNumThreads) {
                        gate_up_scratch_vectors[block] =
                            gate_up_row_vectors[block];
                    }
                    __syncthreads();
                    constexpr uint32_t kBranchVectorBlocks =
                        kIntermediateHidden / kGateUpVectorValues;
                    for (uint32_t destination_block = threadIdx.x;
                         destination_block < kGateUpVectorBlocks;
                         destination_block += kNumThreads) {
                        const bool is_up =
                            destination_block >= kBranchVectorBlocks;
                        const uint32_t branch_block = is_up
                            ? destination_block - kBranchVectorBlocks
                            : destination_block;
                        const uint32_t source_block =
                            branch_block * 2 +
                            static_cast<uint32_t>(is_up);
                        gate_up_row_vectors[destination_block] =
                            gate_up_scratch_vectors[source_block];
                    }
                    __syncthreads();
                }
                // W13 TMA readers may start on a different CTA, so publish the
                // completed in-place permutation grid-wide.
                full_grid_phase_barrier(kTraceSiteCount, 19);
            }

            if constexpr (!kCoSchedulePostDW2W13Builders)
                build_w13_residual_acts_once();

            // Gate/up has had its final SiTU-backward read. A multichunk
            // caller can now reuse that pool allocation as W13's canonical
            // BF16 dequant matrix before the W13 UMMA phase begins.
            if constexpr (
                phase_ordered_w13_dequant &&
                !kInlineW13WeightDequant) {
                dequant_noninline_weights_once(true, true, true);
                full_grid_phase_barrier(kTraceSiteCount, 20);
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
                full_grid_phase_barrier(13, 21);
            }

            if constexpr (
                kComputeRouteGrad && !kInputsPrepared &&
                !kOverlapRouteGradWithW13 &&
                !kExactBF16PipelinedGradYDispatch) {
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
                kNumRanks > 1 && !kBF16Mode &&
                (kExactSourceX || kPipelinedGradYDispatch)) {
              if (late_exact_source_x ||
                  kPipelinedGradYDispatch) {
                // Route groups publish row-level readiness to the exact-X TMA
                // warps above, so exact source dispatch is already complete
                // when this grid join retires. The source combine plane can
                // now be cleared for direct W13 grad-x publication.
                // The route and dispatch roles take different loop trip
                // counts. Join the whole CTA before thread zero enters the
                // persistent-grid barrier; its internal trailing CTA join is
                // not an entry rendezvous for these divergent roles.
#if DG_EXPERIMENTAL_K3_READY_WGRAD
                if constexpr (!kMultiRangeBackward)
                    prepare_ready_wgrad_state_and_operands();
#endif
                __syncthreads();
#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD
                if constexpr (kK3MxFp8WgradOverlap) {
                    // Build and release-publish the exact schedule while all
                    // CTAs remain converged.  The bounded suffix producer
                    // later stashes phase layouts immediately before use, so
                    // no descriptor address remains live through W13 dgrad.
                    prepare_k3_mxfp8_dw2_overlap_state<
                        kNumExperts, BLOCK_M, kNumSMs, kNumThreads>(
                            expert_counts, backward_ranges,
                            weight_tile_states,
                            K3MxFp8WgradGridBarrier<kNumSMs>{
                                phase_count, phase_sense});
                    // The converged state-preparation grid edge publishes the
                    // immutable generation before dW2 producers diverge.  The
                    // ready-first body may now wait on individual experts;
                    // operand readiness, not this epoch, guards their data.
                    if (blockIdx.x == 0u && threadIdx.x == 0u) {
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(weight_tile_states +
                                   K3MxFp8OverlapState::kEpoch),
                               "r"(launch_epoch)
                            : "memory");
                    }
                } else
#endif
                {
                    full_grid_phase_barrier(kTraceSiteCount, 22);
                }
                if constexpr (!kK3MxFp8ExactEpilogueRing)
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
                    full_grid_phase_barrier(14, 23);
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
                    // The ready-driven specialization may give a bounded suffix
                    // of complete clusters an immediate dW2 consumer role. A
                    // zero-sized suffix lets naturally retiring W13 producers
                    // join dW2 dynamically without permanently reserving SMs.
                    // The producer prefix keeps the original CTA-linear mapping
                    // and an even stride, so cluster peers still receive
                    // adjacent N tiles for every expert/M block.
                    if constexpr (
                        kReadyWgradSchedule && !kK3MxFp8WgradOverlap) {
                        if (blockIdx.x >= kReadyW13ProducerCTAs)
                            return;
                    }
                    static_assert(
                        !kK3MxFp8WgradOverlap ||
                            kK3MxFp8W13DgradCTAs == kNumSMs,
                        "Every exact-overlap CTA must enter the W13 schedule");
                    constexpr uint32_t kNumSchedulePasses =
                        kFirstMWaveBuildsResidualWeightCache
                            ? 2
                            : 1;
                    // Keep W13's packed producers in the same dependency-
                    // ordered prefix as W2. The even number of N tiles keeps
                    // both CTAs in every UMMA cluster on the same pass.
                    const uint32_t num_w13_ranges =
                        kMultiRangeBackward
                            ? backward_ranges.num_ranges
                            : 1u;
                    #pragma unroll 1
                    for (uint32_t range_iteration = 0u;
                         range_iteration < num_w13_ranges;
                         ++range_iteration) {
                        if constexpr (kMultiRangeBackward) {
                            active_range_iteration = range_iteration;
                            active_range_index =
                                backward_ranges.reverse_range_index(
                                    range_iteration);
                            const auto& range = backward_ranges.ranges[
                                active_range_index];
                            active_token_begin =
                                backward_ranges.token_begin(
                                    active_range_index);
                            active_pool_row_begin = range.pool_row_begin;
                            active_pool_block_begin =
                                range.pool_row_begin / BLOCK_M;
                            active_num_pool_rows = range.num_pool_rows;
                            active_num_acts_rows = range.num_acts_rows;
                            active_range_epoch = k3_multirange_epoch(
                                K3MultiRangeBackwardPhase::W13Dgrad,
                                backward_ranges.epoch_seed(
                                    active_range_index));
                            active_expert_counts =
                                expert_counts +
                                backward_ranges.expert_counts_begin(
                                    active_range_index, kNumExperts);
                        }
                        #pragma unroll 1
                        for (uint32_t schedule_pass = 0;
                             schedule_pass < kNumSchedulePasses;
                             ++schedule_pass) {
                            uint32_t next_assigned_block = blockIdx.x;
                            constexpr uint32_t kW13ScheduleCTAs =
                                kK3MxFp8WgradOverlap
                                ? kK3MxFp8W13DgradCTAs
                                : kReadyWgradSchedule
                                  ? kReadyW13ProducerCTAs
                                  : kNumSMs;
                            uint32_t global_block = 0;
                            uint32_t pool_block_offset =
                                active_pool_block_begin;
                            // Runtime-loop experts and ranges so the large
                            // W13 body remains one instruction image.
                            #pragma unroll 1
                            for (uint32_t expert_idx = 0;
                                 expert_idx < kNumExperts;
                                 ++expert_idx) {
                                const uint32_t num_tokens =
                                    static_cast<uint32_t>(__ldg(
                                        active_expert_counts +
                                        expert_idx));
                                const uint32_t num_m_blocks =
                                    math::ceil_div(num_tokens, BLOCK_M);
                                const uint32_t first_m_block =
                                    kFirstMWaveBuildsResidualWeightCache
                                        ? schedule_pass
                                        : 0;
                                const uint32_t scheduled_m_blocks =
                                    schedule_pass == 0
                                        ? cute::min(num_m_blocks, 1u)
                                        : num_m_blocks -
                                              cute::min(num_m_blocks, 1u);
                                const uint32_t expert_blocks =
                                    (kFirstMWaveBuildsResidualWeightCache
                                         ? scheduled_m_blocks
                                         : num_m_blocks) *
                                    kNumW13DgradBlockNs;
                                const uint32_t expert_end =
                                    global_block + expert_blocks;

                                while (next_assigned_block < global_block)
                                    next_assigned_block += kW13ScheduleCTAs;
                                while (next_assigned_block < expert_end) {
                                    const uint32_t local_block =
                                        next_assigned_block - global_block;
                                    const uint32_t m_block_idx =
                                        first_m_block +
                                        local_block /
                                            kNumW13DgradBlockNs;
                                    const uint32_t n_block_idx =
                                        local_block %
                                        kNumW13DgradBlockNs;
                                    const uint32_t valid_m = cute::min(
                                        num_tokens -
                                            m_block_idx * BLOCK_M,
                                        BLOCK_M);
                                    func(
                                        expert_idx, pool_block_offset,
                                        m_block_idx, n_block_idx, valid_m);
                                    next_assigned_block +=
                                        kW13ScheduleCTAs;
                                }
                                global_block = expert_end;
                                pool_block_offset += num_m_blocks;
                            }
                        }
                    }
                };

            trace_begin(15);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(15);
            if (warp_idx == 0 &&
                cute::elect_one_sync()) {
                if constexpr (
                    !kEarlyW2Wgrad &&
                    !kK3MxFp8ExactEpilogueRing) {
                    // W13 reuses the W2 transaction barriers. Invalidate the
                    // completed objects before resetting their phase; a
                    // direct mbarrier.init on a still-valid object is
                    // undefined and can deadlock after harmless code-layout
                    // changes perturb the producer timing. Early dW2 and the
                    // exact epilogue ring already retired these objects before
                    // aliasing their shared-memory storage.
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
                                primary_mma_barrier));
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                residual_mma_barrier));
                    }
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
                    primary_mma_barrier->init(1);
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
                                            kNumW13ResidualProducerWarps;
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
                                            kFirstMWaveBuildsResidualWeightCache &&
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
                                                    asm volatile("trap;");
                                                }
                                            }
                                        }
                                        cutlass::arch::NamedBarrier::sync(
                                            kNumW13ResidualProducerThreads,
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
                                             kNumW13ResidualProducerThreads) {
                                        smem_sfa[stage_idx][sf_idx] =
                                            0x7f7f7f7fu;
                                        smem_dgrad_sfa_residual[
                                            stage_idx][sf_idx] =
                                            0x7f7f7f7fu;
                                    }
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumW13ResidualProducerThreads,
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
                                        kNumW13ResidualProducerThreads,
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
                                        k3_mxfp4_to_mxfp8_transposed_tile_one_pass<
                                            kNumW13ResidualProducerWarps>(
                                            smem_dgrad_weight_source,
                                            smem_dgrad_weight_scale_source,
                                            weight_bytes,
                                            smem_sfb[stage_idx],
                                            n_block_idx,
                                            producer_warp_idx * 32 +
                                                lane_idx);
                                    } else {
                                    for (uint32_t local_n =
                                             producer_warp_idx;
                                         local_n < LOAD_BLOCK_N;
                                         local_n +=
                                             kNumW13ResidualProducerWarps) {
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
                                        }
                                    }
                                    }
                                    cutlass::arch::
                                        fence_view_async_shared();
                                    cutlass::arch::NamedBarrier::sync(
                                        kNumW13ResidualProducerThreads,
                                        kResidualWeightProducerBarrier);
                                      if (producer_warp_idx == 0 &&
                                          cute::elect_one_sync()) {
                                          if constexpr (
                                              kOnDemandResidualWeightCache) {
                                              // Publish the cache only through
                                              // TMA, then release its epoch
                                              // after completion. This keeps
                                              // one global write per value and
                                              // scale without weakening the
                                              // consumer's acquire edge.
                                              cute::tma_store_fence();
                                              cute::SM90_TMA_STORE_2D::copy(
                                                  &tensor_map_w13_dgrad_weights,
                                                  smem_dgrad_b[stage_idx],
                                                  global_k_block_idx *
                                                      DGRAD_BLOCK_K,
                                                  expert_idx * kHidden +
                                                      n_block_idx * BLOCK_N);
                                              cute::SM90_TMA_STORE_2D::copy(
                                                  &tensor_map_w13_dgrad_weights_sf,
                                                  smem_sfb[stage_idx],
                                                  n_block_idx * BLOCK_N,
                                                  expert_idx *
                                                          ((2 *
                                                            kIntermediateHidden) /
                                                           (kGranK * 4)) +
                                                      global_k_block_idx);
                                              cute::tma_store_arrive();
                                          }
                                          full_barriers[stage_idx]
                                              ->arrive(0u);
                                          if constexpr (
                                              kOnDemandResidualWeightCache) {
                                              ptx::tma_store_wait<0>();
                                              asm volatile(
                                                  "fence.proxy.async.global;"
                                                  ::: "memory");
                                              asm volatile(
                                                  "st.release.gpu.global.u32 [%0], %1;"
                                                  :: "l"(
                                                         weight_tile_states +
                                                         kNumW2WeightTileStates +
                                                         weight_tile_idx),
                                                     "r"(w13_launch_epoch)
                                                  : "memory");
                                          }
                                      }
                                      if constexpr (
                                          kOnDemandResidualWeightCache) {
                                          cutlass::arch::NamedBarrier::sync(
                                              kNumW13ResidualProducerThreads,
                                              kResidualWeightCacheStoreDoneBarrier);
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
                (kInlineW13WeightDequant &&
                 warp_idx >= 3 && warp_idx < 8)) {
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
                                    kConcurrentResidualWeightCache) {
                                    // The concurrent builder publishes W13
                                    // packed-weight tiles independently of
                                    // this consumer warp.  Acquire the exact
                                    // tile epoch before issuing either TMA
                                    // load; the warp join keeps all lanes on
                                    // the same pipeline phase.
                                    if (lane_idx == 0) {
                                        while (ptx::ld_acq(
                                                   weight_tile_states +
                                                   kNumW2WeightTileStates +
                                                   weight_tile_idx) !=
                                               w13_launch_epoch) {
                                        }
                                    }
                                    __syncwarp();
                                } else if constexpr (
                                    !kBF16Mode &&
                                    !kInlineW13WeightDequant &&
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
                                if constexpr (!kInlineW13WeightDequant) {
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
                    uint32_t primary_mma_phase = 0;
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
                                                primary_mma_barrier),
                                            kCTAMask);
                                    primary_mma_barrier->wait(
                                        primary_mma_phase);
                                    primary_mma_phase ^= 1;
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
                        [&](const uint32_t& expert_idx,
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
#if DG_EXPERIMENTAL_K3_READY_WGRAD || \
    DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD || \
    DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
                            if constexpr (
                                !kK3MxFp8ExactEpilogueRing &&
                                ((kReadyWgradSchedule &&
                                  (!kMultiRangeBackward ||
                                   kK3TwoSegmentBF16ProgressiveWgrad)) ||
                                 kK3MxFp8WgradOverlap ||
                                 kK3MxFp8DW13Hybrid)) {
                                // The final full-barrier wait above retires both
                                // CTAs' last W13-weight TMA read for this output
                                // tile.  Release one cluster-task credit; dW13
                                // may overwrite this expert's aliased BF16
                                // weight slice only after acquiring every credit.
                                if (cute::elect_one_sync()) {
                                    if constexpr (kK3MxFp8WgradOverlap) {
                                        constexpr uint32_t
                                            kRetirementsPerPoolBlock =
                                                kNumW13DgradBlockNs / 2u;
                                        static_assert(
                                            kRetirementsPerPoolBlock == 14u,
                                            "K3 W13 retirement target changed");
                                        const uint32_t pool_blocks =
                                            weight_tile_states[
                                                K3MxFp8OverlapState::
                                                    kPoolBlockPrefix +
                                                expert_idx + 1u] -
                                            weight_tile_states[
                                                K3MxFp8OverlapState::
                                                    kPoolBlockPrefix +
                                                expert_idx];
                                        const uint32_t target =
                                            pool_blocks *
                                                kRetirementsPerPoolBlock;
                                        const uint32_t previous =
                                            ptx::atomic_add_acq_rel(
                                                weight_tile_states +
                                                    K3MxFp8OverlapState::
                                                        kDW13CompositeReady +
                                                    expert_idx,
                                                1u);
                                        DG_DEVICE_ASSERT(previous < target);
                                    } else {
                                        if constexpr (
                                            kK3MxFp8DW13Hybrid &&
                                            !kK3MxFp8ExactEpilogueRing) {
                                            auto* const hybrid_state =
                                                k3_mxfp8_dw13_hybrid_state();
                                            const uint32_t hybrid_pool_blocks =
                                                hybrid_state[
                                                    K3MxFp8OverlapState::
                                                        kPoolBlockPrefix +
                                                    expert_idx + 1u] -
                                                hybrid_state[
                                                    K3MxFp8OverlapState::
                                                        kPoolBlockPrefix +
                                                    expert_idx];
                                            const uint32_t hybrid_target =
                                                hybrid_pool_blocks * 14u;
                                            const uint32_t hybrid_previous =
                                                ptx::atomic_add_acq_rel(
                                                    hybrid_state +
                                                        K3MxFp8OverlapState::
                                                            kDW13CompositeReady +
                                                        expert_idx,
                                                    1u);
                                            DG_DEVICE_ASSERT(
                                                hybrid_previous <
                                                hybrid_target);
                                        } else if constexpr (
                                            kK3TwoSegmentBF16ProgressiveWgrad) {
                                            // Both range states and the union
                                            // prefix were published before W13
                                            // started. Accumulate every final
                                            // packed-weight read into the union
                                            // arena; the ready-first dW13
                                            // provider converts this counter to
                                            // an expert-local task cursor only
                                            // after the exact two-segment target.
                                            DG_DEVICE_ASSERT(
                                                backward_ranges.num_ranges ==
                                                2u);
                                            auto* const union_state =
                                                weight_tile_states +
                                                kReadyTerminalUnionStateWord;
                                            const uint32_t first_pool_blocks =
                                                union_state[
                                                    kReadyPoolPrefixWord +
                                                    expert_idx + 1u] -
                                                union_state[
                                                    kReadyPoolPrefixWord +
                                                    expert_idx];
                                            const uint32_t second_range_index =
                                                backward_ranges.
                                                    reverse_range_index(1u);
                                            const auto* const second_state =
                                                weight_tile_states +
                                                second_range_index *
                                                    kReadyRangeStateStride;
                                            const uint32_t second_pool_blocks =
                                                second_state[
                                                    kReadyPoolPrefixWord +
                                                    expert_idx + 1u] -
                                                second_state[
                                                    kReadyPoolPrefixWord +
                                                    expert_idx];
                                            const uint32_t expected =
                                                (first_pool_blocks +
                                                 second_pool_blocks) *
                                                (kNumW13DgradBlockNs / 2u);
                                            // The full transaction-barrier
                                            // wait above retires both CTAs'
                                            // final W13-weight TMA reads. The
                                            // release counter carries that
                                            // completion to the scheduler;
                                            // the acquiring consumer performs
                                            // the generic-to-async proxy fence
                                            // before publishing an aliased D
                                            // TMA-store batch.
                                            const uint32_t previous =
                                                ptx::atomic_add_rel(
                                                    union_state +
                                                        kReadyW13RetiredWord +
                                                        expert_idx,
                                                    1u);
                                            DG_DEVICE_ASSERT(
                                                expected != 0u &&
                                                previous < expected);
                                        } else {
                                            const uint32_t num_tokens =
                                                static_cast<uint32_t>(__ldg(
                                                    expert_counts + expert_idx));
                                            const uint32_t pool_blocks =
                                                math::ceil_div(
                                                    num_tokens, BLOCK_M);
                                            const uint32_t expected =
                                                pool_blocks *
                                                (kNumW13DgradBlockNs / 2u);
                                            const uint32_t previous =
                                                ptx::atomic_add_rel(
                                                    weight_tile_states +
                                                        kReadyW13RetiredWord +
                                                        expert_idx,
                                                    1u);
                                            DG_DEVICE_ASSERT(
                                                previous < expected);
                                        }
                                    }
                                }
                                __syncwarp();
                            }
#endif
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
                    uint32_t follower_primary_mma_phase = 0;
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
                                    primary_mma_barrier->wait(
                                        follower_primary_mma_phase);
                                    follower_primary_mma_phase ^= 1;
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
                kOverlapExactSourceXPlaneOneWithW13 &&
                warp_idx == 3u) {
                run_w13_overlapped_source_x_and_reclaim_plane_one();
            } else if (
                kOverlapRouteGradWithW13 &&
                warp_idx >= kW13RouteWarpStart &&
                warp_idx <
                    kW13RouteWarpStart +
                        kNumDispatchWarps) {
                // One warp owns one route row, preserving the exact four-lane
                // accumulation and shuffle tree used by the serialized K3
                // reducer.  Only route-to-warp assignment changes; no row's
                // FP32 reduction order changes.
                const uint32_t route_warp_idx =
                    warp_idx - kW13RouteWarpStart;
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
            } else if (
                kK3MxFp8WgradOverlap &&
                !kK3MxFp8SuffixPanelStream &&
                warp_idx >= kK3MxFp8DW2W13QuantFirstWarp &&
                warp_idx < kK3MxFp8DW2W13QuantFirstWarp +
                               kK3MxFp8DW2W13QuantWarps) {
                // One warpgroup per CTA builds exact group-32 primary/residual
                // dW2 operands while this CTA's W13 loader/MMA/24-warp
                // epilogue continue independently. All 148 engines share the
                // exact-only producer cursor and publish individual feature
                // masks, so a cluster can transition directly into ready-first
                // dW work after its own W13 schedule drains.
                detail::k3_mxfp8_stream_dw2_operands_during_w13<
                    kHidden, kIntermediateHidden, kNumExperts, BLOCK_M,
                    kNumSMs, kNumThreads,
                    true, false, true, true,
                    kNumSMs +
                        kK3MxFp8PersistentDW2ProducerCTAs *
                            kK3MxFp8DW2PersistentEnginesPerCTA>(
                        smem_buffer + kElasticDW2QuantScratchBegin,
                        warp_idx, lane_idx);
            } else if (warp_idx >= kW13EpilogueWarpStart) {
                const uint32_t epilogue_warp_idx =
                    warp_idx - kW13EpilogueWarpStart -
                    (kOverlapRouteGradWithW13 &&
                             warp_idx >=
                                 kW13RouteWarpStart +
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
                                              active_token_begin +
                                              metadata.token_idx) *
                                                 kHidden +
                                             out_col);
                                        auto* const scatter_dst =
                                            kOverlapExactSourceXPlaneOneWithW13 &&
                                                    metadata.topk_idx == 1u
                                                ? grad_y_unweighted_output +
                                                      static_cast<uint64_t>(
                                                          pool_row) *
                                                          kHidden +
                                                      out_col
                                                : backward_sym_buffer.map(
                                                      dst,
                                                      metadata.rank_idx);
                                        *reinterpret_cast<uint4*>(
                                            scatter_dst) = packed;
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
                                              active_token_begin +
                                              metadata.token_idx) *
                                                 kHidden +
                                             out_col);
                                        auto* const scatter_dst =
                                            kOverlapExactSourceXPlaneOneWithW13 &&
                                                    metadata.topk_idx == 1u
                                                ? grad_y_unweighted_output +
                                                      static_cast<uint64_t>(
                                                          pool_row) *
                                                          kHidden +
                                                      out_col
                                                : backward_sym_buffer.map(
                                                      dst,
                                                      metadata.rank_idx);
                                        *reinterpret_cast<uint32_t*>(
                                            scatter_dst) = packed;
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
                                              active_token_begin +
                                              metadata.token_idx) *
                                                 kHidden +
                                             out_col);
                                        auto* const scatter_dst =
                                            kOverlapExactSourceXPlaneOneWithW13 &&
                                                    metadata.topk_idx == 1u
                                                ? grad_y_unweighted_output +
                                                      static_cast<uint64_t>(
                                                          pool_row) *
                                                          kHidden +
                                                      out_col
                                                : backward_sym_buffer.map(
                                                      dst,
                                                      metadata.rank_idx);
                                        *scatter_dst = value;
                                    }
                                }
                            }
                        }
                        if constexpr (kStreamingDirectGradXCombine) {
                            // Every N tile scatters one disjoint 128-column
                            // slice. A CTA barrier gathers all scatter threads;
                            // the acq_rel RMW then chains the 28 N-tile CTAs for
                            // this physical pool block. Only the terminal CTA
                            // publishes source-token readiness, after every
                            // route row is complete in symmetric memory.
                            cutlass::arch::NamedBarrier::sync(
                                kNumW13EpilogueThreads, 0);
                            auto* completion_flag =
                                reinterpret_cast<volatile uint32_t*>(
                                    smem_cd[0]);
                            if (epilogue_thread_idx == 0u) {
                                __threadfence_system();
                                const uint32_t pool_block =
                                    pool_block_offset + m_block_idx;
                                const uint32_t previous =
                                    ptx::atomic_add_acq_rel(
                                        direct_grad_x_pool_completions +
                                            pool_block,
                                        1u);
                                DG_DEVICE_ASSERT(
                                    previous < kNumW13DgradBlockNs);
                                *completion_flag =
                                    previous + 1u ==
                                        kNumW13DgradBlockNs
                                    ? 1u
                                    : 0u;
                            }
                            cutlass::arch::NamedBarrier::sync(
                                kNumW13EpilogueThreads, 0);
                            // The first warp fans out the system-scope
                            // publications. Its explicit warp rendezvous
                            // carries lane zero's terminal acquire to every
                            // lane that issues a remote release.
                            if (epilogue_thread_idx < 32u) {
                                __syncwarp();
                            }
                            if (epilogue_thread_idx < 32u &&
                                *completion_flag != 0u) {
                                for (uint32_t local_m =
                                         epilogue_thread_idx;
                                     local_m < valid_m;
                                     local_m += 32u) {
                                    const uint32_t pool_row =
                                        (pool_block_offset + m_block_idx) *
                                            BLOCK_M +
                                        local_m;
                                    const auto metadata =
                                        token_src_metadata[pool_row];
                                    const uint32_t source_token =
                                        active_token_begin +
                                        metadata.token_idx;
                                    DG_DEVICE_ASSERT(
                                        source_token <
                                        backward_workspace
                                            .num_max_tokens_per_rank);
                                    auto* local_ready =
                                        direct_grad_x_ready_counts +
                                        K3DirectGradXReadyContract::
                                            counter_index(source_token);
                                    ptx::red_add_rel_sys(
                                        backward_sym_buffer.map(
                                            local_ready,
                                            metadata.rank_idx),
                                        1);
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

            if constexpr (kOverlapExactSourceXPlaneOneWithW13) {
                // Join only this CTA's split roles. Warp three has already
                // passed both source-warp grid/rank barriers, so plane one is
                // reclaimed globally; this CTA's W13 epilogue has produced
                // every staged value for the N tiles it owns. Replaying the
                // deterministic W13 schedule lets each CTA publish exactly
                // its own tiles without a full-grid join, preserving late-
                // W13/terminal-dW2 overlap.
                __threadfence();
                __syncthreads();
                for_each_w13_dgrad_block(
                    [&](const uint32_t&,
                        const uint32_t& pool_block_offset,
                        const uint32_t& m_block_idx,
                        const uint32_t& n_block_idx,
                        const uint32_t& valid_m) {
                        k3_scatter_deferred_plane_one_grad_x_tile<
                            kHidden, BLOCK_M, BLOCK_N,
                            kNumRanks, kNumThreads>(
                                token_src_metadata,
                                grad_y_unweighted_output, backward_x,
                                backward_sym_buffer,
                                pool_block_offset, m_block_idx,
                                n_block_idx, valid_m);
                    });
                __syncthreads();
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

        // W13's control owner can finish before the borrowed dW2 warpgroup.
        // Request a panel-boundary stop immediately, allowing the cluster to
        // retire without abandoning a partial value/scale publication. The
        // bounded persistent producer pairs resume the same A cursor later.
        if constexpr (
            kK3MxFp8WgradOverlap &&
            !kK3MxFp8SuffixPanelStream) {
            if (blockIdx.x >=
                    kNumSMs - kK3MxFp8PersistentDW2ProducerCTAs &&
                threadIdx.x == 0u) {
                detail::request_k3_mxfp8_dw2_w13_quant_stop(
                    weight_tile_states +
                        K3MxFp8OverlapState::
                            kElasticDW2HandoffRequest);
            }
        }

        trace_end(21);

#if DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL
        if constexpr (kK3BranchMajorBF16EarlyDW2Overlap) {
            // dW2 consumes only operands published before W13. Its dynamic
            // body also owns the rank publication callback, so useful UMMA/TMA
            // work proceeds while late clusters finish W13 and join the queue.
            run_branch_major_early_dw2.template operator()<true>();
        }
#endif

#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD
        if constexpr (kK3MxFp8WgradOverlap) {
            using Overlap = K3MxFp8OverlapState;
            using Prefix = typename Overlap::Prefix;

            // Each complete cluster leaves the parent pipeline only after its
            // final local W13 task.  Shared barriers and TMEM are cluster-local,
            // so suffix clusters can transition immediately while the prefix
            // continues W13.  Retain the already-empty base-zero 512-column
            // TMEM allocation and replace only its barrier generation.
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u && cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0u; i < kNumStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumEpilogueStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumDispatchBarriers; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dispatch_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u; i < 2u; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dequant_barriers + i));
                }
                if constexpr (!kK3MxFp8SuffixPanelStream) {
                    auto* const elastic_barriers =
                        reinterpret_cast<Barrier*>(
                            smem_buffer + kElasticDW2QuantScratchBegin +
                            kK3MxFp8DW2W13QuantBarrierOffset);
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            elastic_barriers + 1));
                }
            }
            comm::cluster_sync_with_relaxed_arrive();

            static_assert(
                !kUseReducedW2ProducerSet,
                "Exact overlap register entry state changed");
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();

            static_assert(
                kK3MxFp8PersistentDW2ProducerCTAs > 0u &&
                    kK3MxFp8PersistentDW2ProducerCTAs % 2u == 0u &&
                    kK3MxFp8PersistentDW2ConsumerCTAs % 2u == 0u &&
                    kK3MxFp8W13DgradCTAs == kNumSMs,
                "Persistent suffix roles must split complete CTA pairs only after every CTA runs W13");

            // One phase-tagged provider owns the exact dW2/dW13 queue and its
            // four-word mailbox.  There is no all-dW2 body return or barrier
            // reset before the first ready dW13 quantum.
            trace_begin(20);
            // A bounded tail of complete clusters temporarily lends all eight
            // warpgroups to dW2 operand production.  It resumes the A cursor
            // left by panel-boundary W13 producers, waits only on the combined
            // producer-engine alias edge, builds B, and then late-joins this
            // same unified body exactly once.  Earlier cluster pairs can enter
            // ready-first UMMA immediately. The body's 148-CTA NVLink callback
            // remains live: every producer CTA unconditionally reaches it.
            if constexpr (!kK3MxFp8SuffixPanelStream) {
                if (blockIdx.x >=
                    kNumSMs - kK3MxFp8PersistentDW2ProducerCTAs) {
                    detail::k3_mxfp8_finish_dw2_with_persistent_producer_cta<
                        kHidden, kIntermediateHidden, kNumExperts, BLOCK_M,
                        kNumSMs, kNumThreads,
                        kK3MxFp8PersistentDW2ProducerCTAs>(
                            expert_counts,
                            &tensor_map_w2_wgrad_slot_d.exact_ranges,
                            num_acts_rows, acts_ptr,
                            &tensor_map_mxfp8_wgrad_pack,
                            weight_tile_states, smem_buffer,
                            warp_idx, lane_idx);
                    // Both peer CTAs must finish their eight engines and
                    // private barrier teardown before either initializes the
                    // cluster-wide grouped-body barriers.
                    comm::cluster_sync_with_relaxed_arrive();
                }
            }
            using ExactTensorMapPack =
                K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>;
            using ExactAuxSlot = K3MxFp8WgradAuxSlot<
                cute::TmaDescriptor, cutlass::bfloat16_t,
                cutlass::float_e4m3_t>;
            using ExactSuffixHandoff = K3MxFp8WgradSuffixHandoff<
                ExactTensorMapPack, layout::SymBuffer<kNumRanks>,
                layout::Workspace, ExactAuxSlot>;
            constexpr uint32_t kExactSuffixHandoffOffset =
                math::constexpr_align(
                    kK3MxFp8DW13QuantBodySmemBytes, 8u);
            static_assert(kExactSuffixHandoffOffset == 153736u);
            static_assert(
                kExactSuffixHandoffOffset +
                        sizeof(ExactSuffixHandoff) <=
                    kK3MxFp8DW13QuantScratchBegin);
            auto* const exact_suffix_handoff =
                reinterpret_cast<ExactSuffixHandoff*>(
                    smem_buffer + kExactSuffixHandoffOffset);
            if (threadIdx.x == 0u) {
                exact_suffix_handoff->tensor_map_pack =
                    &tensor_map_mxfp8_wgrad_pack;
                exact_suffix_handoff->backward_sym_buffer =
                    &backward_sym_buffer;
                exact_suffix_handoff->backward_workspace =
                    &backward_workspace;
                exact_suffix_handoff->dw2_output_slot =
                    &tensor_map_w2_wgrad_slot_a;
                exact_suffix_handoff->dw13_output_slot =
                    &tensor_map_w2_wgrad_slot_b;
                exact_suffix_handoff->ranges_slot =
                    &tensor_map_w2_wgrad_slot_d;
                exact_suffix_handoff->args_slot =
                    &tensor_map_w13_wgrad_slot_a;
                exact_suffix_handoff->expected_launch_epoch =
                    launch_epoch;
                exact_suffix_handoff->reserved = 0u;
            }
            __syncthreads();
            k3_mxfp8_run_dynamic_unified_wgrad_overlap<
                kHidden, kIntermediateHidden,
                kNumExperts, BLOCK_M, kNumSMs, kNumRanks, kNumThreads,
                kAccumulateWgrad, kK3MxFp8WgradBatchTasks,
                kClearWgradPadding && !kInlineWeightDequant &&
                    !kPhaseOrderedWeightDequant,
                kMultiRangeBackward>();
            trace_end(20);
            trace_end(0);
            return;
        }
#endif

#if DG_EXPERIMENTAL_K3_READY_WGRAD
        if constexpr (
                kMultiRangeBackward &&
                !kK3BranchMajorBF16DynamicTail) {
            static_assert(
                kReadyWgradSchedule && kExactSourceX &&
                    kDirectRemoteGradX && !kAccumulateWgrad,
                "multi-range dW2 overlap requires exact ready-wgrad training");

            // Initial-consumer clusters have no W13 tiles and arrive here
            // while the producer prefix is still executing both W13 ranges.
            // Retire the parent resource lifetime cluster-locally, then keep
            // one BF16 UMMA/TMA lifetime across both dW2 ranges and terminal
            // dW13.  Late W13 producers perform the same transition before
            // joining the shared dynamic queue.
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u && cute::elect_one_sync()) {
                #pragma unroll
                for (uint32_t i = 0u; i < kNumStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            empty_barriers[i]));
                }
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumEpilogueStages; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_full_barriers[i]));
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            tmem_empty_barriers[i]));
                }
                if constexpr (!kK3MxFp8ExactEpilogueRing) {
                    // The exact ring retired dispatch immediately after W2 so
                    // its early grouped body could own the terminal lifetime.
                    // mbarrier.inval on that already-invalid object is UB.
                    #pragma unroll
                    for (uint32_t i = 0u;
                         i < kNumDispatchBarriers; ++i) {
                        Barrier::invalidate(
                            reinterpret_cast<Barrier::ValueType const*>(
                                dispatch_barriers[i]));
                    }
                }
                // Exact-ring W13 dequant is inline, so only the independently
                // initialized W2 materializer barrier is live. Other ready
                // schedules initialize both dedicated materializer barriers.
                constexpr uint32_t kNumLiveDequantBarriers =
                    kInlineW13WeightDequant ? 1u : 2u;
                #pragma unroll
                for (uint32_t i = 0u;
                     i < kNumLiveDequantBarriers; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dequant_barriers + i));
                }
            }
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0u)
                Allocator().free(0, kNumTmemCols);
            __syncthreads();

            static_assert(
                !kUseReducedW2ProducerSet,
                "multi-range ready-wgrad register entry state changed");
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();

            using TwoSegmentReadyDW2Provider =
                sched::ExternalKGroupedTwoSegmentDynamicRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    kHidden, kIntermediateHidden,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord,
                    kK3MxFp8DW13Hybrid>;
            using TwoSegmentUnifiedBF16WgradProvider = sched::
                ExternalKGroupedK3TwoSegmentBF16WgradDynamicRangeProvider<
                    kWgradBlockM, kWgradBlockN, kNumSMs,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord>;
            using InitializeRetainResources =
                Sm100Bf16GemmBatchResourceHooks<true, false>;
            using RetainExistingResources =
                Sm100Bf16GemmCallerManagedBatchResourceHooks;
            using ReleaseExistingResources =
                Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<>;
            using InitializeReleaseResources =
                Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<true>;
            static_assert(
                TwoSegmentReadyDW2Provider::kCompleteAcquireMask ==
                    kReadyCompleteRoleMask,
                "multi-range dW2 scheduler role mask drifted");
            static_assert(
                TwoSegmentUnifiedBF16WgradProvider::
                        kCompleteAcquireMask ==
                    kReadyCompleteRoleMask &&
                TwoSegmentUnifiedBF16WgradProvider::
                        kTaskBF16PhaseTagged &&
                TwoSegmentUnifiedBF16WgradProvider::
                        kTaskHasTwoSegmentK &&
                TwoSegmentUnifiedBF16WgradProvider::
                        kTaskKAlignment == 64u,
                "unified K3 BF16 wgrad must retain logical-64 "
                "two-segment phase scheduling");
            static_assert(
                InitializeRetainResources::kInitializeBatchResources &&
                    !InitializeRetainResources::kReleaseBatchResources,
                "multi-range dW2 resources must remain live for dW13");
            static_assert(
                InitializeReleaseResources::kInitializeBatchResources &&
                    InitializeReleaseResources::kReleaseBatchResources &&
                    InitializeReleaseResources::kSynchronizeAfterRelease,
                "hybrid dW2 must invalidate and release before MXFP8 reinit");
            static_assert(
                !ReleaseExistingResources::kInitializeBatchResources &&
                    ReleaseExistingResources::kReleaseBatchResources &&
                    ReleaseExistingResources::kSynchronizeAfterRelease,
                "three-range hybrid dW2 must release its retained lifetime");

            const uint32_t ready_cluster_idx = blockIdx.x / 2u;
            trace_begin(17);
            auto* const hybrid_state =
                k3_mxfp8_dw13_hybrid_state();
            const auto* const hybrid_backward_ranges = &backward_ranges;
            const auto* const hybrid_tensor_map_pack =
                &tensor_map_mxfp8_wgrad_pack;
            const bool hybrid_scale_arena_aliases_dw2_source =
                reinterpret_cast<const void*>(acts_ptr) ==
                reinterpret_cast<const void*>(down_unweighted_output);
            const auto hybrid_input_tile_retired = [
                    hybrid_state, expert_counts,
                    hybrid_backward_ranges] (
                    const uint32_t expert,
                    const uint32_t m_block,
                    const uint32_t n_block) {
                if constexpr (
                    kK3MxFp8DW13Hybrid &&
                    !kK3MxFp8ExactEpilogueRing) {
                    // sm100_bf16_gemm_body invokes this callback on the full
                    // leader MMA warp after both CTAs' final A/B TMA load for
                    // one output tile has retired.  Elect exactly one lane.
                    if (!cute::elect_one_sync())
                        return;
                    const uint32_t previous = ptx::atomic_add_acq_rel(
                        hybrid_state +
                            K3MxFp8OverlapState::kDW2InputRetired + expert,
                        1u);
                    uint32_t active_ranges = 0u;
                    #pragma unroll
                    for (uint32_t range_idx = 0u;
                         range_idx < kK3MaxBackwardRanges; ++range_idx) {
                        if (range_idx >= hybrid_backward_ranges->num_ranges)
                            break;
                        active_ranges += __ldg(
                            expert_counts +
                            hybrid_backward_ranges->expert_counts_begin(
                                range_idx, kNumExperts) + expert) != 0;
                    }
                    // The two-segment provider retires one output task after
                    // both K slices. The three-range path instead invokes one
                    // provider per active range and therefore contributes one
                    // complete output grid for each such range.
                    const uint32_t expected_retirements =
                        K3MxFp8OverlapState::kDW2ClusterTasksPerExpert *
                        (hybrid_backward_ranges->num_ranges == 3u
                             ? active_ranges : 1u);
                    DG_DEVICE_ASSERT(
                        expected_retirements != 0u &&
                        previous < expected_retirements);
                    auto* const pair_counters =
                        hybrid_state +
                        K3MxFp8OverlapState::kDW2InputPairRetired +
                        expert * K3MxFp8OverlapState::
                            kDW2InputPairCountersPerExpert;
                    auto* const feature_masks =
                        hybrid_state +
                        K3MxFp8OverlapState::kDW2FeatureReadyMasks +
                        expert * K3MxFp8OverlapState::
                            kDW2FeatureReadyWordsPerExpert;

                    if (hybrid_backward_ranges->num_ranges == 3u) {
                        // Three ranges execute three independent BF16 GEMM
                        // passes. Keep their conservative whole-expert edge
                        // until the quantizer carries a range-scaled target.
                        if (previous + 1u != expected_retirements)
                            return;
                        #pragma unroll
                        for (uint32_t pair = 0u;
                             pair < kHidden / 256u; ++pair) {
                            const uint32_t target =
                                kIntermediateHidden / 256u;
                            asm volatile(
                                "st.release.gpu.global.u32 [%0], %1;"
                                :: "l"(pair_counters + pair), "r"(target)
                                : "memory");
                        }
                        #pragma unroll
                        for (uint32_t pair = 0u;
                             pair < kIntermediateHidden / 256u; ++pair) {
                            const uint32_t target = kHidden / 256u;
                            asm volatile(
                                "st.release.gpu.global.u32 [%0], %1;"
                                :: "l"(
                                       pair_counters + kHidden / 256u + pair),
                                   "r"(target)
                                : "memory");
                        }
                        constexpr uint32_t kFeatureWord0 = 0xffffffffu;
                        constexpr uint32_t kFeatureWord1 =
                            (1u <<
                                 (K3MxFp8OverlapState::
                                      kDW2FeaturePanelsPerExpert -
                                  32u)) -
                            1u;
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(feature_masks), "r"(kFeatureWord0)
                            : "memory");
                        asm volatile(
                            "st.release.gpu.global.u32 [%0], %1;"
                            :: "l"(feature_masks + 1u), "r"(kFeatureWord1)
                            : "memory");
                        return;
                    }

                    // BF16 dW2 clusters span two adjacent M blocks. Rank
                    // zero therefore reports even M coordinates and one
                    // complete 256-feature A pair; N already names one
                    // complete 256-feature B pair. Publish each pair as soon
                    // as its final TMA reader retires, allowing dW13 operand
                    // quantization to pipeline behind live dW2 UMMA/TMA.
                    DG_DEVICE_ASSERT(m_block % 2u == 0u);
                    const uint32_t a_pair = m_block / 2u;
                    const uint32_t b_pair = n_block;
                    DG_DEVICE_ASSERT(a_pair < kHidden / 256u);
                    DG_DEVICE_ASSERT(
                        b_pair < kIntermediateHidden / 256u);

                    const uint32_t a_previous = ptx::atomic_add_acq_rel(
                        pair_counters + a_pair, 1u);
                    constexpr uint32_t kAConsumers =
                        kIntermediateHidden / 256u;
                    DG_DEVICE_ASSERT(a_previous < kAConsumers);
                    if (a_previous + 1u == kAConsumers) {
                        const uint32_t first_panel = 2u * a_pair;
                        ptx::red_or_rel_gpu(
                            feature_masks + first_panel / 32u,
                            3u << (first_panel % 32u));
                    }

                    const uint32_t b_previous = ptx::atomic_add_acq_rel(
                        pair_counters + kHidden / 256u + b_pair, 1u);
                    constexpr uint32_t kBConsumers = kHidden / 256u;
                    DG_DEVICE_ASSERT(b_previous < kBConsumers);
                    if (b_previous + 1u == kBConsumers) {
                        const uint32_t first_panel =
                            kHidden / 128u + 2u * b_pair;
                        ptx::red_or_rel_gpu(
                            feature_masks + first_panel / 32u,
                            3u << (first_panel % 32u));
                    }
                }
            };
            if (backward_ranges.num_ranges == 3u) {
              if constexpr (!kK3MxFp8DW13Hybrid) {
                // Decode all three immutable physical ranges as one logical K
                // reduction.  The BF16 body keeps one FP32 accumulator and
                // one retained UMMA/TMA resource lifetime, eliminating two
                // intermediate BF16 stores/reloads and grid barriers 31/32.
                using ThreeSegmentReadyDW2Provider = sched::
                    ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider<
                        kWgradBlockM, kWgradBlockN,
                        2, false, kNumSMs,
                        kHidden, kIntermediateHidden,
                        kReadyThreeSegmentBatchTasks,
                        kReadyDW2TasksPerExpert,
                        BLOCK_M, kWgradBlockK,
                        kReadyPoolPrefixWord,
                        kReadyActiveExpertWord,
                        kReadyRangeStateStride>;
                static_assert(
                    ThreeSegmentReadyDW2Provider::kCompleteAcquireMask ==
                        kReadyCompleteRoleMask,
                    "three-segment dW2 scheduler role mask drifted");
                static_assert(
                    kPublishRemoteGradients,
                    "three-segment dW2 must retain the full-grid remote "
                    "publication edge that orders W13 reads before dW13");
                const uint32_t second_range_index =
                    backward_ranges.reverse_range_index(1u);
                const uint32_t third_range_index =
                    backward_ranges.reverse_range_index(2u);
                DG_DEVICE_ASSERT(
                    second_range_index == third_range_index + 1u);
                auto* const union_state =
                    weight_tile_states + kReadyTerminalUnionStateWord;
                const auto* const second_state =
                    weight_tile_states +
                    second_range_index * kReadyRangeStateStride;
                auto* const dw2_mailbox =
                    union_state + kReadyDW2ClusterSlotWord +
                    ready_cluster_idx * kReadyClusterSlotWords;
                const sched::ExternalKGroupedTerminalThreeSegmentRangeStream
                    dw2_stream{
                        union_state,
                        second_state,
                        union_state + kReadyDW2CursorWord,
                        union_state[kReadyDW2TasksWord],
                        dw2_mailbox,
                        kReadyThreeSegmentBatchTasks,
                        kReadyDW2TasksPerExpert,
                    };
                run_ready_wgrad_range.template operator()<
                    ThreeSegmentReadyDW2Provider,
                    InitializeRetainResources,
                    kPublishRemoteGradients,
                    0u, false, false>(
                        kHidden, kIntermediateHidden,
                        tensor_map_w2_wgrad_a,
                        tensor_map_w2_wgrad_b,
                        tensor_map_w2_wgrad_d,
                        dw2_stream, false,
                        no_input_tile_retired,
                        no_background_work);
              } else {
                // A compact third descriptor does not grow the two-segment
                // stream ABI. Drain all three immutable queues in reverse
                // order inside one retained BF16 resource lifetime. The first
                // call overwrites dW2; the following calls accumulate.
                using ThreeRangeReadyDW2Provider =
                    sched::ExternalKGroupedDynamicRangeProvider<
                        kWgradBlockM, kWgradBlockN,
                        2, false, kNumSMs,
                        kHidden, kIntermediateHidden,
                        BLOCK_M, kWgradBlockK,
                        kReadyPoolPrefixWord,
                        kReadyActiveExpertWord>;
                static_assert(
                    ThreeRangeReadyDW2Provider::kCompleteAcquireMask ==
                        kReadyCompleteRoleMask,
                    "three-range dW2 scheduler role mask drifted");
                const auto run_three_range_dw2 = [&]<
                    typename ResourceHooks,
                    bool kRangeAccumulate,
                    bool kFusePublication>(
                        const uint32_t reverse_iteration) {
                    const uint32_t range_index =
                        backward_ranges.reverse_range_index(
                            reverse_iteration);
                    auto* const range_state =
                        weight_tile_states +
                        range_index * kReadyRangeStateStride;
                    auto* const mailbox =
                        range_state + kReadyDW2ClusterSlotWord +
                        ready_cluster_idx * kReadyClusterSlotWords;
                    const sched::ExternalKGroupedRangeStream stream{
                        range_state,
                        0u, 0u,
                        range_state + kReadyDW2CursorWord,
                        0u,
                        mailbox,
                        DG_EXPERIMENTAL_K3_MULTI_RANGE_DW2_BATCH_TASKS,
                        kReadyDW2TasksPerExpert,
                        nullptr, 0u,
                        range_state + kReadyEpochWord,
                        k3_multirange_epoch(
                            K3MultiRangeBackwardPhase::W2Wgrad,
                            backward_ranges.epoch_seed(range_index)),
                        range_state + kReadyDW2TasksWord,
                    };
                    const auto hybrid_background_work = [
                            expert_counts, hybrid_backward_ranges,
                            num_acts_rows, acts_ptr,
                            hybrid_tensor_map_pack, hybrid_state,
                            mailbox,
                            hybrid_scale_arena_aliases_dw2_source] (
                            const uint32_t background_warp_idx,
                            const uint32_t background_lane_idx) {
                        if constexpr (
                            kK3MxFp8DW13Hybrid &&
                            !kK3MxFp8ExactEpilogueRing) {
                            // A no-allocation launch may lend the first half of
                            // saved-down BF16 storage to the exact scale arena.
                            // In that mode, terminal dW13 starts quantization
                            // only after every dW2 read has retired. A distinct
                            // activation ring keeps the normal dW2 overlap.
                            if (hybrid_scale_arena_aliases_dw2_source)
                                return;
                            detail::
                            k3_mxfp8_stream_dw13_operands_background<
                                kHidden, kIntermediateHidden,
                                kNumExperts, BLOCK_M,
                                kNumSMs, kNumThreads,
                                true, 12u, 1u,
                                kHybridDW13QuantScratchBegin,
                                kReadyCompleteRoleMask>(
                                    expert_counts, hybrid_backward_ranges,
                                    num_acts_rows, acts_ptr,
                                    hybrid_tensor_map_pack,
                                    hybrid_state, smem_buffer,
                                    background_warp_idx,
                                    background_lane_idx,
                                    mailbox);
                        }
                    };
                    if constexpr (kK3MxFp8DW13Hybrid) {
                        run_ready_wgrad_range.template operator()<
                            ThreeRangeReadyDW2Provider,
                            ResourceHooks,
                            kFusePublication,
                            0u, false, kRangeAccumulate,
                            kEarlyDW2WgradStages>(
                                kHidden, kIntermediateHidden,
                                tensor_map_w2_wgrad_a,
                                tensor_map_w2_wgrad_b,
                                tensor_map_w2_wgrad_d,
                                stream, false,
                                hybrid_input_tile_retired,
                                hybrid_background_work);
                    } else {
                        run_ready_wgrad_range.template operator()<
                            ThreeRangeReadyDW2Provider,
                            ResourceHooks,
                            kFusePublication,
                            0u, false, kRangeAccumulate>(
                                kHidden, kIntermediateHidden,
                                tensor_map_w2_wgrad_a,
                                tensor_map_w2_wgrad_b,
                                tensor_map_w2_wgrad_d,
                                stream, false,
                                no_input_tile_retired,
                                no_background_work);
                    }
                };
                run_three_range_dw2.template operator()<
                    InitializeRetainResources, false,
                    kPublishRemoteGradients>(0u);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount, 31u);
                run_three_range_dw2.template operator()<
                    RetainExistingResources, true, false>(1u);
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount, 32u);
                if constexpr (kK3MxFp8DW13Hybrid) {
                    run_three_range_dw2.template operator()<
                        ReleaseExistingResources, true, false>(2u);
                } else {
                    run_three_range_dw2.template operator()<
                        RetainExistingResources, true, false>(2u);
                }
              }
            } else {
                // One output task traverses both disjoint K slices. A
                // one-range launch binds an unused physical arena as an empty
                // second segment to preserve the same specialization.
                const uint32_t first_range_index =
                    backward_ranges.reverse_range_index(0u);
                const uint32_t second_range_index =
                    backward_ranges.num_ranges > 1u
                    ? backward_ranges.reverse_range_index(1u)
                    : (first_range_index + 1u) % kK3MaxBackwardRanges;
                auto* const union_state =
                    weight_tile_states + kReadyTerminalUnionStateWord;
                const auto* const second_state =
                    weight_tile_states +
                    second_range_index * kReadyRangeStateStride;
                auto* const dw2_mailbox =
                    union_state + kReadyDW2ClusterSlotWord +
                    ready_cluster_idx * kReadyClusterSlotWords;
                const sched::ExternalKGroupedTwoSegmentRangeStream dw2_stream{
                    {
                        union_state,
                        0u, 0u,
                        union_state + kReadyDW2CursorWord,
                        0u,
                        dw2_mailbox,
                        DG_EXPERIMENTAL_K3_MULTI_RANGE_DW2_BATCH_TASKS,
                        kReadyDW2TasksPerExpert,
                        nullptr, 0u,
                        union_state + kReadyEpochWord,
                        ready_terminal_wgrad_epoch(),
                        union_state + kReadyDW2TasksWord,
                    },
                    second_state,
                    kK3MxFp8DW13Hybrid &&
                            !kK3MxFp8ExactEpilogueRing
                        ? hybrid_state + K3MxFp8OverlapState::
                              kDW13QuantExpertsDone
                        : nullptr,
                    kK3MxFp8DW13Hybrid &&
                            !kK3MxFp8ExactEpilogueRing
                        ? kK3MxFp8DW13ShepherdClusters : 0u,
                };
                const auto hybrid_background_work = [
                        expert_counts, hybrid_backward_ranges,
                        num_acts_rows, acts_ptr,
                        hybrid_tensor_map_pack, hybrid_state,
                        dw2_mailbox,
                        hybrid_scale_arena_aliases_dw2_source] (
                        const uint32_t background_warp_idx,
                        const uint32_t background_lane_idx) {
                    if constexpr (
                        kK3MxFp8DW13Hybrid &&
                        !kK3MxFp8ExactEpilogueRing) {
                        if (hybrid_scale_arena_aliases_dw2_source)
                            return;
                        detail::k3_mxfp8_stream_dw13_operands_background<
                            kHidden, kIntermediateHidden,
                            kNumExperts, BLOCK_M, kNumSMs, kNumThreads,
                            true, 12u, 1u,
                            kHybridDW13QuantScratchBegin,
                            kReadyCompleteRoleMask>(
                                expert_counts, hybrid_backward_ranges,
                                num_acts_rows, acts_ptr,
                                hybrid_tensor_map_pack,
                                hybrid_state, smem_buffer,
                                background_warp_idx,
                                background_lane_idx,
                                dw2_mailbox);
                    }
                };
                if constexpr (kK3MxFp8DW13Hybrid) {
                    static_assert(
                        kPublishRemoteGradients,
                        "Hybrid exact dW13 requires cross-rank dX publication");
                    // Defer the blocking rank rendezvous to dW13's reduction
                    // warps. BF16 dW2 can release its local resources as soon
                    // as UMMA/TMA retires, while exact compute overlaps the
                    // publication required by the terminal combine.
                    run_ready_wgrad_range.template operator()<
                        TwoSegmentReadyDW2Provider,
                        InitializeReleaseResources,
                        false,
                        0u, false, false,
                        kEarlyDW2WgradStages>(
                            kHidden, kIntermediateHidden,
                            tensor_map_w2_wgrad_a,
                            tensor_map_w2_wgrad_b,
                            tensor_map_w2_wgrad_d,
                            dw2_stream, false,
                            hybrid_input_tile_retired,
                            hybrid_background_work);
                } else {
                    // The progressive two-range suffix has one scheduler and
                    // one BF16 resource lifetime for both descriptors.  W13
                    // retirement words become expert-local dW13 cursors only
                    // after the final packed-weight reader release-publishes
                    // the exact two-segment target.  Until then every cluster
                    // can claim dW2 from the ordinary union cursor.  The body
                    // selects dW13 descriptors only for high-bit-tagged tasks,
                    // while each task preserves first-range then second-range
                    // accumulation in one FP32 accumulator.
                    static_assert(
                        !kK3TwoSegmentBF16ProgressiveWgrad ||
                            (kPublishRemoteGradients &&
                             kDirectRemoteGradX),
                        "progressive unified wgrad owns remote "
                        "publication/reduction");
                    if constexpr (kK3TwoSegmentBF16ProgressiveWgrad) {
                        const sched::
                            ExternalKGroupedTwoSegmentRangeStream
                                unified_wgrad_stream{
                                    {
                                        union_state,
                                        0u, 0u,
                                        union_state +
                                            kReadyDW2CursorWord,
                                        0u,
                                        dw2_mailbox,
                                        DG_EXPERIMENTAL_K3_MULTI_RANGE_DW2_BATCH_TASKS,
                                        kReadyDW2TasksPerExpert,
                                        union_state +
                                            kReadyW13RetiredWord,
                                        kNumW13DgradBlockNs / 2u,
                                        union_state +
                                            kReadyEpochWord,
                                        ready_terminal_wgrad_epoch(),
                                        union_state +
                                            kReadyDW2TasksWord,
                                    },
                                    second_state,
                                };
                        run_ready_wgrad_range.template operator()<
                            TwoSegmentUnifiedBF16WgradProvider,
                            InitializeReleaseResources,
                            kPublishRemoteGradients,
                            0u,
                            true,
                            false>(
                                kHidden, kIntermediateHidden,
                                tensor_map_w2_wgrad_a,
                                tensor_map_w2_wgrad_b,
                                tensor_map_w2_wgrad_d,
                                unified_wgrad_stream,
                                true,
                                no_input_tile_retired,
                                no_background_work,
                                &tensor_map_w13_wgrad_a,
                                &tensor_map_w13_wgrad_b,
                                &tensor_map_w13_wgrad_d);
                        trace_end(17);
                        trace_end(0);
                        return;
                    } else {
                        run_ready_wgrad_range.template operator()<
                            TwoSegmentReadyDW2Provider,
                            InitializeRetainResources,
                            kPublishRemoteGradients,
                            0u,
                            false,
                            false>(
                                kHidden, kIntermediateHidden,
                                tensor_map_w2_wgrad_a,
                                tensor_map_w2_wgrad_b,
                                tensor_map_w2_wgrad_d,
                                dw2_stream,
                                false,
                                no_input_tile_retired,
                                no_background_work);
                    }
                }
            }
            trace_end(17);

#if DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
            if constexpr (kK3MxFp8DW13Hybrid) {
                // BF16 dW2 releases every local mbarrier/TMEM object without
                // waiting for a rank rendezvous. Transition each completed
                // cluster immediately: exact dW13's reduction warps publish
                // remote dX and then combine it while UMMA/TMA and operand
                // quantization run on their disjoint roles.
                trace_begin(19);
                if constexpr (kK3MxFp8ExactEpilogueRing) {
                    // dW13 was computed by the early A/B ring stream.  Keep
                    // only the communication tail that must follow W13 dgrad's
                    // remote writes; re-entering the grouped body would both
                    // duplicate dW13 and dereference the reclaimed ring state.
                    constexpr uint32_t kReduceFirstWarp = 8u;
                    constexpr uint32_t kReduceWarps = 4u;
                    if (warp_idx >= kReduceFirstWarp &&
                        warp_idx < kReduceFirstWarp + kReduceWarps) {
                        const uint32_t reduce_warp =
                            warp_idx - kReduceFirstWarp;
                        if (backward_ranges.num_ranges != 3u) {
                            constexpr uint32_t kPublishThreads =
                                kReduceWarps * 32u;
                            constexpr uint32_t kGridSyncIndex = 2u;
                            constexpr uint32_t kBarrierTag = 7u;
                            constexpr uint32_t kNamedBarrier = 15u;
                            const uint32_t publish_thread =
                                reduce_warp * 32u + lane_idx;
                            comm::nvlink_barrier<
                                kNumRanks, kNumSMs, kPublishThreads,
                                kGridSyncIndex, kBarrierTag>(
                                    backward_workspace,
                                    backward_sym_buffer,
                                    blockIdx.x, publish_thread,
                                    [=]() {
                                        ptx::sync_aligned(
                                            kPublishThreads,
                                            kNamedBarrier);
                                    },
                                    true, true);
                        }
                        k3_mxfp8_wgrad_fixed_topk_combine<
                            kNumSMs, kReduceWarps>(
                                backward_grad_x_output,
                                backward_grad_y,
                                &backward_ranges,
                                num_backward_tokens,
                                backward_workspace
                                    .num_max_tokens_per_rank,
                                num_topk, kHidden,
                                reduce_warp, lane_idx);
                    }
                    __syncthreads();
                } else {
                    const uint32_t dw13_epoch =
                        launch_epoch ^ 0x80000000u;
                    k3_mxfp8_run_dynamic_dw13_overlap<
                        kHidden, kIntermediateHidden,
                        kNumExperts, BLOCK_M, kNumSMs, kNumRanks,
                        kNumThreads,
                        false, kK3MxFp8WgradBatchTasks,
                        true, false, true, true, false>(
                            k3_mxfp8_dw13_hybrid_state(), dw13_epoch,
                            &tensor_map_mxfp8_wgrad_pack,
                            tensor_map_w13_wgrad_slot_d.exact_output_map,
                            &backward_sym_buffer, &backward_workspace,
                            smem_buffer,
                            expert_counts, backward_ranges,
                            num_acts_rows, acts_ptr,
                            w2_dequant_scratch, w13_dequant_scratch,
                            clear_empty_wgrad_expert_outputs,
                            backward_grad_x_output, backward_grad_y,
                            num_backward_tokens,
                            combine_first_range_tokens,
                            combine_second_range_begin,
                            backward_workspace.num_max_tokens_per_rank,
                            num_topk);
                }
                trace_end(19);
                trace_end(0);
                return;
            }
#endif
        }

        if constexpr (
                kMultiRangeBackward &&
                !kK3TwoSegmentBF16ProgressiveWgrad &&
                !kK3BranchMajorBF16DynamicTail) {
            static_assert(
                kReadyWgradSchedule && kExactSourceX &&
                    kDirectRemoteGradX && !kAccumulateWgrad,
                "multi-range W13 requires exact ready-wgrad training");

            if (backward_ranges.num_ranges != 3u) {
                // Conservative one/two-range schedules retain their explicit
                // all-range retirement edge.  The unified three-segment path
                // cleared empty experts before phase 14 and transitions from
                // dW2 to dW13 after the BF16 body's cluster completion join.
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount, 24);

                constexpr uint64_t kBF16PerVector =
                    sizeof(uint4) / sizeof(cd_dtype_t);
                constexpr uint64_t kW13VectorsPerExpert =
                    static_cast<uint64_t>(2u * kIntermediateHidden) *
                    kHidden / kBF16PerVector;
                auto* w13_vectors =
                    reinterpret_cast<uint4*>(w13_dequant_scratch);
                const uint4 zero = {0u, 0u, 0u, 0u};
                const uint64_t global_thread =
                    static_cast<uint64_t>(blockIdx.x) * kNumThreads +
                    threadIdx.x;
                constexpr uint64_t kGlobalThreads =
                    static_cast<uint64_t>(kNumSMs) * kNumThreads;
                #pragma unroll 1
                for (uint32_t expert_idx = 0u;
                     expert_idx < kNumExperts; ++expert_idx) {
                    bool active_in_union = false;
                    #pragma unroll 1
                    for (uint32_t range_iteration = 0u;
                         range_iteration < backward_ranges.num_ranges;
                         ++range_iteration) {
                        const uint32_t range_index =
                            backward_ranges.reverse_range_index(
                                range_iteration);
                        active_in_union |= __ldg(
                            expert_counts +
                            backward_ranges.expert_counts_begin(
                                range_index, kNumExperts) +
                            expert_idx) != 0;
                    }
                    if (active_in_union ||
                        !clear_empty_wgrad_expert_outputs)
                        continue;
                    for (uint64_t linear = global_thread;
                         linear < kW13VectorsPerExpert;
                         linear += kGlobalThreads) {
                        w13_vectors[
                            static_cast<uint64_t>(expert_idx) *
                                kW13VectorsPerExpert +
                            linear] = zero;
                    }
                }

                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                full_grid_phase_barrier(kTraceSiteCount, 25);
            }

            // dW2 retained the embedded BF16 barriers, TMEM allocation, and
            // 64-register role budget across this publication edge. dW13
            // traverses the same two K segments as one FP32 accumulation,
            // performs one BF16 store, and releases the shared lifetime. The
            // progressive path has already reduced remote dX under dW2; the
            // conservative path retains its terminal dW13 combine roles.
            using TwoSegmentReadyDW13Provider =
                sched::ExternalKGroupedTwoSegmentRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    2u * kIntermediateHidden, kHidden,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord,
                    16u,
                    true, 4u>;
            using ReleaseResources =
                Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<>;
            trace_begin(19);
#if DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID
            if constexpr (
                kK3MxFp8DW13Hybrid &&
                !kK3MxFp8ExactEpilogueRing) {
                // BF16 dW2 has invalidated every embedded barrier, freed its
                // base-zero TMEM allocation, and cluster-joined through
                // InitializeReleaseResources.  Open a distinct generation;
                // the exact body allocates/releases a fresh 512-column TMEM
                // lifetime and consumes 14*dgrad+1 operand-ready credits.
                const uint32_t dw13_epoch =
                    launch_epoch ^ 0x80000000u;
                auto* const hybrid_state =
                    k3_mxfp8_dw13_hybrid_state();
                if (blockIdx.x == 0u && threadIdx.x == 0u) {
                    asm volatile(
                        "st.release.gpu.global.u32 [%0], %1;"
                        :: "l"(
                               hybrid_state +
                               K3MxFp8OverlapState::kDW13Epoch),
                           "r"(dw13_epoch)
                        : "memory");
                }
                // This conservative fallback already crossed the earlier
                // publication edge. Terminal dW13 performs only the sole
                // fixed-top-k reduction and must not signal peers twice.
                k3_mxfp8_run_dynamic_dw13_overlap<
                    kHidden, kIntermediateHidden,
                    kNumExperts, BLOCK_M, kNumSMs, kNumRanks,
                    kNumThreads,
                    false, kK3MxFp8WgradBatchTasks,
                    false, false, true, false,
                    kK3MxFp8ExactEpilogueRing>(
                        hybrid_state, dw13_epoch,
                        &tensor_map_mxfp8_wgrad_pack,
                        tensor_map_w13_wgrad_slot_d.exact_output_map,
                        &backward_sym_buffer, &backward_workspace,
                        smem_buffer,
                        expert_counts, backward_ranges,
                        num_acts_rows, acts_ptr,
                        w2_dequant_scratch, w13_dequant_scratch,
                        false,
                        backward_grad_x_output, backward_grad_y,
                        num_backward_tokens,
                        combine_first_range_tokens,
                        combine_second_range_begin,
                        backward_workspace.num_max_tokens_per_rank,
                        num_topk);
                trace_end(19);
                trace_end(0);
                return;
            }
#endif
            if (backward_ranges.num_ranges == 3u) {
                // Complete the same retained BF16 resource generation opened
                // by unified dW2.  One logical task traverses all three K
                // segments, stores dW13 once, and releases TMEM/barriers once.
                using ThreeSegmentReadyDW13Provider = sched::
                    ExternalKGroupedTerminalThreeSegmentDynamicRangeProvider<
                        kWgradBlockM, kWgradBlockN,
                        2, false, kNumSMs,
                        2u * kIntermediateHidden, kHidden,
                        kReadyThreeSegmentBatchTasks,
                        kReadyDW13TasksPerExpert,
                        BLOCK_M, kWgradBlockK,
                        kReadyPoolPrefixWord,
                        kReadyActiveExpertWord,
                        kReadyRangeStateStride>;
                static_assert(
                    ThreeSegmentReadyDW13Provider::kCompleteAcquireMask ==
                        kReadyCompleteRoleMask,
                    "three-segment dW13 scheduler role mask drifted");
                const uint32_t ready_cluster_idx = blockIdx.x / 2u;
                const uint32_t second_range_index =
                    backward_ranges.reverse_range_index(1u);
                const uint32_t third_range_index =
                    backward_ranges.reverse_range_index(2u);
                DG_DEVICE_ASSERT(
                    second_range_index == third_range_index + 1u);
                auto* const union_state =
                    weight_tile_states + kReadyTerminalUnionStateWord;
                const auto* const second_state =
                    weight_tile_states +
                    second_range_index * kReadyRangeStateStride;
                auto* const dw13_mailbox =
                    union_state + kReadyDW13ClusterSlotWord +
                    ready_cluster_idx * kReadyClusterSlotWords;
                const sched::ExternalKGroupedTerminalThreeSegmentRangeStream
                    dw13_stream{
                        union_state,
                        second_state,
                        union_state + kReadyDW13CursorWord,
                        union_state[kReadyDW13TasksWord],
                        dw13_mailbox,
                        kReadyThreeSegmentBatchTasks,
                        kReadyDW13TasksPerExpert,
                    };
                // The generic BF16 reducer has a two-range logical-to-physical
                // token map.  Three true-varlen ranges contain two independent
                // capacity gaps, so use the existing range-set-aware reducer
                // on otherwise idle warps while dW13 UMMA/TMA runs.  dW2's
                // fused publication barrier has already ordered every remote
                // plane; the body-wide completion join orders this reduction
                // before resource release and return.
                constexpr uint32_t kThreeRangeReduceFirstWarp = 8u;
                constexpr uint32_t kThreeRangeReduceWarps = 4u;
                const auto* const three_range_backward_ranges =
                    &backward_ranges;
                const auto reduce_three_range_grad_x = [=](
                        const uint32_t background_warp_idx,
                        const uint32_t background_lane_idx) {
                    if (background_warp_idx >=
                            kThreeRangeReduceFirstWarp &&
                        background_warp_idx <
                            kThreeRangeReduceFirstWarp +
                                kThreeRangeReduceWarps) {
                        k3_mxfp8_wgrad_fixed_topk_combine<
                            kNumSMs, kThreeRangeReduceWarps>(
                                backward_grad_x_output,
                                backward_grad_y,
                                three_range_backward_ranges,
                                num_backward_tokens,
                                backward_workspace.
                                    num_max_tokens_per_rank,
                                num_topk, kHidden,
                                background_warp_idx -
                                    kThreeRangeReduceFirstWarp,
                                background_lane_idx);
                    }
                };
                run_ready_wgrad_range.template operator()<
                    ThreeSegmentReadyDW13Provider,
                    ReleaseResources,
                    false,
                    0u,
                    false,
                    false>(
                        2u * kIntermediateHidden, kHidden,
                        tensor_map_w13_wgrad_a,
                        tensor_map_w13_wgrad_b,
                        tensor_map_w13_wgrad_d,
                        dw13_stream, false,
                        no_input_tile_retired,
                        reduce_three_range_grad_x);
            } else {
                const uint32_t first_range_index =
                    backward_ranges.reverse_range_index(0u);
                const uint32_t second_range_index =
                    backward_ranges.num_ranges > 1u
                    ? backward_ranges.reverse_range_index(1u)
                    : (first_range_index + 1u) % kK3MaxBackwardRanges;
                auto* const union_state =
                    weight_tile_states + kReadyTerminalUnionStateWord;
                const auto* const second_state =
                    weight_tile_states +
                    second_range_index * kReadyRangeStateStride;
                const uint32_t dw13_tasks =
                    union_state[kReadyDW13TasksWord];
                const sched::ExternalKGroupedTwoSegmentRangeStream
                    dw13_stream{
                        {
                            union_state,
                            0u, dw13_tasks,
                        },
                        second_state,
                    };
                run_ready_wgrad_range.template operator()<
                    TwoSegmentReadyDW13Provider,
                    ReleaseResources,
                    kDirectRemoteGradX,
                    kDirectRemoteGradX ? 64u : 0u,
                    false,
                    false>(
                        2u * kIntermediateHidden, kHidden,
                        tensor_map_w13_wgrad_a,
                        tensor_map_w13_wgrad_b,
                        tensor_map_w13_wgrad_d,
                        dw13_stream, kDirectRemoteGradX,
                        no_input_tile_retired,
                        no_background_work);
            }
            trace_end(19);
            trace_end(0);
            return;
        }
#endif

#if DG_EXPERIMENTAL_K3_READY_WGRAD
        if constexpr (
                kReadyWgradSchedule &&
                !kK3BranchMajorBF16DynamicTail) {
            trace_begin(20);

            // Every role has exhausted its cluster-local W13 task stream. End
            // the parent dgrad resource lifetime exactly once, then enter one
            // persistent BF16 body per descriptor. Initial-consumer clusters
            // reach dW2 immediately; W13 producers join the same dynamic task
            // stream after exhausting their dgrad work. v410's materializers
            // (hardware named barriers 13/14) retired at the first dgrad phase
            // join, and v409 dispatch (barrier 15) retired at the earlier
            // route/dispatch grid join. The BF16 combine role may therefore
            // reuse barrier 15 only in the uniform suffix below.
            comm::cluster_sync_with_relaxed_arrive();
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
                #pragma unroll
                for (uint32_t i = 0;
                     i < kNumDispatchBarriers; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dispatch_barriers[i]));
                }
                // Unlike v403's shared full-barrier slots, v410's concurrent
                // W2/W13 materializers own two dedicated transaction
                // barriers. Their work is long retired, but the objects remain
                // valid and sit inside the BF16 body's aliased A/B/CD range.
                // Invalidate both before the first retained-resource call.
                #pragma unroll
                for (uint32_t i = 0; i < 2u; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dequant_barriers + i));
                }
            }
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0)
                Allocator().free(0, kNumTmemCols);
            __syncthreads();

            // The ordinary ready-wgrad suffix is terminal and all eight
            // warpgroups arrive with the parent's uniform 40-register budget.
            // Allocate the full legal callable ceiling once for both outlined
            // BF16 wgrad bodies so their scheduler/TMA state does not spill.
            static_assert(
                !kUseReducedW2ProducerSet,
                "ready-wgrad register entry state changed");
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();

            const uint32_t ready_cluster_idx = blockIdx.x / 2u;
            using ReadyDW2Provider =
                sched::ExternalKGroupedDynamicRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    kHidden, kIntermediateHidden,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord>;
            using ReadyDW13Provider =
                sched::ExternalKGroupedDynamicRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    2u * kIntermediateHidden, kHidden,
                    BLOCK_M, kWgradBlockK,
                    kReadyPoolPrefixWord,
                    kReadyActiveExpertWord>;
            // Both descriptor phases use the same BLOCK_M/N/K, stage count,
            // shared control layout, and base-zero two-SM TMEM allocation.
            // dW2 initializes that lifetime but deliberately retains it;
            // dW13 starts only after dW2's body-wide cluster completion join,
            // reuses the phase-reset barriers/TMEM, then invalidates and frees
            // them with a post-release cluster join.
            using ReadyDW2RetainedResources =
                Sm100Bf16GemmBatchResourceHooks<true, false>;
            using ReadyDW13ReleaseResources =
                Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<>;
            static_assert(
                ReadyDW2RetainedResources::kInitializeBatchResources &&
                    !ReadyDW2RetainedResources::kReleaseBatchResources,
                "dW2 must initialize and retain BF16 batch resources");
            static_assert(
                !ReadyDW13ReleaseResources::kInitializeBatchResources &&
                    ReadyDW13ReleaseResources::kReleaseBatchResources &&
                    ReadyDW13ReleaseResources::kSynchronizeAfterRelease,
                "dW13 must reuse, release, and hand off BF16 resources");
            static_assert(
                ReadyDW2Provider::kCompleteAcquireMask ==
                    kReadyCompleteRoleMask,
                "dW2 scheduler role mask drifted from mailbox initialization");
            static_assert(
                ReadyDW13Provider::kCompleteAcquireMask ==
                    kReadyCompleteRoleMask,
                "dW13 scheduler role mask drifted from mailbox initialization");

            auto* dw2_mailbox =
                weight_tile_states + kReadyDW2ClusterSlotWord +
                ready_cluster_idx * kReadyClusterSlotWords;
            auto* dw13_mailbox =
                weight_tile_states + kReadyDW13ClusterSlotWord +
                ready_cluster_idx * kReadyClusterSlotWords;
            const uint32_t dw2_total = weight_tile_states[
                kReadyDW2TasksWord];
            const sched::ExternalKGroupedRangeStream dw2_stream{
                weight_tile_states,
                0u, 0u,
                weight_tile_states + kReadyDW2CursorWord,
                dw2_total,
                dw2_mailbox,
                kReadyBatchTasks,
                kReadyDW2TasksPerExpert};

            // A readiness plane removes the old device-wide NVLink join, so
            // the otherwise-idle dW2 combine warps may consume tokens while
            // producer clusters are still scattering W13 dgrad. Keeping this
            // work in dW13 would start it only after the local producer join.
            trace_begin(17);
            run_ready_wgrad_range.template operator()<
                ReadyDW2Provider, ReadyDW2RetainedResources,
                kStreamingDirectGradXCombine,
                kStreamingDirectGradXCombine ? 64u : 0u,
                kStreamingDirectGradXCombine>(
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    dw2_stream,
                    kStreamingDirectGradXCombine,
                    no_input_tile_retired,
                    no_background_work);
            trace_end(17);

            // dW13 owns a disjoint mailbox initialized at launch. A delayed
            // scheduler-role acknowledgement from dW2 can therefore never
            // create a stale sequence/mask ABA at this descriptor boundary.
            // No global join is needed: dW13's scheduler waits on exact
            // per-expert W13 retirement counters. The readiness-driven direct
            // dX combine has already completed inside retained dW2.
            const uint32_t dw13_total = weight_tile_states[
                kReadyDW13TasksWord];
            const sched::ExternalKGroupedRangeStream dw13_stream{
                weight_tile_states,
                0u, 0u,
                weight_tile_states + kReadyDW13CursorWord,
                dw13_total,
                dw13_mailbox,
                kReadyBatchTasks,
                kReadyDW13TasksPerExpert,
                weight_tile_states + kReadyW13RetiredWord,
                kNumW13DgradBlockNs / 2u};
            trace_begin(19);
            run_ready_wgrad_range.template operator()<
                ReadyDW13Provider, ReadyDW13ReleaseResources,
                false, 0u, false>(
                    2u * kIntermediateHidden, kHidden,
                    tensor_map_w13_wgrad_a,
                    tensor_map_w13_wgrad_b,
                    tensor_map_w13_wgrad_d,
                    dw13_stream, false,
                    no_input_tile_retired,
                    no_background_work);
            trace_end(19);
            trace_end(20);
            trace_end(0);
            return;
        }
#endif

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
        if constexpr (
                kClearWgradPadding && !kK3MxFp8WgradOverlap &&
                !(kK3BranchMajorBF16DynamicTail &&
                  kMultiRangeBackward))
            clear_wgrad_padding_rows();

        // This compile-time combination identifies the first chunk whose
        // non-inline dequant workspaces are the uninitialized dW outputs.
        // Inline fallback is used only after live accumulation state exists;
        // phase-ordered dequant aliases consumed activation storage instead.
#if DG_EXPERIMENTAL_K3_RANGE_WGRAD
        if constexpr (kInlineWgrad && !kK3MxFp8WgradOverlap) {
            DG_STATIC_ASSERT(
                kHidden == 3584u && kIntermediateHidden == 3072u &&
                kNumExperts == 112u && BLOCK_M == 192u &&
                kNumSMs == 148u,
                "Experimental range scheduling is restricted to K3 EP8");
            DG_STATIC_ASSERT(
                kWgradBlockK == 64u,
                "K3 range metadata assumes 64-row BF16 K alignment");
            // The residual production specialization computes dW2 before
            // W13 dgrad so its compact output can back the residual-
            // activation workspace.  Do not replay that gradient here.  Its
            // deferred remote-store publication is instead fused into the
            // mandatory first dW13 range below, before the fixed-top-k dX
            // reduction, exactly as in the legacy grouped-wgrad schedule.

            // No CTA may reuse the global weight-readiness prefix until all
            // dgrad and padding writers have retired. Publish every writer's
            // generic stores to TMA, then perform the uniform pre-reuse grid
            // join. initialize_overlap_state performs the matching second
            // grid join after block zero publishes the compact range state.
            asm volatile(
                "fence.proxy.async.global;" ::: "memory");
            __threadfence();
            __syncthreads();
            full_grid_phase_barrier(kTraceSiteCount, 27);

            // Retire the enclosing dgrad pipeline before the first embedded
            // BF16 range body aliases its shared-memory barriers or requests
            // a new base-zero two-SM TMEM allocation. This is the same
            // lifecycle handoff used by the legacy inline-wgrad branch.
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
                            primary_mma_barrier));
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

            // The active-only range provider never writes empty experts.
            // Preserve the legacy first-chunk clear after every aliased W13
            // cache reader has retired and before range TMA can observe dW.
            if constexpr (
                kClearWgradPadding && !kInlineWeightDequant &&
                !kPhaseOrderedWeightDequant) {
                if (clear_empty_wgrad_expert_outputs)
                    clear_empty_wgrad_experts();
            }

            // Padding and empty-expert operands were written through the
            // generic proxy by every participating thread. Each writer must
            // publish its own stores to TMA before the state initializer's
            // full-grid barrier releases any range-body TMA producer.
            asm volatile(
                "fence.proxy.async.global;" ::: "memory");
            __threadfence();
            __syncthreads();

            initialize_overlap_state();

            using DW2RangeProvider =
                sched::ExternalKGroupedRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    kHidden, kIntermediateHidden,
                    BLOCK_M, kWgradBlockK>;
            using DW13RangeProvider =
                sched::ExternalKGroupedRangeProvider<
                    kWgradBlockM, kWgradBlockN,
                    2, false, kNumSMs,
                    2u * kIntermediateHidden, kHidden,
                    BLOCK_M, kWgradBlockK>;

            constexpr uint32_t kNumClusters = kNumSMs / 2u;
            const auto cluster_idx = blockIdx.x / 2u;
            const auto* overlap_state = weight_tile_states;
            const auto run_ranges = [&]<
                typename TaskProvider,
                uint32_t kRangeBatchTasks,
                bool kFuseFirstRange,
                uint32_t kFirstRangeExtraCombineThreads = 0,
                bool kFirstRangePublishBeforeReduce = false>(
                const uint32_t total_tasks,
                const uint32_t shape_m,
                const uint32_t shape_n,
                const cute::TmaDescriptor& tensor_map_a,
                const cute::TmaDescriptor& tensor_map_b,
                const cute::TmaDescriptor& tensor_map_d,
                const bool combine_reduce) {
                DG_STATIC_ASSERT(
                    kRangeBatchTasks % 4u == 0u,
                    "Range batches must reset both epilogue phases");
                DG_STATIC_ASSERT(
                    TaskProvider::kNumClusterTasksPerGroup % 4u == 0u,
                    "Every expert tail must preserve epilogue phase reset");
                const auto run_one_range = [&]<
                    bool kFuseRangeCombine,
                    uint32_t kRangeExtraCombineThreads = 0,
                    bool kRangePublishBeforeReduce = false>(
                    const uint32_t first_task,
                    const uint32_t num_tasks) {
                    uint32_t remaining = num_tasks;
                    uint32_t logical_task = first_task;
                    uint32_t total_k_blocks = 0;
                    while (remaining != 0u) {
                        const auto active_expert_idx =
                            logical_task /
                            TaskProvider::kNumClusterTasksPerGroup;
                        const auto task_in_expert =
                            logical_task %
                            TaskProvider::kNumClusterTasksPerGroup;
                        const auto tasks_in_segment = cute::min(
                            remaining,
                            TaskProvider::kNumClusterTasksPerGroup -
                                task_in_expert);
                        const auto expert = overlap_state[
                            kOverlapActiveExpertWord + active_expert_idx];
                        const auto pool_blocks =
                            overlap_state[
                                kOverlapPoolPrefixWord + expert + 1u] -
                            overlap_state[
                                kOverlapPoolPrefixWord + expert];
                        total_k_blocks +=
                            tasks_in_segment * pool_blocks *
                            (BLOCK_M / kWgradBlockK);
                        logical_task += tasks_in_segment;
                        remaining -= tasks_in_segment;
                    }
                    if (num_tasks != 0u) {
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            num_tasks % 4u == 0u);
                        DG_TRAP_ONLY_DEVICE_ASSERT(
                            total_k_blocks %
                                (2u * kWgradStages) == 0u);
                    }
                    const sched::ExternalKGroupedRangeStream task_stream{
                        overlap_state, first_task, num_tasks};
                    run_wgrad_range.template operator()<
                        TaskProvider,
                        kFuseRangeCombine,
                        kRangeExtraCombineThreads,
                        kRangePublishBeforeReduce>(
                        shape_m, shape_n,
                        tensor_map_a, tensor_map_b, tensor_map_d,
                        task_stream, combine_reduce);
                    // sm100_bf16_gemm_body invalidates barriers and frees TMEM
                    // without a post-cleanup handoff. Both peer CTAs must join
                    // before the next batch aliases shared memory or allocates
                    // TMEM again.
                    comm::cluster_sync_with_relaxed_arrive();
                };

                // Every CTA enters the first rectangular range invocation,
                // including clusters with zero logical work. Therefore the
                // fused NVLink publication/reduction roles retain exact
                // kNumSMs grid participation while overlapping useful UMMA
                // on nonempty clusters. Later ranges contain no grid-wide
                // communication and may finish independently.
                const uint32_t first_task =
                    cluster_idx * kRangeBatchTasks;
                const uint32_t first_num_tasks =
                    first_task < total_tasks
                        ? cute::min(
                              kRangeBatchTasks,
                              total_tasks - first_task)
                        : 0u;
                run_one_range.template operator()<
                    kFuseFirstRange,
                    kFirstRangeExtraCombineThreads,
                    kFirstRangePublishBeforeReduce>(
                        first_task, first_num_tasks);

                for (uint32_t later_first_task =
                         first_task +
                         kNumClusters * kRangeBatchTasks;
                     later_first_task < total_tasks;
                     later_first_task +=
                         kNumClusters * kRangeBatchTasks) {
                    const auto later_num_tasks = cute::min(
                        kRangeBatchTasks,
                        total_tasks - later_first_task);
                    run_one_range.template operator()<false>(
                        later_first_task, later_num_tasks);
                }
            };

            if constexpr (!kEarlyW2Wgrad) {
                trace_begin(17);
                run_ranges.template operator()<
                    DW2RangeProvider, kDW2RangeBatchTasks,
                    kPublishRemoteGradients>(
                    overlap_state[kOverlapDW2TasksWord],
                    kHidden, kIntermediateHidden,
                    tensor_map_w2_wgrad_a,
                    tensor_map_w2_wgrad_b,
                    tensor_map_w2_wgrad_d,
                    false);
                trace_end(17);
            }

            // In the late-dW2 schedule, clusters enter dW13 independently
            // after exhausting their dW2 ranges.  In the production early-dW2
            // schedule, every CTA still enters the mandatory first dW13 range:
            // its combine warps first publish Kernel A's remote stores, then
            // perform the fixed-order receive-plane reduction while nonempty
            // clusters execute dW13 UMMA.  The template flags mirror the
            // already-validated legacy grouped-wgrad branch below.
            trace_begin(19);
            run_ranges.template operator()<
                DW13RangeProvider, kDW13RangeBatchTasks,
                kEarlyW2Wgrad
                    ? kPublishRemoteGradients
                    : kDirectRemoteGradX,
                kDirectRemoteGradX ? 64u : 0u,
                kEarlyW2Wgrad &&
                    kDirectRemoteGradX &&
                    kPublishRemoteGradients>(
                overlap_state[kOverlapDW13TasksWord],
                2u * kIntermediateHidden, kHidden,
                tensor_map_w13_wgrad_a,
                tensor_map_w13_wgrad_b,
                tensor_map_w13_wgrad_d,
                kDirectRemoteGradX);
            trace_end(19);
        }
#else
        if constexpr (kInlineWgrad) {
            if constexpr (
                    kK3BranchMajorBF16DynamicTail &&
                    !kMultiRangeBackward) {
                // Both dgrad waves have retired before this terminal schedule.
                // Reuse the first 960 words of W2's dead readiness arena for a
                // compact expert prefix plus independent dW2/dW13 cursors and
                // four-word mailboxes. The existing full-grid publication edge
                // below releases these writes to every scheduler role.
                DG_STATIC_ASSERT(
                    kReadyStateWords <= kNumW2WeightTileStates,
                    "dynamic BF16 tail exceeds retired W2 state");
                if (blockIdx.x == 0u) {
                    for (uint32_t word = threadIdx.x;
                         word < kReadyStateWords;
                         word += kNumThreads) {
                        weight_tile_states[word] = 0u;
                    }
                    __syncthreads();
                    if (threadIdx.x == 0u) {
                        auto* const pool_prefix =
                            weight_tile_states + kReadyPoolPrefixWord;
                        auto* const active_experts =
                            weight_tile_states + kReadyActiveExpertWord;
                        uint32_t pool_blocks = 0u;
                        uint32_t active_count = 0u;
                        pool_prefix[0] = 0u;
                        #pragma unroll 1
                        for (uint32_t expert = 0u;
                             expert < kNumExperts; ++expert) {
                            const uint32_t count =
                                static_cast<uint32_t>(
                                    __ldg(expert_counts + expert));
                            if (count != 0u)
                                active_experts[active_count++] = expert;
                            pool_blocks += math::ceil_div(count, BLOCK_M);
                            pool_prefix[expert + 1u] = pool_blocks;
                        }
                        weight_tile_states[kReadyMagicWord] =
                            0x4b334454u; // "K3DT"
                        weight_tile_states[kReadyDW2CursorWord] = 0u;
                        weight_tile_states[kReadyDW13CursorWord] = 0u;
                        weight_tile_states[kReadyActiveCountWord] =
                            active_count;
                        weight_tile_states[kReadyPoolBlocksWord] =
                            pool_blocks;
                        weight_tile_states[kReadyDW2TasksWord] =
                            active_count * kPairedDW2TasksPerExpert;
                        weight_tile_states[kReadyDW13TasksWord] =
                            active_count * kPairedDW13TasksPerExpert;
                        for (uint32_t cluster = 0u;
                             cluster < kReadyNumClusters; ++cluster) {
                            weight_tile_states[
                                kReadyDW2ClusterSlotWord +
                                cluster * kReadyClusterSlotWords] =
                                    kReadyCompleteRoleMask;
                            weight_tile_states[
                                kReadyDW13ClusterSlotWord +
                                cluster * kReadyClusterSlotWords] =
                                    kReadyCompleteRoleMask;
                        }
                    }
                }
            } else if constexpr (
                    kK3BranchMajorBF16DynamicTail &&
                    kMultiRangeBackward) {
                // The ready-state preparation above already published both
                // physical prefixes, the immutable union, independent
                // cursors/mailboxes, and every padded operand.  Do not rebuild
                // or alias either live prefix after W13 dgrad.
                DG_DEVICE_ASSERT(backward_ranges.num_ranges == 2u);
            }
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
            full_grid_phase_barrier(kTraceSiteCount, 28);
        } else {
            __syncthreads();
        }
        // The W13 on-demand MXFP8 cache aliases the future dW destinations.
        // Do not clear empty-expert dW slices until every W13 reader has
        // crossed the retirement barrier above, then publish the clear before
        // grouped wgrad can schedule an arbitrary expert on another CTA.
        if constexpr (
            kClearWgradPadding && !kInlineWeightDequant &&
            !kPhaseOrderedWeightDequant &&
            !(kK3BranchMajorBF16DynamicTail &&
              kMultiRangeBackward)) {
            if (clear_empty_wgrad_expert_outputs)
                clear_empty_wgrad_experts();
        }
        trace_begin(20);
        if constexpr (
            kInlineWgrad && !kK3MxFp8WgradOverlap &&
            !kK3BranchMajorBF16EarlyDW2Overlap) {
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
                        primary_mma_barrier));
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
            if constexpr (
                !kBF16Mode && !kInlineWeightDequant &&
                !kResidualMXFP8Dgrad &&
                (kOverlapInitialBF16WeightDequant ||
                 kPhaseOrderedWeightDequant)) {
                // Concurrent or phase-ordered non-inline materialization owns
                // two dedicated transaction barriers beyond the parent arrays.
                // The terminal BF16 body aliases their shared-memory locations,
                // so retire both objects before that first body initializes its
                // own barrier layout.
                #pragma unroll
                for (uint32_t i = 0; i < 2u; ++i) {
                    Barrier::invalidate(
                        reinterpret_cast<Barrier::ValueType const*>(
                            dequant_barriers + i));
                }
            }
          }
        }
        if constexpr (
            !kK3MxFp8WgradOverlap &&
            !kK3BranchMajorBF16EarlyDW2Overlap) {
            comm::cluster_sync_with_relaxed_arrive();
            if (warp_idx == 0)
                Allocator().free(0, kNumTmemCols);
        }
        __syncthreads();

        if constexpr (
            kK3BranchMajorBF16WgradTail &&
            !kK3BranchMajorBF16EarlyDW2Overlap) {
            // Parent role-divergent work is retired above. Restore the measured
            // uniform register budget before either outlined BF16 body aliases
            // parent SMEM and reclaims the base-zero TMEM allocation.
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();
        }

#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD
        if constexpr (
            kK3MxFp8ThreeTermWgrad && !kK3MxFp8WgradOverlap) {
            // Earlier dgrad roles temporarily deallocate to 40 registers.
            // Their resources and all role-divergent work are retired above;
            // restore one uniform full-CTA budget before declaring the
            // producer/maps and entering the two grouped-UMMA bodies.
            cutlass::arch::warpgroup_reg_alloc<64>();
            __syncthreads();
        }
#endif

        if constexpr (kInlineWgrad) {
            DG_STATIC_ASSERT(
                !kK3BranchMajorBF16WgradTail ||
                    (kWgradBlockM == 128u &&
                     kWgradBlockN == 256u &&
                     kWgradBlockK == 64u &&
                     kWgradRoleThreads == 128u),
                "terminal BF16 UMMA/TMA geometry changed");
#if DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD
          if constexpr (
              kK3MxFp8ThreeTermWgrad && !kK3MxFp8WgradOverlap) {
            constexpr uint32_t kPoolBlockRows = BLOCK_M;
            using Prefix = K3MxFp8WgradPrefixLayout<
                kNumExperts, kK3MaxBackwardRanges>;
            using Overlap = K3MxFp8OverlapState;
            using DW2Config = Sm100K3MxFp8ThreeTermWgradConfig<
                kHidden, kIntermediateHidden,
                kNumExperts, kNumSMs, kAccumulateWgrad>;
            using DW13Config = Sm100K3MxFp8ThreeTermWgradConfig<
                2u * kIntermediateHidden, kHidden,
                kNumExperts, kNumSMs, kAccumulateWgrad>;

            // Seven compile-time-dead parent arguments carry immutable base
            // layouts only.  Three UINT8 layouts describe H/I/2I value
            // boxes.  Four INT32 layouts are required because H is an A
            // scale in dW2 but a B scale in dW13 (256x1 versus 128x1 boxes).
            // Primary/residual addresses are replaced on device into eight
            // phase-local maps; the historical BF16 wgrad A/B descriptors
            // are never consumed by this suffix.
            constexpr uint32_t kTensorMapBytes =
                sizeof(cute::TmaDescriptor);
            constexpr uint32_t kTensorMapWords =
                kTensorMapBytes / sizeof(uint32_t);
            constexpr uint32_t kNumPhaseTensorMaps = 8u;
            constexpr uint32_t kPhaseTensorMapWord =
                kK3MxFp8WgradOverlap
                ? Overlap::kDW13TensorMaps
                : math::constexpr_align(Prefix::kNumWords, 32u);
            constexpr uint32_t kPhaseTensorMapWords =
                kNumPhaseTensorMaps * kTensorMapWords;
            constexpr uint32_t kTensorMapSmemOffset =
                math::constexpr_align(
                    k3_mxfp8_wgrad_producer_smem_bytes(), 128u);
            constexpr uint32_t kScaleStorageSmemOffset =
                kTensorMapSmemOffset + kTensorMapBytes;
            DG_STATIC_ASSERT(
                kTensorMapBytes == 128u && kTensorMapWords == 32u,
                "K3 device tensor-map layout changed");
            DG_STATIC_ASSERT(
                kPhaseTensorMapWord + kPhaseTensorMapWords <=
                    kNumW2WeightTileStates,
                "K3 MXFP8 maps exceed retired W2 state storage");
            DG_STATIC_ASSERT(
                kScaleStorageSmemOffset +
                    2u * sizeof(K3MxFp8WgradScaleStorage) <=
                    sm100_k3_mxfp8_three_term_wgrad_barrier_offset<
                        DW2Config>(),
                "K3 preparation records overlap grouped-wgrad barriers");
            DG_STATIC_ASSERT(
                kHidden == 3584u && kIntermediateHidden == 3072u &&
                    2u * kIntermediateHidden == 6144u,
                "K3 MXFP8 natural-width descriptor ABI changed");
            auto* const phase_tensor_maps =
                reinterpret_cast<cute::TmaDescriptor*>(
                    weight_tile_states + kPhaseTensorMapWord);
            auto* const tensor_map_staging =
                reinterpret_cast<cute::TmaDescriptor*>(
                    smem_buffer + kTensorMapSmemOffset);
            auto* const phase_scale_storage =
                reinterpret_cast<K3MxFp8WgradScaleStorage*>(
                    smem_buffer + kScaleStorageSmemOffset);
            DG_DEVICE_ASSERT(
                (reinterpret_cast<uint64_t>(phase_tensor_maps) & 127u) ==
                0u);
            // Use only proportional activation storage.  `num_acts_rows` is
            // the host-proven common backing extent and removes the old
            // weight-shaped threshold that made ordinary 16K/32K calls
            // unreachable.  Exact X aliases at most one of saved-down and
            // grad-y-unweighted; the other H-wide BF16 arena is dead after
            // route-grad and safely backs two H-wide FP8 planes.
            const uint32_t k_capacity = num_acts_rows;
            const uint64_t h_bf16_arena_bytes =
                static_cast<uint64_t>(k_capacity) * kHidden *
                sizeof(cd_dtype_t);
            const uint64_t i_bf16_arena_bytes =
                static_cast<uint64_t>(k_capacity) *
                kIntermediateHidden * sizeof(cd_dtype_t);
            const uint64_t scale_arena_bytes =
                static_cast<uint64_t>(k_capacity) * kHidden;
            auto* const safe_h_arena =
                reinterpret_cast<uint8_t*>(
                    x_pool_output == grad_y_unweighted_output
                    ? const_cast<cd_dtype_t*>(down_unweighted_output)
                    : grad_y_unweighted_output);
            auto* const retired_grad_ye_arena =
                reinterpret_cast<uint8_t*>(grad_ye_output);
            auto* const retired_h_arena =
                reinterpret_cast<uint8_t*>(h_weighted_output);
            auto* const scale_arena = const_cast<uint8_t*>(
                reinterpret_cast<const uint8_t*>(acts_ptr));
            DG_DEVICE_ASSERT(
                2u * static_cast<uint64_t>(k_capacity) * kHidden ==
                h_bf16_arena_bytes);
            DG_DEVICE_ASSERT(
                reinterpret_cast<const void*>(safe_h_arena) !=
                    reinterpret_cast<const void*>(x_pool_output));

            const uint64_t h_weighted_bytes =
                static_cast<uint64_t>(k_capacity) *
                kIntermediateHidden * sizeof(cd_dtype_t);
            // Equality-only checks miss partial overlaps.  The outlined
            // verifier proves all phase source/destination byte ranges while
            // keeping that address arithmetic out of the 40-register parent.
            k3_mxfp8_assert_wgrad_parent_aliases(
                grad_ye_output, h_bf16_arena_bytes,
                h_weighted_output, h_weighted_bytes,
                grad_gate_up_output,
                    2u * i_bf16_arena_bytes,
                x_pool_output, h_bf16_arena_bytes,
                safe_h_arena, h_bf16_arena_bytes,
                retired_grad_ye_arena, h_bf16_arena_bytes,
                retired_h_arena, i_bf16_arena_bytes,
                scale_arena, scale_arena_bytes);

            const auto producer_grid_barrier = [&]() {
                if constexpr (kPhaseOrderedWeightDequant) {
                    // This parent variant already carries the phase-ordered
                    // dgrad closure.  Retaining only the two phase pointers
                    // here avoids extending that closure into four outlined
                    // producer calls.  The other variants compile smaller
                    // with their existing barrier closure.
                    K3MxFp8WgradGridBarrier<kNumSMs>{
                        phase_count, phase_sense}();
                } else {
                    full_grid_phase_barrier(kTraceSiteCount, 29);
                }
            };
            const auto* const dw2_maps =
                tensor_map_mxfp8_wgrad_pack.maps +
                kK3MxFp8DW2ValueAPrimaryMap;
            const auto* const dw13_maps =
                tensor_map_mxfp8_wgrad_pack.maps +
                kK3MxFp8DW13ValueAPrimaryMap;
            uint32_t total_k = kK3MxFp8WgradOverlap
                ? weight_tile_states[
                      Prefix::kValuePrefix + kNumExperts]
                : 0u;
            if constexpr (!kK3MxFp8WgradOverlap) {
            const auto dw2_scale_layout =
                k3_mxfp8_wgrad_scale_arena_layout(
                    kHidden, kIntermediateHidden,
                    k_capacity, kPoolBlockRows);
            const auto dw13_scale_layout_for_capacity =
                k3_mxfp8_wgrad_scale_arena_layout(
                    2u * kIntermediateHidden, kHidden,
                    k_capacity, kPoolBlockRows);
            DG_DEVICE_ASSERT(
                k3_mxfp8_wgrad_two_phase_scale_bytes(
                    dw2_scale_layout,
                    dw13_scale_layout_for_capacity) <=
                scale_arena_bytes);
            auto* const dw2_a_primary_values = safe_h_arena;
            auto* const dw2_a_residual_values =
                safe_h_arena +
                static_cast<uint64_t>(k_capacity) * kHidden;
            auto* const dw2_b_primary_values = retired_grad_ye_arena;
            auto* const dw2_b_residual_values =
                retired_grad_ye_arena + kIntermediateHidden;
            auto* const dw2_raw_base = scale_arena;
            auto* const dw2_packed_base =
                dw2_raw_base + dw2_scale_layout.raw_bytes;
            K3MxFp8WgradScaleStorage local_dw2_scale_storage[2];
            K3MxFp8WgradScaleStorage* dw2_scale_storage;
            if constexpr (kPhaseOrderedWeightDequant) {
                // Phase-ordered dgrad has already retired its larger parent
                // frame; two stack-colored records create less spill traffic
                // here than shared-address loads.  P0/P1-small instead place
                // the records in preparation-only shared bytes, removing the
                // address-taken records from their parent frames entirely.
                k3_mxfp8_bind_wgrad_scale_storage(
                    dw2_scale_layout, dw2_raw_base, dw2_packed_base,
                    local_dw2_scale_storage[0],
                    local_dw2_scale_storage[1]);
                dw2_scale_storage = local_dw2_scale_storage;
            } else {
                if (threadIdx.x == 0u) {
                    k3_mxfp8_bind_wgrad_scale_storage(
                        dw2_scale_layout,
                        dw2_raw_base, dw2_packed_base,
                        phase_scale_storage[0], phase_scale_storage[1]);
                }
                __syncthreads();
                dw2_scale_storage = phase_scale_storage;
            }
            const auto& dw2_a_scales = dw2_scale_storage[0];
            const auto& dw2_b_scales = dw2_scale_storage[1];
            trace_begin(17);
            total_k =
                prepare_k3_mxfp8_three_term_wgrad_operands<
                    kNumExperts, BLOCK_M, kNumSMs, kNumThreads>(
                        expert_counts, backward_ranges,
                        grad_ye_output, kHidden, kHidden, kHidden,
                        h_weighted_output, kIntermediateHidden,
                        2u * kHidden, 2u * kHidden,
                        dw2_a_scales, dw2_b_scales,
                        k_capacity,
                        reinterpret_cast<int*>(weight_tile_states),
                        dw2_a_primary_values, dw2_a_residual_values,
                        dw2_b_primary_values, dw2_b_residual_values,
                        dw2_maps[0], dw2_maps[1],
                        dw2_maps[2], dw2_maps[3],
                        smem_buffer, producer_grid_barrier);

            constexpr uint32_t kPublishFirstWarp = 8u;
            constexpr uint32_t kPublishWarps = 2u;
            constexpr uint32_t kPublishThreads =
                kPublishWarps * 32u;
            const auto publish_remote_gradients = [&] (
                    uint32_t background_warp_idx,
                    uint32_t background_lane_idx) {
                if (background_warp_idx >= kPublishFirstWarp &&
                    background_warp_idx <
                        kPublishFirstWarp + kPublishWarps) {
                    constexpr uint32_t kGridSyncIndex = 2u;
                    constexpr uint32_t kBarrierTag = 7u;
                    constexpr uint32_t kNamedBarrier = 15u;
                    const uint32_t thread_idx =
                        (background_warp_idx - kPublishFirstWarp) * 32u +
                        background_lane_idx;
                    comm::nvlink_barrier<
                        kNumRanks, kNumSMs, kPublishThreads,
                        kGridSyncIndex, kBarrierTag>(
                            backward_workspace, backward_sym_buffer,
                            blockIdx.x, thread_idx,
                            [=]() {
                                ptx::sync_aligned(
                                    kPublishThreads, kNamedBarrier);
                            },
                            true, true);
                }
            };

            constexpr uint32_t kDW2ClusterTasks =
                (DW2Config::kShapeM / DW2Config::kBlockM) *
                (DW2Config::kShapeN / DW2Config::kBlockN);
            constexpr uint32_t kDW13ClusterTasks =
                (DW13Config::kShapeM / DW13Config::kBlockM) *
                (DW13Config::kShapeN / DW13Config::kBlockN);
            DG_STATIC_ASSERT(
                kNumSMs == 148u && kNumThreads == 1024u &&
                    DW2Config::kNumMulticast == 2u &&
                    DW13Config::kNumMulticast == 2u,
                "K3 MXFP8 parent launch topology changed");
            DG_STATIC_ASSERT(
                kDW2ClusterTasks == 336u &&
                    kDW13ClusterTasks == 672u &&
                    kDW2ClusterTasks % 2u == 0u &&
                    kDW13ClusterTasks % 2u == 0u,
                "K3 MXFP8 cluster task parity changed");
            DG_STATIC_ASSERT(
                sm100_k3_mxfp8_three_term_wgrad_barrier_offset<
                    DW2Config>() ==
                    sm100_k3_mxfp8_three_term_wgrad_barrier_offset<
                        DW13Config>() &&
                    sm100_k3_mxfp8_three_term_wgrad_smem_bytes<
                        DW2Config>() ==
                    sm100_k3_mxfp8_three_term_wgrad_smem_bytes<
                        DW13Config>(),
                "dW2/dW13 retained TMEM resource signature changed");
            DG_STATIC_ASSERT(
                k3_mxfp8_wgrad_producer_smem_bytes() <=
                    sm100_k3_mxfp8_three_term_wgrad_barrier_offset<
                        DW2Config>() &&
                    sm100_k3_mxfp8_three_term_wgrad_smem_bytes<
                        DW2Config>() <= 232448u,
                "K3 producer/body shared-memory alias proof changed");
            using DW2RetainTmem =
                Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks;
            sm100_k3_mxfp8_three_term_grouped_wgrad_body<
                DW2Config,
                Sm100K3MxFp8ThreeTermDefaultTaskProvider<DW2Config>,
                DW2RetainTmem>(
                    reinterpret_cast<int*>(weight_tile_states), total_k,
                    dw2_maps[0], dw2_maps[1],
                    dw2_maps[2], dw2_maps[3],
                    dw2_maps[4], dw2_maps[5],
                    dw2_maps[6], dw2_maps[7],
                    tensor_map_w2_wgrad_d,
                    smem_buffer, false,
                    publish_remote_gradients);
            trace_end(17);
            }

            // dW2's callback has completed its NVLink acknowledgments and the
            // body has drained, cluster-joined, and invalidated every
            // CTA-specific barrier generation while retaining only empty
            // TMEM. Publish all writers and retire all 148 CTAs before dW13
            // replaces either value arena; the producer's own metadata edge
            // would be too late to establish this initial alias handoff.
            if constexpr (!kK3MxFp8WgradOverlap) {
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                producer_grid_barrier();
            }

            if constexpr (!kK3MxFp8WgradOverlap) {
            // The dead dW2 operand tiles occupy only the lower shared-memory
            // prefix. dW13 preparation reuses that prefix while the empty
            // base-zero TMEM allocation remains live, then initializes fresh
            // phase-zero barriers rather than inheriting dW2 generations.
            const auto dw13_scale_layout =
                k3_mxfp8_wgrad_scale_arena_layout(
                    2u * kIntermediateHidden, kHidden,
                    k_capacity, kPoolBlockRows);
            const auto dw2_scale_layout_for_offset =
                k3_mxfp8_wgrad_scale_arena_layout(
                    kHidden, kIntermediateHidden,
                    k_capacity, kPoolBlockRows);
            const uint64_t dw13_scale_phase_offset =
                k3_mxfp8_wgrad_next_scale_phase_offset(
                    dw2_scale_layout_for_offset);
            DG_DEVICE_ASSERT(
                k3_mxfp8_wgrad_two_phase_scale_bytes(
                    dw2_scale_layout_for_offset,
                    dw13_scale_layout) <=
                scale_arena_bytes);
            auto* const dw13_a_primary_values = retired_grad_ye_arena;
            auto* const dw13_a_residual_values = retired_h_arena;
            auto* const dw13_b_primary_values = safe_h_arena;
            auto* const dw13_b_residual_values =
                safe_h_arena +
                static_cast<uint64_t>(k_capacity) * kHidden;
            auto* const dw13_raw_base =
                scale_arena + dw13_scale_phase_offset;
            auto* const dw13_packed_base =
                dw13_raw_base + dw13_scale_layout.raw_bytes;
            K3MxFp8WgradScaleStorage local_dw13_scale_storage[2];
            K3MxFp8WgradScaleStorage* dw13_scale_storage;
            if constexpr (kPhaseOrderedWeightDequant) {
                k3_mxfp8_bind_wgrad_scale_storage(
                    dw13_scale_layout, dw13_raw_base, dw13_packed_base,
                    local_dw13_scale_storage[0],
                    local_dw13_scale_storage[1]);
                dw13_scale_storage = local_dw13_scale_storage;
            } else {
                if (threadIdx.x == 0u) {
                    k3_mxfp8_bind_wgrad_scale_storage(
                        dw13_scale_layout,
                        dw13_raw_base, dw13_packed_base,
                        phase_scale_storage[0], phase_scale_storage[1]);
                }
                __syncthreads();
                dw13_scale_storage = phase_scale_storage;
            }
            const auto& dw13_a_scales = dw13_scale_storage[0];
            const auto& dw13_b_scales = dw13_scale_storage[1];
            const uint32_t dw13_total_k =
                prepare_k3_mxfp8_three_term_wgrad_operands<
                    kNumExperts, BLOCK_M, kNumSMs, kNumThreads>(
                        expert_counts, backward_ranges,
                        grad_gate_up_output,
                        2u * kIntermediateHidden,
                        2u * kHidden, 2u * kIntermediateHidden,
                        x_pool_output, kHidden, kHidden, kHidden,
                        dw13_a_scales, dw13_b_scales,
                        k_capacity,
                        reinterpret_cast<int*>(weight_tile_states),
                        dw13_a_primary_values, dw13_a_residual_values,
                        dw13_b_primary_values, dw13_b_residual_values,
                        dw13_maps[0], dw13_maps[1],
                        dw13_maps[2], dw13_maps[3],
                        smem_buffer, producer_grid_barrier);
            DG_DEVICE_ASSERT(dw13_total_k == total_k);

            constexpr uint32_t kReduceFirstWarp = 8u;
            constexpr uint32_t kReduceWarps = 4u;
            const auto reduce_remote_grad_x = [&] (
                    uint32_t background_warp_idx,
                    uint32_t background_lane_idx) {
                if (background_warp_idx >= kReduceFirstWarp &&
                    background_warp_idx <
                        kReduceFirstWarp + kReduceWarps) {
                    k3_mxfp8_wgrad_fixed_topk_combine<
                        kNumSMs, kReduceWarps>(
                            backward_grad_x_output,
                            backward_grad_y,
                            &backward_ranges,
                            num_backward_tokens,
                            backward_workspace.num_max_tokens_per_rank,
                            num_topk, kHidden,
                            background_warp_idx - kReduceFirstWarp,
                            background_lane_idx);
                }
            };

            using DW13ReleaseTmem =
                Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks;
            trace_begin(19);
            sm100_k3_mxfp8_three_term_grouped_wgrad_body<
                DW13Config,
                Sm100K3MxFp8ThreeTermDefaultTaskProvider<DW13Config>,
                DW13ReleaseTmem>(
                    reinterpret_cast<int*>(weight_tile_states),
                    dw13_total_k,
                    dw13_maps[0], dw13_maps[1],
                    dw13_maps[2], dw13_maps[3],
                    dw13_maps[4], dw13_maps[5],
                    dw13_maps[6], dw13_maps[7],
                    tensor_map_w13_wgrad_d,
                    smem_buffer, false,
                    reduce_remote_grad_x);
            trace_end(19);
            }
          } else if constexpr (!kK3MxFp8WgradOverlap)
#endif
          {
            // Kernel A's weight-tile epochs are dead after both dgrad phases.
            // Reuse their caller-owned storage for the block-padded K-grouped
            // schedule rather than allocating another CUDA tensor.
            if constexpr (
                !kEarlyW2Wgrad &&
                !kK3BranchMajorBF16DynamicTail)
                initialize_wgrad_grouped_layout();

#if DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL
            if constexpr (kK3BranchMajorBF16DynamicTail) {
                // Operands and communication stores are already published.
                // Four-task claims balance variable expert K without a producer
                // partition, readiness polling, or a new allocation. dW2 and
                // dW13 keep independent terminal queues so a cluster can enter
                // dW13 while peers retire already-claimed dW2 work.
                constexpr uint32_t kDynamicBatchTasks = 4u;
                static_assert(
                    !kEarlyW2Wgrad,
                    "the terminal BF16 lifetime begins with dynamic dW2");
                // Both dynamic bodies use the same BF16 tile shape, stage
                // count, shared-control layout, and base-zero 2-SM TMEM
                // allocation.  Keep that lifetime live across the descriptor
                // switch: dW2 performs the one initialization, while dW13
                // performs the one invalidation/deallocation.  Each body
                // already finishes with a cluster-wide completion join, and
                // the release hook adds the post-teardown join, so no extra
                // caller-side cluster barrier is required.
                using DynamicDW2RetainedResources =
                    Sm100Bf16GemmBatchResourceHooks<true, false>;
                using DynamicDW13ReleaseResources =
                    Sm100Bf16GemmEmbeddedReleaseBatchResourceHooks<>;
                static_assert(
                    DynamicDW2RetainedResources::
                            kInitializeBatchResources &&
                        !DynamicDW2RetainedResources::
                            kReleaseBatchResources,
                    "dynamic dW2 must initialize and retain BF16 resources");
                static_assert(
                    !DynamicDW13ReleaseResources::
                            kInitializeBatchResources &&
                        DynamicDW13ReleaseResources::
                            kReleaseBatchResources &&
                        DynamicDW13ReleaseResources::
                            kSynchronizeAfterRelease,
                    "dynamic dW13 must reuse, release, and join resources");
                if constexpr (kMultiRangeBackward) {
                    using DynamicTwoSegmentDW2Provider = sched::
                        ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider<
                            kWgradBlockM, kWgradBlockN,
                            2, false, kNumSMs,
                            kHidden, kIntermediateHidden,
                            kDynamicBatchTasks,
                            kReadyDW2TasksPerExpert,
                            BLOCK_M, kWgradBlockK,
                            kReadyPoolPrefixWord,
                            kReadyActiveExpertWord>;
                    using DynamicTwoSegmentDW13Provider = sched::
                        ExternalKGroupedTerminalTwoSegmentDynamicRangeProvider<
                            kWgradBlockM, kWgradBlockN,
                            2, false, kNumSMs,
                            2u * kIntermediateHidden, kHidden,
                            kDynamicBatchTasks,
                            kReadyDW13TasksPerExpert,
                            BLOCK_M, kWgradBlockK,
                            kReadyPoolPrefixWord,
                            kReadyActiveExpertWord>;
                    DG_STATIC_ASSERT(
                        DynamicTwoSegmentDW2Provider::
                                kCompleteAcquireMask ==
                                    kReadyCompleteRoleMask &&
                        DynamicTwoSegmentDW13Provider::
                                kCompleteAcquireMask ==
                                    kReadyCompleteRoleMask,
                        "two-segment terminal scheduler role mask changed");
                    DG_STATIC_ASSERT(
                        kReadyDW2TasksPerExpert %
                                kDynamicBatchTasks == 0u &&
                        kReadyDW13TasksPerExpert %
                                kDynamicBatchTasks == 0u,
                        "two-segment terminal batches must not cross experts");
                    DG_DEVICE_ASSERT(backward_ranges.num_ranges == 2u);

                    const uint32_t second_range_index =
                        backward_ranges.reverse_range_index(1u);
                    auto* const union_state =
                        weight_tile_states + kReadyTerminalUnionStateWord;
                    const auto* const second_state =
                        weight_tile_states +
                        second_range_index * kReadyRangeStateStride;
                    const uint32_t dynamic_cluster_idx = blockIdx.x / 2u;

                    if constexpr (
                        !kEarlyW2Wgrad &&
                        !kK3BranchMajorBF16EarlyDW2Overlap) {
                        auto* const dw2_mailbox =
                            union_state + kReadyDW2ClusterSlotWord +
                            dynamic_cluster_idx * kReadyClusterSlotWords;
                        const sched::
                            ExternalKGroupedTerminalTwoSegmentRangeStream
                                dw2_stream{
                                    union_state,
                                    second_state,
                                    union_state + kReadyDW2CursorWord,
                                    union_state[kReadyDW2TasksWord],
                                    dw2_mailbox,
                                    kDynamicBatchTasks,
                                    kReadyDW2TasksPerExpert,
                                };
                        trace_begin(17);
                        run_ready_wgrad_range.template operator()<
                            DynamicTwoSegmentDW2Provider,
                            DynamicDW2RetainedResources,
                            kPublishRemoteGradients>(
                                kHidden, kIntermediateHidden,
                                tensor_map_w2_wgrad_a,
                                tensor_map_w2_wgrad_b,
                                tensor_map_w2_wgrad_d,
                                dw2_stream, false,
                                no_input_tile_retired,
                                no_background_work);
                        trace_end(17);
                    }

                    auto* const dw13_mailbox =
                        union_state + kReadyDW13ClusterSlotWord +
                        dynamic_cluster_idx * kReadyClusterSlotWords;
                    const sched::
                        ExternalKGroupedTerminalTwoSegmentRangeStream
                            dw13_stream{
                                union_state,
                                second_state,
                                union_state + kReadyDW13CursorWord,
                                union_state[kReadyDW13TasksWord],
                                dw13_mailbox,
                                kDynamicBatchTasks,
                                kReadyDW13TasksPerExpert,
                            };
                    trace_begin(19);
                    run_ready_wgrad_range.template operator()<
                        DynamicTwoSegmentDW13Provider,
                        DynamicDW13ReleaseResources,
                        kDirectRemoteGradX,
                        kDirectRemoteGradX ? 64u : 0u>(
                            2u * kIntermediateHidden, kHidden,
                            tensor_map_w13_wgrad_a,
                            tensor_map_w13_wgrad_b,
                            tensor_map_w13_wgrad_d,
                            dw13_stream, kDirectRemoteGradX,
                            no_input_tile_retired,
                            no_background_work);
                    trace_end(19);
                } else {
                using DynamicDW2Provider =
                    sched::ExternalKGroupedTerminalDynamicRangeProvider<
                        kWgradBlockM, kWgradBlockN,
                        2, false, kNumSMs,
                        kHidden, kIntermediateHidden,
                        kDynamicBatchTasks, kPairedDW2TasksPerExpert,
                        BLOCK_M, kWgradBlockK,
                        kReadyPoolPrefixWord,
                        kReadyActiveExpertWord,
                        4u, 4u, 16u, true>;
                using DynamicDW13Provider =
                    sched::ExternalKGroupedTerminalDynamicRangeProvider<
                        kWgradBlockM, kWgradBlockN,
                        2, false, kNumSMs,
                        2u * kIntermediateHidden, kHidden,
                        kDynamicBatchTasks, kPairedDW13TasksPerExpert,
                        BLOCK_M, kWgradBlockK,
                        kReadyPoolPrefixWord,
                        kReadyActiveExpertWord,
                        4u, 4u, 16u, true>;
                DG_STATIC_ASSERT(
                    DynamicDW2Provider::kCompleteAcquireMask ==
                            kReadyCompleteRoleMask &&
                    DynamicDW13Provider::kCompleteAcquireMask ==
                            kReadyCompleteRoleMask,
                    "dynamic BF16 scheduler role mask changed");
                DG_STATIC_ASSERT(
                    kPairedDW2TasksPerExpert % kDynamicBatchTasks == 0u &&
                    kPairedDW13TasksPerExpert % kDynamicBatchTasks == 0u,
                    "dynamic BF16 batches must not cross experts");

                const uint32_t dynamic_cluster_idx = blockIdx.x / 2u;
                if constexpr (!kEarlyW2Wgrad) {
                    auto* const dw2_mailbox =
                        weight_tile_states +
                        kReadyDW2ClusterSlotWord +
                        dynamic_cluster_idx * kReadyClusterSlotWords;
                    const sched::ExternalKGroupedRangeStream dw2_stream{
                        weight_tile_states,
                        0u, 0u,
                        weight_tile_states + kReadyDW2CursorWord,
                        weight_tile_states[kReadyDW2TasksWord],
                        dw2_mailbox,
                        kDynamicBatchTasks,
                        kPairedDW2TasksPerExpert,
                    };
                    trace_begin(17);
                    run_ready_wgrad_range.template operator()<
                        DynamicDW2Provider,
                        DynamicDW2RetainedResources,
                        kPublishRemoteGradients>(
                            kHidden, kIntermediateHidden,
                            tensor_map_w2_wgrad_a,
                            tensor_map_w2_wgrad_b,
                            tensor_map_w2_wgrad_d,
                            dw2_stream, false,
                            no_input_tile_retired,
                            no_background_work);
                    trace_end(17);
                }

                auto* const dw13_mailbox =
                    weight_tile_states +
                    kReadyDW13ClusterSlotWord +
                    dynamic_cluster_idx * kReadyClusterSlotWords;
                const sched::ExternalKGroupedRangeStream dw13_stream{
                    weight_tile_states,
                    0u, 0u,
                    weight_tile_states + kReadyDW13CursorWord,
                    weight_tile_states[kReadyDW13TasksWord],
                    dw13_mailbox,
                    kDynamicBatchTasks,
                    kPairedDW13TasksPerExpert,
                };
                trace_begin(19);
                run_ready_wgrad_range.template operator()<
                    DynamicDW13Provider,
                    DynamicDW13ReleaseResources,
                    kDirectRemoteGradX,
                    kDirectRemoteGradX ? 64u : 0u>(
                        2u * kIntermediateHidden, kHidden,
                        tensor_map_w13_wgrad_a,
                        tensor_map_w13_wgrad_b,
                        tensor_map_w13_wgrad_d,
                        dw13_stream, kDirectRemoteGradX,
                        no_input_tile_retired,
                        no_background_work);
                trace_end(19);
                }
            } else
#endif
            {
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
                    false,
                    no_input_tile_retired,
                    no_background_work);
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
                    kDirectRemoteGradX,
                    no_input_tile_retired,
                    no_background_work);
            comm::cluster_sync_with_relaxed_arrive();
            trace_end(19);
            }
          }
        }
#endif

        trace_end(20);
        trace_end(0);
    }
#endif
}

}  // namespace deep_gemm
