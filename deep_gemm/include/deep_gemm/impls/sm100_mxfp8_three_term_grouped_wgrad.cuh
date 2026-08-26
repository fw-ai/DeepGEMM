#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/impls/k3_multirange_backward.hpp>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

#ifndef DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD
// Keep template declarations available even when a runtime JIT includes this
// header before the parent selects the experimental specialization. The
// parent selection flag still controls instantiation and launch behavior.
#define DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD 1
#endif

#ifndef DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
// The fused parent may opt one JIT specialization into bounded ring tracing.
// Ordinary builds retain no printf, clock, or scheduler-state instructions.
#define DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG 0
#endif

namespace deep_gemm {

/** Product order for the minimum K3 MXFP8 weight-gradient approximation.
 *
 * Given independently group-32 quantized primary and BF16-rounded residual
 * token-major operands A0/A1 and B0/B1, the body computes
 *
 *     A0^T * B0 + A1^T * B0 + A0^T * B1.
 *
 * TMA descriptors expose their transposed [MN, K] views, so the physical
 * UMMA operation is descriptor-A * descriptor-B^T for each term.
 *
 * P11 is intentionally absent.  The order is part of the numeric contract:
 * every product accumulates directly into the same FP32 TMEM tile and the
 * completed tile is converted to BF16 only once by the epilogue.
 */
enum class K3MxFp8WgradProduct : uint32_t {
    P00 = 0,
    P10 = 1,
    P01 = 2,
};

struct K3MxFp8WgradProductOperands {
    bool residual_a;
    bool residual_b;
};

constexpr K3MxFp8WgradProductOperands
k3_mxfp8_wgrad_product_operands(K3MxFp8WgradProduct product) {
    switch (product) {
        case K3MxFp8WgradProduct::P00:
            return {false, false};
        case K3MxFp8WgradProduct::P10:
            return {true, false};
        case K3MxFp8WgradProduct::P01:
            return {false, true};
    }
    return {false, false};
}

constexpr uint32_t k3_mxfp8_wgrad_scale_rows(uint32_t k_rows) {
    return (k_rows + 127u) / 128u;
}

/** Map one logical feature to UTCCP's native packed-scale order.
 *
 * SM100 UTCCP consumes each 128-word scale tile as four interleaved 32-word
 * rows: logical features [0, 32, 64, 96] are adjacent, followed by
 * [1, 33, 65, 97], and so on.  Publishing packed UE8M0 words in that order
 * lets the consumer load the tile directly and removes the software
 * [4][32] -> [32][4] shared-memory transpose.  The mapping is a bijection
 * within every aligned 128-feature tile and preserves the tile base.
 */
CUTLASS_HOST_DEVICE constexpr uint32_t
k3_mxfp8_utccp_scale_feature(uint32_t logical_feature) {
    const uint32_t tile_base = logical_feature & ~127u;
    const uint32_t feature_in_tile = logical_feature & 127u;
    return tile_base + (feature_in_tile % 32u) * 4u +
        feature_in_tile / 32u;
}

/** Map one logical product stage to its compact primary/residual slot.
 *
 * Logical stages are P00/P10/P01 triplets. P10 reuses B0 from P00 and P01
 * reuses A0 from P00, so each triplet needs only two physical slots per
 * operand. Keeping this mapping explicit lets the six-entry barrier ring use
 * four operand slots without changing its latency-hiding depth.
 */
CUTLASS_HOST_DEVICE constexpr uint32_t
k3_mxfp8_wgrad_operand_stage(
        uint32_t logical_stage, bool use_residual) {
    return 2u * (logical_stage / 3u) +
        static_cast<uint32_t>(use_residual);
}

/** Caller-owned scale aliases for one natural-width wgrad operand.
 *
 * Primary/residual value planes are named by the supplied TMA descriptors and
 * can therefore live in distinct phase-specific BF16 arenas.  Raw one-byte
 * group scales and compact four-byte scale rows may independently occupy dead
 * tails of other activation pools.  The producer never assumes contiguity
 * between these fields, allocates nothing, and retains no process-global
 * state.
 */
struct K3MxFp8WgradScaleStorage {
    uint8_t* primary_raw = nullptr;
    uint8_t* residual_raw = nullptr;
    uint32_t* primary_packed = nullptr;
    uint32_t* residual_packed = nullptr;
    uint32_t raw_row_capacity = 0;
    uint32_t packed_row_capacity = 0;
};

/** Minimal full-grid phase barrier used by the fused K3 wgrad suffix.
 *
 * The parent barrier's trace site is intentionally out of range for every
 * suffix preparation edge, so capturing that large parent lambda only keeps
 * unrelated trace state live.  This value object carries the two persistent
 * phase words and preserves the same release/wait protocol and watchdog.
 */
template <uint32_t kNumSMs>
struct K3MxFp8WgradGridBarrier {
    uint32_t* phase_count;
    uint32_t* phase_sense;

    CUTLASS_DEVICE void operator()() const {
        // Publish every lane's distributed global stores, then join the CTA
        // before lane zero advertises its grid-barrier ticket.  A lane-zero
        // fence cannot by itself flush another lane's writes, and the trailing
        // join alone is too late: a peer CTA could otherwise observe the phase
        // transition while lanes 1..767 still have launch state in flight.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint32_t old_sense = atomicAdd(phase_sense, 0u);
            __threadfence();
            const uint32_t ticket = atomicAdd(phase_count, 1u);
            if (ticket == kNumSMs - 1u) {
                atomicExch(phase_count, 0u);
                __threadfence();
                atomicAdd(phase_sense, 1u);
            } else {
                const uint64_t wait_start = clock64();
                while (ptx::ld_acq(phase_sense) == old_sense) {
                    if (clock64() - wait_start > 4000000000ull)
                        asm volatile("trap;");
                }
            }
        }
        __syncthreads();
    }
};

/** Reusable phase barrier for a compile-time subset of persistent CTAs.
 *
 * The enclosing parent initializes both words before its last uniform-grid
 * rendezvous.  After clusters split, only the selected producer CTAs call this
 * object, allowing A-source retirement, B aliasing, scale compaction, and
 * descriptor publication to remain kernel-local without waiting for W13 dgrad
 * CTAs.  Every participating CTA must call each phase exactly once.
 */
template <uint32_t kNumProducerCTAs>
struct K3MxFp8WgradSubsetBarrier {
    uint32_t* phase_count;
    uint32_t* phase_sense;

    static_assert(kNumProducerCTAs > 0u,
                  "An MXFP8 producer subset must contain at least one CTA");

    CUTLASS_DEVICE void operator()() const {
        // Participating lanes publish their own producer stores before this
        // call.  Join them before lane zero advertises the subset ticket.
        __syncthreads();
        if (threadIdx.x == 0u) {
            const uint32_t old_sense = ptx::ld_acq(phase_sense);
            __threadfence();
            const uint32_t ticket = atomicAdd(phase_count, 1u);
            if (ticket == kNumProducerCTAs - 1u) {
                atomicExch(phase_count, 0u);
                __threadfence();
                atomicAdd(phase_sense, 1u);
            } else {
                const uint64_t wait_start = clock64();
                while (ptx::ld_acq(phase_sense) == old_sense) {
                    if (clock64() - wait_start > 4000000000ull)
                        asm volatile("trap;");
                }
            }
        }
        __syncthreads();
    }
};

/** Bind one two-operand scale layout to caller-proven arena bases. */
CUTLASS_HOST_DEVICE void
k3_mxfp8_bind_wgrad_scale_storage(
        const K3MxFp8WgradScaleArenaLayout& layout,
        uint8_t* raw_base, uint8_t* packed_base,
        K3MxFp8WgradScaleStorage& a,
        K3MxFp8WgradScaleStorage& b) {
    a = {
        raw_base + layout.raw_a_primary,
        raw_base + layout.raw_a_residual,
        reinterpret_cast<uint32_t*>(
            packed_base + layout.packed_a_primary),
        reinterpret_cast<uint32_t*>(
            packed_base + layout.packed_a_residual),
        layout.raw_row_capacity,
        layout.packed_row_capacity,
    };
    b = {
        raw_base + layout.raw_b_primary,
        raw_base + layout.raw_b_residual,
        reinterpret_cast<uint32_t*>(
            packed_base + layout.packed_b_primary),
        reinterpret_cast<uint32_t*>(
            packed_base + layout.packed_b_residual),
        layout.raw_row_capacity,
        layout.packed_row_capacity,
    };
}

/** Publish one phase's eight address-specialized TensorMaps without an ABI.
 *
 * The parent supplies four immutable base layouts through descriptor slots
 * that are compile-time dead in the selected K3 configuration.  CTA 0 clones
 * each layout in shared memory, replaces only its global address, and uses
 * CUTLASS's fused shared-to-global TensorMap copy/release instruction.  That
 * instruction is ``sync.aligned`` and therefore must be executed by an entire
 * converged warp, even though only its elected lane edits the shared staging
 * descriptor. A caller-provided full-grid barrier retires publication before
 * lane 0 of each CTA acquires the maps used by the producer and grouped GEMM.
 */
template <uint32_t kNumMaps, typename GridBarrier>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_wgrad_tensor_maps_subset(
        cute::TmaDescriptor* global_maps,
        cute::TmaDescriptor* shared_staging,
        const cute::TmaDescriptor& value_a_base,
        const cute::TmaDescriptor& value_b_base,
        const cute::TmaDescriptor& scale_a_base,
        const cute::TmaDescriptor& scale_b_base,
        const void* a_primary_values,
        const void* a_residual_values,
        const void* b_primary_values,
        const void* b_residual_values,
        const void* a_primary_scales,
        const void* a_residual_scales,
        const void* b_primary_scales,
        const void* b_residual_scales,
        uint32_t producer_cta_idx,
        GridBarrier grid_barrier) {
    static_assert(kNumMaps == 8u, "K3 wgrad publishes exactly eight maps");
    if (producer_cta_idx == 0u && threadIdx.x < 32u) {
        const auto publish_one = [&] (
                uint32_t index,
                const cute::TmaDescriptor& base,
                const void* address) {
            if (cute::elect_one_sync()) {
                *shared_staging = base;
                ptx::tensor_map_replace_global_addr_in_smem(
                    shared_staging, address);
                cute::tma_desc_commit_group();
                cute::tma_desc_wait_group();
            }
            __syncwarp();
            cute::tma_descriptor_cp_fence_release(
                global_maps + index, *shared_staging);
            // Do not let the elected lane overwrite the single staging slot
            // until every lane has retired the collective copy/fence.
            __syncwarp();
        };
        publish_one(0u, value_a_base, a_primary_values);
        publish_one(1u, value_a_base, a_residual_values);
        publish_one(2u, value_b_base, b_primary_values);
        publish_one(3u, value_b_base, b_residual_values);
        publish_one(4u, scale_a_base, a_primary_scales);
        publish_one(5u, scale_a_base, a_residual_scales);
        publish_one(6u, scale_b_base, b_primary_scales);
        publish_one(7u, scale_b_base, b_residual_scales);
        if (cute::elect_one_sync())
            __threadfence();
    }
    __syncthreads();
    grid_barrier();
    if (threadIdx.x == 0u) {
        #pragma unroll
        for (uint32_t index = 0u; index < kNumMaps; ++index)
            ptx::tensor_map_acquire_gpu(global_maps + index);
    }
    __syncthreads();
}

/** Full-grid compatibility wrapper for the serial producer path. */
template <uint32_t kNumMaps, typename GridBarrier>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_publish_wgrad_tensor_maps(
        cute::TmaDescriptor* global_maps,
        cute::TmaDescriptor* shared_staging,
        const cute::TmaDescriptor& value_a_base,
        const cute::TmaDescriptor& value_b_base,
        const cute::TmaDescriptor& scale_a_base,
        const cute::TmaDescriptor& scale_b_base,
        const void* a_primary_values,
        const void* a_residual_values,
        const void* b_primary_values,
        const void* b_residual_values,
        const void* a_primary_scales,
        const void* a_residual_scales,
        const void* b_primary_scales,
        const void* b_residual_scales,
        GridBarrier grid_barrier) {
    k3_mxfp8_publish_wgrad_tensor_maps_subset<kNumMaps>(
        global_maps, shared_staging,
        value_a_base, value_b_base, scale_a_base, scale_b_base,
        a_primary_values, a_residual_values,
        b_primary_values, b_residual_values,
        a_primary_scales, a_residual_scales,
        b_primary_scales, b_residual_scales,
        static_cast<uint32_t>(blockIdx.x), grid_barrier);
}

/** Trap if a source slated for quantization overlaps any phase destination. */
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_assert_wgrad_parent_aliases(
        const void* dw2_a_source, uint64_t dw2_a_bytes,
        const void* dw2_b_source, uint64_t dw2_b_bytes,
        const void* dw13_a_source, uint64_t dw13_a_bytes,
        const void* dw13_b_source, uint64_t dw13_b_bytes,
        const void* safe_h_arena, uint64_t safe_h_arena_bytes,
        const void* retired_grad_ye, uint64_t retired_grad_ye_bytes,
        const void* retired_h, uint64_t retired_h_bytes,
        const void* scale_arena, uint64_t scale_arena_bytes) {
    const auto disjoint = [](
            const void* lhs, uint64_t lhs_bytes,
            const void* rhs, uint64_t rhs_bytes) {
        const uint64_t lhs_begin = reinterpret_cast<uint64_t>(lhs);
        const uint64_t rhs_begin = reinterpret_cast<uint64_t>(rhs);
        return lhs_begin + lhs_bytes <= rhs_begin ||
            rhs_begin + rhs_bytes <= lhs_begin;
    };
    // dW2-A writes safe_h and scales while reading grad_ye.  dW2-B starts
    // only after the explicit full-grid A-retirement edge, so grad_ye may
    // then become its destination, but h_weighted must remain untouched.
    DG_DEVICE_ASSERT(
        disjoint(dw2_a_source, dw2_a_bytes,
                 safe_h_arena, safe_h_arena_bytes) &&
        disjoint(dw2_a_source, dw2_a_bytes,
                 scale_arena, scale_arena_bytes) &&
        disjoint(dw2_b_source, dw2_b_bytes,
                 safe_h_arena, safe_h_arena_bytes) &&
        disjoint(dw2_b_source, dw2_b_bytes,
                 retired_grad_ye, retired_grad_ye_bytes) &&
        disjoint(dw2_b_source, dw2_b_bytes,
                 scale_arena, scale_arena_bytes));

    // After dW2, grad_ye and h_weighted become dW13-A's two independent
    // logical-2I FP8 planes. The grad_ye primary retains its physical 2H row
    // pitch; h_weighted remains compact 2I. safe_h becomes dW13-B's two H
    // planes. The sources and all simultaneous destinations must be pairwise
    // disjoint.
    DG_DEVICE_ASSERT(
        disjoint(dw13_a_source, dw13_a_bytes,
                 safe_h_arena, safe_h_arena_bytes) &&
        disjoint(dw13_a_source, dw13_a_bytes,
                 retired_grad_ye, retired_grad_ye_bytes) &&
        disjoint(dw13_a_source, dw13_a_bytes,
                 retired_h, retired_h_bytes) &&
        disjoint(dw13_b_source, dw13_b_bytes,
                 safe_h_arena, safe_h_arena_bytes) &&
        disjoint(dw13_b_source, dw13_b_bytes,
                 retired_grad_ye, retired_grad_ye_bytes) &&
        disjoint(dw13_b_source, dw13_b_bytes,
                 retired_h, retired_h_bytes) &&
        disjoint(dw13_a_source, dw13_a_bytes,
                 scale_arena, scale_arena_bytes) &&
        disjoint(dw13_b_source, dw13_b_bytes,
                 scale_arena, scale_arena_bytes));
    DG_DEVICE_ASSERT(
        disjoint(safe_h_arena, safe_h_arena_bytes,
                 retired_grad_ye, retired_grad_ye_bytes) &&
        disjoint(safe_h_arena, safe_h_arena_bytes,
                 retired_h, retired_h_bytes) &&
        disjoint(retired_grad_ye, retired_grad_ye_bytes,
                 retired_h, retired_h_bytes) &&
        disjoint(scale_arena, scale_arena_bytes,
                 safe_h_arena, safe_h_arena_bytes) &&
        disjoint(scale_arena, scale_arena_bytes,
                 retired_grad_ye, retired_grad_ye_bytes) &&
        disjoint(scale_arena, scale_arena_bytes,
                 retired_h, retired_h_bytes));
}

/** Reduce fixed-top-k symmetric dX planes on wgrad's idle suffix warps.
 *
 * The enclosing dW2 callback first performs the rank publication barrier.
 * This helper is then called by dW13 after that edge and preserves the native
 * slot-order FP32 accumulation followed by one BF16 output rounding.  It is
 * outlined so the UMMA/TMA body does not inherit the reduction's vector
 * accumulators or address arithmetic.
 */
template <uint32_t kNumSMs, uint32_t kNumCombineWarps>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_wgrad_fixed_topk_combine(
        cutlass::bfloat16_t* grad_x_output,
        const cutlass::bfloat16_t* combine_buffer,
        const K3BackwardRangeSet* backward_ranges,
        uint32_t num_tokens,
        uint32_t num_max_tokens,
        uint32_t num_topk,
        uint32_t hidden,
        uint32_t combine_warp_idx,
        uint32_t lane_idx) {
    constexpr uint32_t kValuesPerVector =
        2u * sizeof(uint4) / sizeof(cutlass::bfloat16_t);
    const uint32_t num_vectors_per_token = hidden / kValuesPerVector;
    DG_DEVICE_ASSERT(hidden % kValuesPerVector == 0u);

    for (uint32_t token_idx =
             blockIdx.x * kNumCombineWarps + combine_warp_idx;
         token_idx < num_tokens;
         token_idx += kNumSMs * kNumCombineWarps) {
        const uint32_t physical_token_idx =
            k3_multirange_physical_token_index(
                *backward_ranges, token_idx);
        DG_DEVICE_ASSERT(
            physical_token_idx != static_cast<uint32_t>(-1));
        DG_DEVICE_ASSERT(physical_token_idx < num_max_tokens);

        for (uint32_t vector_idx = lane_idx;
             vector_idx < num_vectors_per_token;
             vector_idx += 32u) {
            float values[kValuesPerVector] = {0.0f};
            #pragma unroll 1
            for (uint32_t topk_idx = 0;
                 topk_idx < num_topk; ++topk_idx) {
                const uint64_t packed_idx =
                    ((static_cast<uint64_t>(topk_idx) *
                          num_max_tokens +
                      physical_token_idx) *
                         num_vectors_per_token +
                     vector_idx) *
                    2u;
                uint4 packed[2];
                packed[0] = reinterpret_cast<const uint4*>(
                    combine_buffer)[packed_idx];
                packed[1] = reinterpret_cast<const uint4*>(
                    combine_buffer)[packed_idx + 1u];
                const auto* packed_values =
                    reinterpret_cast<const cutlass::bfloat16_t*>(packed);
                #pragma unroll
                for (uint32_t i = 0; i < kValuesPerVector; ++i)
                    values[i] += static_cast<float>(packed_values[i]);
            }

            uint4 packed_output[2];
            auto* output_values =
                reinterpret_cast<cutlass::bfloat16_t*>(packed_output);
            #pragma unroll
            for (uint32_t i = 0; i < kValuesPerVector; ++i)
                output_values[i] = cutlass::bfloat16_t(values[i]);
            auto* output = reinterpret_cast<uint4*>(grad_x_output) +
                (static_cast<uint64_t>(token_idx) *
                     num_vectors_per_token +
                 vector_idx) *
                    2u;
            output[0] = packed_output[0];
            output[1] = packed_output[1];
        }
    }
}

/** Prefix words retained in K3's retired weight-tile state allocation.
 *
 * The first `kNumExperts` words deliberately are the grouped-layout ABI used
 * by the GEMM scheduler.  The remaining arrays translate the range-major BF16
 * pools into one expert-major, reverse-range-concatenated K axis and its
 * independently compact `ceil(expert_K / 128)` scale axis.
 */
template <uint32_t kNumExperts, uint32_t kMaxRanges>
struct K3MxFp8WgradPrefixLayout {
    static constexpr uint32_t kGroupedLayout = 0;
    static constexpr uint32_t kValuePrefix =
        kGroupedLayout + kNumExperts;
    static constexpr uint32_t kScalePrefix =
        kValuePrefix + kNumExperts + 1;
    static constexpr uint32_t kPhysicalRangePrefix =
        kScalePrefix + kNumExperts + 1;
    static constexpr uint32_t kNumWords =
        kPhysicalRangePrefix + kMaxRanges * (kNumExperts + 1);
};

/** Allocation-free control layout for pipelined exact K3 wgrad.
 *
 * Every word aliases the retired W2 tile-state allocation.  Descriptor arrays
 * are immutable for the lifetime of an overlap generation, while scheduler
 * cursors, readiness counters, and cluster mailboxes occupy disjoint cache
 * lines.  Keeping the layout in one compile-time contract prevents the exact
 * value/scale prefixes from accidentally overlapping the older BF16 ready-
 * wgrad state.
 */
template <uint32_t kNumExperts, uint32_t kNumSMs,
          uint32_t kMaxRanges>
struct K3MxFp8WgradOverlapStateLayout {
    using Prefix = K3MxFp8WgradPrefixLayout<kNumExperts, kMaxRanges>;

    static constexpr uint32_t kTensorMapWords = 32u;
    static constexpr uint32_t kNumTensorMapsPerPhase = 8u;
    static constexpr uint32_t kTensorMapPhaseWords =
        kTensorMapWords * kNumTensorMapsPerPhase;
    static constexpr uint32_t kNumClusters = kNumSMs / 2u;
    static constexpr uint32_t kMailboxWordsPerCluster = 4u;
    static constexpr uint32_t kMailboxPhaseWords =
        kNumClusters * kMailboxWordsPerCluster;

    static constexpr uint32_t kDW2TensorMaps =
        math::constexpr_align(Prefix::kNumWords, 32u);
    static constexpr uint32_t kDW13TensorMaps =
        kDW2TensorMaps + kTensorMapPhaseWords;
    static constexpr uint32_t kPoolBlockPrefix =
        kDW13TensorMaps + kTensorMapPhaseWords;
    static constexpr uint32_t kActiveExperts =
        kPoolBlockPrefix + kNumExperts + 1u;
    static constexpr uint32_t kControl =
        kActiveExperts + kNumExperts;

    static constexpr uint32_t kMagic = kControl;
    static constexpr uint32_t kEpoch = kControl + 1u;
    static constexpr uint32_t kActiveCount = kControl + 2u;
    static constexpr uint32_t kTotalPoolBlocks = kControl + 3u;
    static constexpr uint32_t kDW2Tasks = kControl + 4u;
    static constexpr uint32_t kDW13Tasks = kControl + 5u;
    static constexpr uint32_t kDW2Cursor = kControl + 6u;
    static constexpr uint32_t kDW13Cursor = kControl + 7u;
    static constexpr uint32_t kDW2SubsetBarrierCount = kControl + 8u;
    static constexpr uint32_t kDW2SubsetBarrierSense = kControl + 9u;
    static constexpr uint32_t kDW13SubsetBarrierCount = kControl + 10u;
    static constexpr uint32_t kDW13SubsetBarrierSense = kControl + 11u;
    // Exact elastic dW2 production reuses the otherwise-idle dW13 subset
    // barrier count as its feature-panel ticket. The legacy subset producer
    // and the elastic producer are mutually exclusive specializations, so
    // this alias adds no state and, unlike kDW2Cursor, cannot race the
    // ready-first dW2 scheduler.
    static constexpr uint32_t kElasticDW2ProducerCursor =
        kDW13SubsetBarrierCount;
    // dW2 providers keep acquiring kEpoch until their terminal mailbox batch.
    // A disjoint generation word prevents early dW13 publication from making
    // those late providers wait forever on a replaced equality value.
    static constexpr uint32_t kDW13Epoch = kControl + 12u;
    // Number of active experts whose full dW13 operand task set has published.
    // Shepherd clusters use this scalar to terminate without rescanning every
    // expert or introducing a grid barrier.
    static constexpr uint32_t kDW13QuantExpertsDone = kControl + 13u;
    // Option-A replaces the legacy feature-major dW13 quantizer, so its
    // retired scalar becomes one block-major A/B producer cursor.  The alias
    // is deliberately distinct from both grouped scheduler cursors; late
    // consumer CTAs and background producer engines may therefore claim work
    // independently without racing the same atomic word.
    static constexpr uint32_t kDW13ABBlockCursor =
        kDW13QuantExpertsDone;
    // The first post-W13 producer-pair controller release-opens this handoff.
    // Every live borrowed engine acquire-checks it only between complete A
    // panels, then contributes one nonblocking arrival and returns to W13.
    static constexpr uint32_t kElasticDW2HandoffRequest = kControl + 14u;
    // The two-range suffix producer terminates from one scalar instead of
    // repeatedly scanning every per-expert panel cursor.  It is disjoint from
    // the legacy elastic handoff words so both implementations remain
    // independently selectable while the exact path is validated.
    static constexpr uint32_t kDW2SuffixQuantExpertsDone = kControl + 15u;
    static constexpr uint32_t kControlEnd = kControl + 16u;

    static constexpr uint32_t kDW2OperandReady =
        math::constexpr_align(kControlEnd, 32u);
    static constexpr uint32_t kDW2InputRetired =
        kDW2OperandReady + kNumExperts;
    static constexpr uint32_t kDW13CompositeReady =
        kDW2InputRetired + kNumExperts;
    // Streaming quantization claims expert-local 32x128 TMA tasks independently
    // from their completion.  The last completion contributes exactly one
    // operand-ready credit to the composite dW13 counter; ready-first then owns
    // that word permanently as its expert-local task cursor.
    static constexpr uint32_t kDW13QuantCursor =
        kDW13CompositeReady + kNumExperts;
    static constexpr uint32_t kDW13QuantDone =
        kDW13QuantCursor + kNumExperts;
    // The two-range suffix claims complete 128-feature dW2 panels.  A and B
    // share one monotonically increasing per-expert cursor: A occupies
    // [0, 28), B occupies [28, 52).  B may overwrite an aliased range-major
    // BF16 source only after every intersecting A expert release-publishes its
    // completion bit.
    static constexpr uint32_t kDW2SuffixQuantCursor =
        kDW13QuantDone + kNumExperts;
    static constexpr uint32_t kDW2SuffixQuantDone =
        kDW2SuffixQuantCursor + kNumExperts;
    static constexpr uint32_t kDW2SuffixAQuantDone =
        kDW2SuffixQuantDone + kNumExperts;
    static constexpr uint32_t kDW2SuffixExpertMaskWords =
        (kNumExperts + 31u) / 32u;
    static constexpr uint32_t kDW2SuffixACompleteMasks =
        kDW2SuffixAQuantDone + kNumExperts;
    static constexpr uint32_t kDW2SuffixBAliasDependencies =
        kDW2SuffixACompleteMasks + kDW2SuffixExpertMaskWords;
    // dW2 has 28 A and 24 B feature panels.  A unique producer CTA owns all
    // K rows for one panel and release-ORs its bit only after value and scale
    // stores retire.  Two acquire-loaded words are therefore sufficient to
    // gate any exact cluster output task without a per-panel counter array.
    static constexpr uint32_t kDW2FeaturePanelsPerExpert = 52u;
    static constexpr uint32_t kDW2FeatureReadyWordsPerExpert = 2u;
    static constexpr uint32_t kDW2FeatureReadyMasks =
        kDW2SuffixBAliasDependencies +
        kNumExperts * kDW2SuffixExpertMaskWords;
    // A dW2 output cluster task retires one A-panel pair and one B-panel pair.
    // Fourteen A pairs each have twelve consumers; twelve B pairs each have
    // fourteen.  dW13 quantization may overwrite only the corresponding
    // retired pair, so preserve these 26 exact counters per expert.
    static constexpr uint32_t kDW2InputPairCountersPerExpert = 26u;
    static constexpr uint32_t kDW2InputPairRetired =
        kDW2FeatureReadyMasks +
        kNumExperts * kDW2FeatureReadyWordsPerExpert;
    // dW13 has 48 A and 28 B panels. A unique feature-major producer writes
    // all packed scale rows for one panel and release-publishes the exact row
    // count here; the unified scheduler acquire-checks four counters per task.
    static constexpr uint32_t kDW13FeaturePanelsPerExpert = 76u;
    static constexpr uint32_t kDW13FeatureDone =
        kDW2InputPairRetired +
        kNumExperts * kDW2InputPairCountersPerExpert;
    static constexpr uint32_t kDW2Mailboxes =
        kDW13FeatureDone +
        kNumExperts * kDW13FeaturePanelsPerExpert;
    static constexpr uint32_t kDW13Mailboxes =
        kDW2Mailboxes + kMailboxPhaseWords;
    static constexpr uint32_t kNumWords =
        kDW13Mailboxes + kMailboxPhaseWords;

    static constexpr uint32_t kDW2ClusterTasksPerExpert = 168u;
    static constexpr uint32_t kDW13ClusterTasksPerExpert = 336u;
    // TMA roles 0/1, the leader MMA role, and eight epilogue roles.  Scale
    // publication is UTCCP-native, so there is no auxiliary transpose role.
    static constexpr uint32_t kExactSchedulerRoleMask = 0x7ffu;

    static_assert(kNumSMs % 2u == 0u,
                  "Exact wgrad overlap requires complete CTA pairs");
    static_assert(
        kNumExperts <= 128u,
        "dW2 suffix alias masks reserve at most four 32-bit words");
    static_assert(kDW2TensorMaps % 32u == 0u &&
                      kDW13TensorMaps % 32u == 0u,
                  "Device TensorMaps must remain 128-byte aligned");
    static_assert(
        kDW13ABBlockCursor != kDW13Cursor &&
        kDW13ABBlockCursor != kDW13QuantCursor,
        "Option-A producer and grouped consumer require disjoint cursors");
};

// One 128-thread engine uses one 17536-byte scratch slice. Four independent
// engines fit after the compact exact body while keeping the combined launch
// below the SM103 per-CTA shared-memory limit. No CTA, global allocation, or
// occupancy slot is reserved for quantization.
constexpr uint32_t kK3MxFp8DW13QuantWarpsPerEngine = 4u;
constexpr uint32_t kK3MxFp8DW13QuantNumEngines = 4u;
// During W13 dgrad the exact K3 specialization lends one of its seven
// epilogue warpgroups to dW2 operand production. The remaining six
// warpgroups still provide 24 epilogue warps. The parent passes a disjoint
// 32x64 scratch slice from its live-state tail; this header deliberately does
// not assume that the later dW13 suffix scratch is retired yet.
constexpr uint32_t kK3MxFp8DW2W13QuantFirstWarp = 4u;
constexpr uint32_t kK3MxFp8DW2W13QuantWarps = 4u;
constexpr uint32_t kK3MxFp8DW2W13QuantThreads =
    kK3MxFp8DW2W13QuantWarps * 32u;
constexpr uint32_t kK3MxFp8DW2W13QuantFeatureSubtile = 64u;
constexpr uint32_t kK3MxFp8DW2W13QuantSubtilesPerPanel = 2u;
constexpr uint32_t kK3MxFp8DW2W13QuantScaleFeatureTile =
    kK3MxFp8DW2W13QuantFeatureSubtile *
    kK3MxFp8DW2W13QuantSubtilesPerPanel;
constexpr uint32_t kK3MxFp8DW2W13QuantBarrierOffset =
    kK3MxFp8DW13QuantRowsPerGroup *
        kK3MxFp8DW2W13QuantFeatureSubtile *
        sizeof(cutlass::bfloat16_t) +
    2u * kK3MxFp8DW13QuantRowsPerGroup *
        kK3MxFp8DW2W13QuantFeatureSubtile *
        sizeof(cutlass::float_e4m3_t) +
    2u * kK3MxFp8DW2W13QuantScaleFeatureTile * sizeof(uint32_t);
constexpr uint32_t kK3MxFp8DW2W13QuantPayloadBytes =
    kK3MxFp8DW2W13QuantBarrierOffset +
    2u * sizeof(cutlass::arch::ClusterTransactionBarrier) +
    4u * sizeof(uint32_t);
constexpr uint32_t kK3MxFp8DW2W13QuantControlOffset =
    kK3MxFp8DW2W13QuantBarrierOffset +
    2u * sizeof(cutlass::arch::ClusterTransactionBarrier);
constexpr uint32_t kK3MxFp8DW2W13QuantTicketWord = 0u;
constexpr uint32_t kK3MxFp8DW2W13QuantStopWord = 1u;
constexpr uint32_t kK3MxFp8DW2W13QuantScratchBytes =
    10u * 1024u;
constexpr uint32_t kK3MxFp8DW2PersistentEnginesPerCTA = 8u;
constexpr uint32_t kK3MxFp8DW2PersistentScratchBytes =
    kK3MxFp8DW2PersistentEnginesPerCTA *
    kK3MxFp8DW2W13QuantScratchBytes;
struct alignas(8) K3MxFp8DW2W13QuantContext {
    const int* expert_counts;
    const K3BackwardRangeSet* backward_ranges;
    const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
        tensor_map_pack;
    uint32_t* a_primary_scales;
    uint32_t* a_residual_scales;
    uint32_t* b_primary_scales;
    uint32_t* b_residual_scales;
    uint32_t* state;
    uint32_t k_capacity;
};
struct alignas(4) K3MxFp8DW2W13QuantWork {
    uint32_t expert;
    uint32_t local_group;
    uint32_t feature_begin;
    uint32_t reserved;
};
// NamedBarrier's uint32_t API adds CUTLASS's reserved eight-barrier prefix,
// whereas the parent's publication callback passes hardware ID 15 directly
// to raw PTX. Four engine-local barriers therefore use user IDs 1..4, which
// map to hardware IDs 9..12 and cannot alias the publication barrier.
constexpr uint32_t kK3MxFp8DW13QuantFirstUserNamedBarrier = 1u;
constexpr uint32_t kK3MxFp8WgradPublishHardwareNamedBarrier = 15u;
constexpr uint32_t kK3MxFp8DW13QuantFirstHardwareNamedBarrier =
    cutlass::arch::NamedBarrier::ReservedNamedBarrierCount +
    kK3MxFp8DW13QuantFirstUserNamedBarrier;
constexpr uint32_t kK3MxFp8DW13QuantLastHardwareNamedBarrier =
    kK3MxFp8DW13QuantFirstHardwareNamedBarrier +
    kK3MxFp8DW13QuantNumEngines - 1u;
// The exact body keeps the scheduler's 256x128 logical task but computes it
// as two independently retired 128x128 output panels.  The second panel adds
// one full/empty TMEM-barrier pair (16 bytes) without moving the 128-byte
// aligned quantizer scratch boundary.
constexpr uint32_t kK3MxFp8DW13QuantBodySmemBytes = 153732u;
constexpr uint32_t kK3MxFp8DW13QuantScratchBegin =
    math::constexpr_align(kK3MxFp8DW13QuantBodySmemBytes, 128u);
constexpr uint32_t kK3MxFp8DW13QuantEnginePayloadBytes =
    kK3MxFp8DW13QuantRowsPerGroup * kK3MxFp8DW13QuantFeatureTile *
        sizeof(cutlass::bfloat16_t) +
    2u * kK3MxFp8DW13QuantRowsPerGroup *
        kK3MxFp8DW13QuantFeatureTile *
        sizeof(cutlass::float_e4m3_t) +
    2u * kK3MxFp8DW13QuantFeatureTile * sizeof(uint32_t) +
    sizeof(cutlass::arch::ClusterTransactionBarrier) + 4u * sizeof(uint32_t);
constexpr uint32_t kK3MxFp8DW13QuantEngineStride =
    math::constexpr_align(kK3MxFp8DW13QuantEnginePayloadBytes, 128u);
constexpr uint32_t kK3MxFp8DW13QuantSmemEnd =
    kK3MxFp8DW13QuantScratchBegin +
    kK3MxFp8DW13QuantNumEngines * kK3MxFp8DW13QuantEngineStride;
static_assert(kK3MxFp8DW13QuantEnginePayloadBytes == 17432u);
static_assert(kK3MxFp8DW13QuantEngineStride == 17536u);
static_assert(
    kK3MxFp8DW13QuantFirstUserNamedBarrier +
            kK3MxFp8DW13QuantNumEngines - 1u <
        cutlass::arch::NamedBarrier::HardwareMaxNumNamedBarriers -
            cutlass::arch::NamedBarrier::ReservedNamedBarrierCount);
static_assert(kK3MxFp8DW13QuantFirstHardwareNamedBarrier == 9u);
static_assert(kK3MxFp8DW13QuantLastHardwareNamedBarrier == 12u);
static_assert(
    kK3MxFp8DW13QuantLastHardwareNamedBarrier <
    kK3MxFp8WgradPublishHardwareNamedBarrier);
static_assert(kK3MxFp8DW13QuantScratchBegin == 153856u);
static_assert(kK3MxFp8DW13QuantSmemEnd == 224000u);
static_assert(kK3MxFp8WgradStreamingSmemBytes == 224000u);
static_assert(
    kK3MxFp8DW13QuantSmemEnd <= kK3MxFp8WgradStreamingSmemBytes);
static_assert(kK3MxFp8DW2W13QuantThreads == 128u);
static_assert(
    kK3MxFp8DW2W13QuantFeatureSubtile *
            kK3MxFp8DW2W13QuantSubtilesPerPanel ==
        kK3MxFp8DW13QuantFeatureTile,
    "Two 64-feature subtiles must publish one existing 128-panel bit");
static_assert(kK3MxFp8DW2W13QuantScaleFeatureTile == 128u);
static_assert(kK3MxFp8DW2W13QuantPayloadBytes == 9248u);
static_assert(kK3MxFp8DW2W13QuantScratchBytes == 10240u);
static_assert(kK3MxFp8DW2W13QuantBarrierOffset == 9216u);
static_assert(kK3MxFp8DW2W13QuantControlOffset == 9232u);
static_assert(kK3MxFp8DW2PersistentScratchBytes == 81920u);
static_assert(
    kK3MxFp8DW2PersistentScratchBytes <
        kK3MxFp8DW13QuantBodySmemBytes,
    "Eight retired-body dW2 engines must not touch the suffix handoff");
static_assert(sizeof(K3MxFp8DW2W13QuantContext) == 72u);
static_assert(
    kK3MxFp8DW2W13QuantPayloadBytes +
            sizeof(K3MxFp8DW2W13QuantContext) <=
        kK3MxFp8DW2W13QuantScratchBytes);
static_assert(sizeof(K3MxFp8DW2W13QuantWork) == 16u);
static_assert(
    kK3MxFp8DW2W13QuantPayloadBytes +
            sizeof(K3MxFp8DW2W13QuantContext) +
            sizeof(K3MxFp8DW2W13QuantWork) <=
        kK3MxFp8DW2W13QuantScratchBytes);
static_assert(
    kK3MxFp8DW2W13QuantPayloadBytes +
            sizeof(K3MxFp8DW2W13QuantContext) +
            sizeof(K3MxFp8DW2W13QuantWork) ==
        9336u);

namespace detail {

template <uint32_t kNumExperts, uint32_t kMaxRanges>
CUTLASS_DEVICE uint32_t k3_mxfp8_find_prefix_owner(
        const uint32_t* prefix, uint32_t value) {
    uint32_t lo = 0;
    uint32_t hi = kNumExperts;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (value < prefix[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo - 1;
}

template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kMaxRanges>
CUTLASS_DEVICE uint32_t k3_mxfp8_expert_source_pool_row(
        const uint32_t expert,
        uint32_t local_k,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t* physical_range_prefix) {
    #pragma unroll
    for (uint32_t reverse_iteration = 0;
         reverse_iteration < kMaxRanges; ++reverse_iteration) {
        if (reverse_iteration >= backward_ranges.num_ranges)
            break;
        const uint32_t range_idx =
            backward_ranges.reverse_range_index(reverse_iteration);
        const uint32_t count = static_cast<uint32_t>(
            __ldg(
                expert_counts +
                backward_ranges.expert_counts_begin(
                    range_idx, kNumExperts) +
                expert));
        const uint32_t padded_k = math::ceil_div(count, kBlockM) * kBlockM;
        if (local_k < padded_k) {
            return physical_range_prefix[
                range_idx * (kNumExperts + 1u) + expert] + local_k;
        }
        local_k -= padded_k;
    }
    return static_cast<uint32_t>(-1);
}

template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kMaxRanges>
CUTLASS_DEVICE uint32_t k3_mxfp8_source_pool_row(
        const uint32_t global_k,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t* value_prefix,
        const uint32_t* physical_range_prefix) {
    const uint32_t expert =
        k3_mxfp8_find_prefix_owner<kNumExperts, kMaxRanges>(
            value_prefix, global_k);
    return k3_mxfp8_expert_source_pool_row<
        kNumExperts, kBlockM, kMaxRanges>(
            expert, global_k - value_prefix[expert], expert_counts,
            backward_ranges, physical_range_prefix);
}

CUTLASS_DEVICE uint32_t k3_mxfp8_swizzle_128b_offset(
        const uint32_t row, const uint32_t column) {
    const uint32_t row_in_atom = row & 7u;
    return (row >> 3) * 8u * 128u + row_in_atom * 128u +
        (((column >> 4) ^ row_in_atom) << 4) + (column & 15u);
}

CUTLASS_DEVICE void k3_mxfp8_scale_pair(
        const float amax, float& scale, float& scale_inv,
        uint32_t& scale_byte) {
    if (amax == 0.0f) {
        scale = 1.0f;
        scale_inv = 1.0f;
        scale_byte = 0x7fu;
        return;
    }
    float2 sf;
    float2 sf_inv;
    math::get_e4m3_sf_and_sf_inv(
        make_float2(amax, 0.0f), sf, sf_inv);
    scale = sf.x;
    scale_inv = sf_inv.x;
    scale_byte =
        ((*reinterpret_cast<const uint32_t*>(&scale)) >> 23) & 0xffu;
}

/** Initialize one private dW2 quantization engine.
 *
 * The caller elects exactly one leader for the engine and supplies the
 * rendezvous that publishes this initialization to its other 127 threads.
 * Keeping the leader work separate lets the live W13 path initialize one
 * engine with a CTA barrier and the retired-body producer path initialize
 * eight disjoint engines with the same numerical and TensorMap context.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kBlockM>
CUTLASS_DEVICE void initialize_k3_mxfp8_dw2_quant_scratch_leader(
        uint8_t* quant_scratch,
        const int* expert_counts,
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        uint32_t* state,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    auto* const barriers = reinterpret_cast<Barrier*>(
        quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset);
    auto* const control = reinterpret_cast<uint32_t*>(
        quant_scratch + kK3MxFp8DW2W13QuantControlOffset);
    auto* const context =
        reinterpret_cast<K3MxFp8DW2W13QuantContext*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
    const auto scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden, k_capacity, kBlockM);
    const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden, k_capacity, kBlockM);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    auto* const packed_scale_arena =
        scale_arena + scale_layout.raw_bytes;
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            scale_layout, dw13_scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    barriers[0].init(1u);
    barriers[1].init(kK3MxFp8DW2W13QuantThreads);
    control[0] = 0u;
    control[1] = 0u;
    control[2] = 0u;
    control[3] = 0u;
    *context = {
        expert_counts,
        backward_ranges,
        tensor_map_pack,
        reinterpret_cast<uint32_t*>(
            packed_scale_arena + scale_layout.packed_a_primary),
        reinterpret_cast<uint32_t*>(
            packed_scale_arena + scale_layout.packed_a_residual),
        reinterpret_cast<uint32_t*>(
            packed_scale_arena + scale_layout.packed_b_primary),
        reinterpret_cast<uint32_t*>(
            packed_scale_arena + scale_layout.packed_b_residual),
        state,
        k_capacity,
    };
    const auto* const maps = tensor_map_pack->maps;
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerSourceAMap);
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerSourceBMap);
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerValueAPrimaryMap);
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerValueAResidualMap);
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerValueBPrimaryMap);
    cute::prefetch_tma_descriptor(
        maps + kK3MxFp8DW2ProducerValueBResidualMap);
    cutlass::arch::fence_barrier_init();
}

/** Initialize the elastic dW2 engine before the W13 role split.
 *
 * Every parent thread must call this helper while the CTA is converged. The
 * private participant mbarrier replaces a named barrier, whose IDs are still
 * owned by W13's loader, residual producer, and epilogue roles.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kBlockM>
CUTLASS_DEVICE void initialize_k3_mxfp8_dw2_w13_quant_scratch(
        uint8_t* quant_scratch,
        const int* expert_counts,
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        uint32_t* state,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack) {
    if (threadIdx.x == 0u) {
        initialize_k3_mxfp8_dw2_quant_scratch_leader<
            kHidden, kIntermediateHidden, kBlockM>(
                quant_scratch, expert_counts, backward_ranges,
                k_capacity, scale_arena_source, state,
                tensor_map_pack);
    }
    __syncthreads();
}

/** Ask the W13-time dW2 engine to stop after its current complete panel.
 *
 * The first selected post-W13 producer controller release-stores this global
 * generation request. All 148 borrowed engine leaders acquire-check it before
 * claiming their next ticket; an already claimed panel always completes and
 * publishes atomically. Repeated stores of the same generation value are
 * harmless and avoid allocating a separate controller-election word.
 */
CUTLASS_DEVICE void request_k3_mxfp8_dw2_w13_quant_stop(
        uint32_t* handoff_request) {
    constexpr uint32_t kRequested = 1u;
    asm volatile(
        "st.release.gpu.global.u32 [%0], %1;"
        :: "l"(handoff_request), "r"(kRequested)
        : "memory");
}

/** Quantize and store one exact group-32 by 64-feature dW2 subtile.
 *
 * The persistent controller publishes the three scalar coordinates in the
 * scratch-tail work record. Outlining this dependency-heavy primary/residual
 * conversion keeps its arrays and pointer arithmetic out of the W13 parent
 * and the panel scheduler. The caller carries the participant phase across
 * groups: one generation retires a pending value store, when present, and
 * two generations bracket publication of the current primary/residual pair.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, bool kIsA,
          bool kPersistentRole = false>
CUTLASS_DEVICE __noinline__ uint32_t
k3_mxfp8_quantize_dw2_group32_subtile(
        uint8_t* quant_scratch,
        const uint32_t engine_thread,
        uint32_t& participant_phase,
        uint32_t& load_phase,
        const bool value_store_pending) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kRows = kK3MxFp8DW13QuantRowsPerGroup;
    constexpr uint32_t kFeatures =
        kK3MxFp8DW2W13QuantFeatureSubtile;
    constexpr uint32_t kSourceBytes =
        kRows * kFeatures * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kValueBytes =
        kRows * kFeatures * sizeof(cutlass::float_e4m3_t);

    const uint32_t warp_in_engine = engine_thread / 32u;
    const uint32_t feature =
        warp_in_engine * 16u + (threadIdx.x & 15u);
    const uint32_t row_half = (threadIdx.x & 31u) >> 4u;
    const uint32_t row_begin = row_half * 16u;
    auto* const source_smem =
        reinterpret_cast<cutlass::bfloat16_t*>(quant_scratch);
    auto* const primary_smem = quant_scratch + kSourceBytes;
    auto* const residual_smem = primary_smem + kValueBytes;
    auto* const barriers = reinterpret_cast<Barrier*>(
        quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset);
    auto* const load_barrier = barriers;
    auto* const participant_barrier = barriers + 1;
    const auto* const context =
        reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
    const auto* const work =
        reinterpret_cast<const K3MxFp8DW2W13QuantWork*>(context + 1);
    const uint32_t expert = work->expert;
    const uint32_t local_group = work->local_group;
    const uint32_t feature_begin = work->feature_begin;
    const uint32_t local_row = local_group * kRows;
    auto* const state = context->state;
    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const auto* const maps = context->tensor_map_pack->maps;
    const auto* const source_map = maps + (kIsA
        ? kK3MxFp8DW2ProducerSourceAMap
        : kK3MxFp8DW2ProducerSourceBMap);
    const auto* const primary_map = maps + (kIsA
        ? kK3MxFp8DW2ProducerValueAPrimaryMap
        : kK3MxFp8DW2ProducerValueBPrimaryMap);
    const auto* const residual_map = maps + (kIsA
        ? kK3MxFp8DW2ProducerValueAResidualMap
        : kK3MxFp8DW2ProducerValueBResidualMap);
    const uint32_t source_row =
        k3_mxfp8_expert_source_pool_row<
            kNumExperts, kBlockM, kK3MaxBackwardRanges>(
                expert, local_row, context->expert_counts,
                *context->backward_ranges, physical_range_prefix);
    DG_DEVICE_ASSERT(
        source_row != static_cast<uint32_t>(-1) &&
        source_row + kRows <= context->k_capacity);

    if (engine_thread == 0u) {
        tma::copy<kFeatures, kRows, 0, cutlass::bfloat16_t>(
            source_map, load_barrier, source_smem,
            feature_begin, source_row);
        load_barrier->arrive_and_expect_tx(kSourceBytes);
    }
    load_barrier->wait(load_phase);
    load_phase ^= 1u;

    // The source tile is disjoint from both output tiles.  Load the next
    // group while the previous primary/residual stores are still in flight,
    // then retire those stores only immediately before their buffers are
    // overwritten.
    if (value_store_pending) {
        if (engine_thread == 0u)
            ptx::tma_store_wait<0>();
        participant_barrier->arrive();
        participant_barrier->wait(participant_phase);
        participant_phase ^= 1u;
    }

    float primary_amax = 0.0f;
    #pragma unroll 1
    for (uint32_t row = row_begin; row < row_begin + 16u; ++row) {
        const float value = static_cast<float>(
            source_smem[row * kFeatures + feature]);
        primary_amax = cute::max(primary_amax, cute::abs(value));
    }
    primary_amax = cute::max(
        primary_amax,
        __shfl_xor_sync(0xffffffffu, primary_amax, 16));
    float primary_scale = 1.0f;
    float primary_scale_inv = 1.0f;
    uint32_t primary_scale_byte = 0x7fu;
    k3_mxfp8_scale_pair(
        primary_amax, primary_scale, primary_scale_inv,
        primary_scale_byte);

    float residual_amax = 0.0f;
    #pragma unroll 1
    for (uint32_t row = row_begin; row < row_begin + 16u; row += 4u) {
        float values[4];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            values[i] = static_cast<float>(
                source_smem[(row + i) * kFeatures + feature]);
        }
        const auto primary = __nv_fp8x4_e4m3(make_float4(
            values[0] * primary_scale_inv,
            values[1] * primary_scale_inv,
            values[2] * primary_scale_inv,
            values[3] * primary_scale_inv));
        const float4 primary_float = static_cast<float4>(primary);
        const uint32_t primary_bits = primary.__x;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            const float primary_value = i == 0u
                ? primary_float.x : i == 1u
                ? primary_float.y : i == 2u
                ? primary_float.z : primary_float.w;
            const cutlass::bfloat16_t rounded_residual(
                values[i] - primary_value * primary_scale);
            source_smem[(row + i) * kFeatures + feature] =
                rounded_residual;
            primary_smem[(row + i) * kFeatures + feature] =
                static_cast<uint8_t>(primary_bits >> (i * 8u));
            residual_amax = cute::max(
                residual_amax,
                cute::abs(static_cast<float>(rounded_residual)));
        }
    }
    residual_amax = cute::max(
        residual_amax,
        __shfl_xor_sync(0xffffffffu, residual_amax, 16));
    float residual_scale = 1.0f;
    float residual_scale_inv = 1.0f;
    uint32_t residual_scale_byte = 0x7fu;
    k3_mxfp8_scale_pair(
        residual_amax, residual_scale, residual_scale_inv,
        residual_scale_byte);
    #pragma unroll
    for (uint32_t row = row_begin; row < row_begin + 16u; row += 4u) {
        float residual_values[4];
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            residual_values[i] = static_cast<float>(
                source_smem[(row + i) * kFeatures + feature]);
        }
        const auto residual = __nv_fp8x4_e4m3(make_float4(
            residual_values[0] * residual_scale_inv,
            residual_values[1] * residual_scale_inv,
            residual_values[2] * residual_scale_inv,
            residual_values[3] * residual_scale_inv));
        const uint32_t residual_bits = residual.__x;
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            residual_smem[(row + i) * kFeatures + feature] =
                static_cast<uint8_t>(residual_bits >> (i * 8u));
        }
    }

    cute::tma_store_fence();
    participant_barrier->arrive();
    participant_barrier->wait(participant_phase);
    participant_phase ^= 1u;
    if (engine_thread == 0u) {
        const uint32_t destination_row =
            value_prefix[expert] + local_row;
        cute::SM90_TMA_STORE_2D::copy(
            primary_map, primary_smem,
            feature_begin, destination_row);
        cute::SM90_TMA_STORE_2D::copy(
            residual_map, residual_smem,
            feature_begin, destination_row);
        cute::tma_store_arrive();
    }
    participant_barrier->arrive();
    participant_barrier->wait(participant_phase);
    participant_phase ^= 1u;
    return primary_scale_byte | (residual_scale_byte << 8u);
}

CUTLASS_DEVICE void k3_mxfp8_dw2_w13_engine_sync(
        uint8_t* quant_scratch,
        uint32_t& participant_phase) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    auto* const barriers = reinterpret_cast<Barrier*>(
        quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset);
    auto* const participant_barrier = barriers + 1;
    participant_barrier->arrive();
    participant_barrier->wait(participant_phase);
    participant_phase ^= 1u;
}

/** Produce and store one packed scale row for one 128-feature panel.
 *
 * The engine owns both 64-feature value subtiles.  Keeping the scale-row loop
 * outside the subtile loop lets both halves coexist in one UTCCP-native
 * 128-word shared tile, which is then published with two contiguous bulk
 * stores.  This avoids per-word generic global stores and their per-writer
 * proxy/release fences while preserving the existing 64-feature value maps.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, bool kIsA,
          bool kPersistentRole = false>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_produce_dw2_scale_row_during_w13(
        uint8_t* quant_scratch,
        const uint32_t engine_thread,
        const uint32_t expert,
        const uint32_t local_scale_row,
        const uint32_t panel_feature_begin,
        uint32_t& participant_phase,
        uint32_t& load_phase) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kRowsPerGroup = 32u;
    constexpr uint32_t kGroupsPerScaleRow = 4u;
    constexpr uint32_t kFeatureSubtile =
        kK3MxFp8DW2W13QuantFeatureSubtile;
    constexpr uint32_t kScaleFeatureTile =
        kK3MxFp8DW2W13QuantScaleFeatureTile;
    constexpr uint32_t kSourceBytes =
        kRowsPerGroup * kFeatureSubtile *
        sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kValueBytes =
        kRowsPerGroup * kFeatureSubtile *
        sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kScaleBytes =
        kScaleFeatureTile * sizeof(uint32_t);
    constexpr uint32_t kSourceWidth =
        kIsA ? kHidden : kIntermediateHidden;

    const uint32_t feature_in_subtile =
        (engine_thread / 32u) * 16u + (threadIdx.x & 15u);
    const uint32_t row_half = (threadIdx.x & 31u) >> 4u;
    auto* const work =
        reinterpret_cast<K3MxFp8DW2W13QuantWork*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes +
            sizeof(K3MxFp8DW2W13QuantContext));
    const uint32_t num_groups = [&]() {
        const auto* const context =
            reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
                quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
        const auto* const value_prefix =
            context->state + Prefix::kValuePrefix;
        return (value_prefix[expert + 1u] - value_prefix[expert]) /
            kRowsPerGroup;
    }();
    if (row_half == 0u) {
        auto* const primary_scale_smem = reinterpret_cast<uint32_t*>(
            quant_scratch + kSourceBytes + 2u * kValueBytes);
        auto* const residual_scale_smem =
            primary_scale_smem + kScaleFeatureTile;
        #pragma unroll
        for (uint32_t subtile = 0u;
             subtile < kK3MxFp8DW2W13QuantSubtilesPerPanel; ++subtile) {
            const uint32_t logical_feature =
                subtile * kFeatureSubtile + feature_in_subtile;
            const uint32_t native_feature =
                k3_mxfp8_utccp_scale_feature(logical_feature);
            primary_scale_smem[native_feature] = 0x7f7f7f7fu;
            residual_scale_smem[native_feature] = 0x7f7f7f7fu;
        }
    }
    if (engine_thread == 0u) {
        work->expert = expert;
    }

    bool value_store_pending = false;
    #pragma unroll
    for (uint32_t subtile = 0u;
         subtile < kK3MxFp8DW2W13QuantSubtilesPerPanel; ++subtile) {
        const uint32_t feature_begin =
            panel_feature_begin + subtile * kFeatureSubtile;
        if (engine_thread == 0u)
            work->feature_begin = feature_begin;
        #pragma unroll 1
        for (uint32_t group_in_scale_row = 0u;
             group_in_scale_row < kGroupsPerScaleRow;
             ++group_in_scale_row) {
            const uint32_t local_group =
                local_scale_row * kGroupsPerScaleRow + group_in_scale_row;
            if (local_group >= num_groups)
                continue;
            if (engine_thread == 0u)
                work->local_group = local_group;
            k3_mxfp8_dw2_w13_engine_sync(
                quant_scratch, participant_phase);
            const uint32_t scale_bytes =
                k3_mxfp8_quantize_dw2_group32_subtile<
                    kHidden, kIntermediateHidden, kNumExperts,
                    kBlockM, kNumSMs, kIsA, kPersistentRole>(
                        quant_scratch, engine_thread,
                        participant_phase, load_phase,
                        value_store_pending);
            value_store_pending = true;
            if (row_half == 0u) {
                auto* const primary_scale_smem =
                    reinterpret_cast<uint32_t*>(
                        quant_scratch + kSourceBytes + 2u * kValueBytes);
                auto* const residual_scale_smem =
                    primary_scale_smem + kScaleFeatureTile;
                const uint32_t logical_feature =
                    subtile * kFeatureSubtile + feature_in_subtile;
                const uint32_t native_feature =
                    k3_mxfp8_utccp_scale_feature(logical_feature);
                reinterpret_cast<uint8_t*>(primary_scale_smem)[
                    native_feature * 4u + group_in_scale_row] =
                    static_cast<uint8_t>(scale_bytes);
                reinterpret_cast<uint8_t*>(residual_scale_smem)[
                    native_feature * 4u + group_in_scale_row] =
                    static_cast<uint8_t>(scale_bytes >> 8u);
            }
        }
    }
    // Every scale writer publishes both of its shared words to the async proxy
    // before the elected lane issues the two contiguous scale stores.
    if (row_half == 0u)
        cute::tma_store_fence();
    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
    if (engine_thread == 0u) {
        if (value_store_pending)
            ptx::tma_store_wait<0>();
        const auto* const context =
            reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
                quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
        const auto* const scale_prefix =
            context->state + Prefix::kScalePrefix;
        auto* const primary_scale_smem = reinterpret_cast<uint32_t*>(
            quant_scratch + kSourceBytes + 2u * kValueBytes);
        auto* const residual_scale_smem =
            primary_scale_smem + kScaleFeatureTile;
        const uint32_t global_scale_row =
            scale_prefix[expert] + local_scale_row;
        const uint64_t scale_offset =
            static_cast<uint64_t>(global_scale_row) * kSourceWidth +
            panel_feature_begin;
        auto* const primary_scales = kIsA
            ? context->a_primary_scales
            : context->b_primary_scales;
        auto* const residual_scales = kIsA
            ? context->a_residual_scales
            : context->b_residual_scales;
        ptx::tma_store_1d(
            primary_scales + scale_offset,
            primary_scale_smem, kScaleBytes);
        ptx::tma_store_1d(
            residual_scales + scale_offset,
            residual_scale_smem, kScaleBytes);
        cute::tma_store_arrive();
        ptx::tma_store_wait<0>();
    }
    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
}

/** Produce both 64-feature subtiles and publish one ready dW2 panel. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, bool kIsA,
          bool kPersistentRole = false>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_produce_dw2_panel_during_w13(
        uint8_t* quant_scratch,
        const uint32_t engine_thread,
        const uint32_t expert,
        const uint32_t feature_panel,
        uint32_t& participant_phase,
        uint32_t& load_phase) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kFeaturePanel = 128u;
    constexpr uint32_t kAFeaturePanels = kHidden / kFeaturePanel;
    constexpr uint32_t kSourceWidth =
        kIsA ? kHidden : kIntermediateHidden;
    const auto* const context =
        reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
    auto* const state = context->state;
    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const scale_prefix = state + Prefix::kScalePrefix;
    const uint32_t expert_k =
        value_prefix[expert + 1u] - value_prefix[expert];
    const uint32_t num_scale_rows =
        scale_prefix[expert + 1u] - scale_prefix[expert];
    DG_DEVICE_ASSERT(
        expert_k != 0u && expert_k % 32u == 0u &&
        (feature_panel + 1u) * kFeaturePanel <= kSourceWidth);

    const uint32_t panel_feature_begin = feature_panel * kFeaturePanel;
    for (uint32_t local_scale_row = 0u;
         local_scale_row < num_scale_rows; ++local_scale_row) {
        k3_mxfp8_produce_dw2_scale_row_during_w13<
            kHidden, kIntermediateHidden, kNumExperts,
            kBlockM, kNumSMs, kIsA, kPersistentRole>(
                quant_scratch, engine_thread, expert, local_scale_row,
                panel_feature_begin, participant_phase, load_phase);
    }

    if (engine_thread == 0u) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        const uint32_t panel =
            (kIsA ? 0u : kAFeaturePanels) + feature_panel;
        auto* const ready =
            state + Overlap::kDW2FeatureReadyMasks + expert * 2u;
        ptx::red_or_rel_gpu(
            ready + panel / 32u, 1u << (panel % 32u));
    }
    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
}

/** Claim and produce every panel of one exact dW2 operand phase. */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, bool kIsA,
          bool kStopAtPanelBoundary = false,
          bool kPersistentRole = false>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_produce_dw2_phase_during_w13(
        uint8_t* quant_scratch,
        const uint32_t engine_thread,
        uint32_t& participant_phase,
        uint32_t& load_phase) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    constexpr uint32_t kFeaturePanel = 128u;
    constexpr uint32_t kNumFeaturePanels =
        (kIsA ? kHidden : kIntermediateHidden) / kFeaturePanel;
    constexpr uint32_t kNoTask = 0xffffffffu;
    const auto* const context =
        reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes);
    auto* const state = context->state;
    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);
    auto* const barriers = reinterpret_cast<Barrier*>(
        quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset);
    auto* const control = reinterpret_cast<volatile uint32_t*>(
        barriers + 2);
    const uint32_t num_tasks = active_count * kNumFeaturePanels;

    while (true) {
        if (engine_thread == 0u) {
            if constexpr (kStopAtPanelBoundary) {
                if (ptx::ld_acq(
                        state + Overlap::kElasticDW2HandoffRequest) != 0u) {
                    control[kK3MxFp8DW2W13QuantTicketWord] = kNoTask;
                } else {
                    const uint32_t ticket = ptx::atomic_add_acq_rel(
                        state + Overlap::kElasticDW2ProducerCursor, 1u);
                    control[kK3MxFp8DW2W13QuantTicketWord] =
                        ticket < num_tasks ? ticket : kNoTask;
                }
            } else {
                const uint32_t ticket = ptx::atomic_add_acq_rel(
                    state + Overlap::kElasticDW2ProducerCursor, 1u);
                control[kK3MxFp8DW2W13QuantTicketWord] =
                    ticket < num_tasks ? ticket : kNoTask;
            }
        }
        k3_mxfp8_dw2_w13_engine_sync(
            quant_scratch, participant_phase);
        const uint32_t ticket =
            control[kK3MxFp8DW2W13QuantTicketWord];
        if (ticket == kNoTask)
            break;
        const uint32_t active = ticket / kNumFeaturePanels;
        const uint32_t feature_panel =
            ticket - active * kNumFeaturePanels;
        const uint32_t expert =
            state[Overlap::kActiveExperts + active];
        k3_mxfp8_produce_dw2_panel_during_w13<
            kHidden, kIntermediateHidden, kNumExperts,
            kBlockM, kNumSMs, kIsA, kPersistentRole>(
                quant_scratch, engine_thread, expert, feature_panel,
                participant_phase, load_phase);
    }
}

/** Stream exact dW2 panels from either a live or retired parent role.
 *
 * One 128-thread engine claims complete feature panels from the shared cursor.
 * The live-W13 specialization produces only A and honors the global handoff
 * request at the next panel boundary. It then contributes one nonblocking
 * arrival. Retired-body producer engines resume the same A cursor, and their
 * combined A barrier includes every borrowed-engine arrival before resetting
 * the cursor for B. This protects the A-source/B-destination alias without a
 * full-grid CTA barrier; consumer CTA pairs may already be polling ready-first
 * UMMA work while the bounded producer pairs finish both operands.
 *
 * `kABarrierParticipants` counts one borrowed engine per persistent CTA plus
 * every retired-body engine. A complete panel owns its values, packed scales,
 * proxy fences, and ready-bit publication, so stopping or changing roles can
 * never expose a partially prepared operand.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          bool kProduceA = true, bool kProduceB = true,
          bool kStopAtPanelBoundary = false,
          bool kSignalBorrowedCompletion = false,
          uint32_t kABarrierParticipants = kNumSMs>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_stream_dw2_operands_during_w13(
        uint8_t* quant_scratch,
        const uint32_t warp_idx,
        const uint32_t lane_idx,
        const uint32_t first_warp =
            kK3MxFp8DW2W13QuantFirstWarp) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    constexpr uint32_t kFeaturePanel = 128u;
    constexpr uint32_t kAFeaturePanels = kHidden / kFeaturePanel;
    constexpr uint32_t kBFeaturePanels =
        kIntermediateHidden / kFeaturePanel;
    static_assert(
        kNumSMs == 148u && kNumThreads == 1024u &&
            kHidden == 3584u && kIntermediateHidden == 3072u &&
            kAFeaturePanels == 28u && kBFeaturePanels == 24u &&
            kAFeaturePanels + kBFeaturePanels ==
                Overlap::kDW2FeaturePanelsPerExpert,
        "The elastic dW2 producer is exact K3 geometry");
    static_assert(kProduceA,
                  "Every exact dW2 stream must establish the A generation");
    static_assert(!kSignalBorrowedCompletion || !kProduceB,
                  "A borrowed W13 engine never crosses the A/B alias edge");
    static_assert(
        kABarrierParticipants >= kNumSMs,
        "The A/B edge must include every borrowed W13 engine");
    static_assert(
        kK3MxFp8DW2W13QuantPayloadBytes == 9248u,
        "The elastic producer scratch layout must remain allocation-free");

    DG_DEVICE_ASSERT(first_warp + kK3MxFp8DW2W13QuantWarps <=
                     kNumThreads / 32u);
    if (warp_idx < first_warp ||
        warp_idx >= first_warp + kK3MxFp8DW2W13QuantWarps)
        return;

    const uint32_t engine_thread =
        (warp_idx - first_warp) * 32u + lane_idx;
    uint32_t participant_phase = 0u;
    uint32_t load_phase = 0u;

    // The converged owner initialized both private mbarriers before roles
    // diverged. This first generation publishes the context to all 128 lanes.
    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
    k3_mxfp8_produce_dw2_phase_during_w13<
        kHidden, kIntermediateHidden, kNumExperts,
        kBlockM, kNumSMs, true, kStopAtPanelBoundary, kProduceB>(
            quant_scratch, engine_thread,
            participant_phase, load_phase);

    // All threads have retired their last complete A panel before the leader
    // either signals a borrowed-engine arrival or joins the A/B edge.
    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
    auto* const state =
        reinterpret_cast<const K3MxFp8DW2W13QuantContext*>(
            quant_scratch + kK3MxFp8DW2W13QuantPayloadBytes)->state;
    if constexpr (kSignalBorrowedCompletion) {
        if (engine_thread == 0u) {
            __threadfence();
            const uint32_t arrival = ptx::atomic_add_acq_rel(
                state + Overlap::kDW2SubsetBarrierCount, 1u);
            DG_DEVICE_ASSERT(arrival < kABarrierParticipants);
            // A late W13 CTA may contribute the terminal combined arrival
            // after every persistent engine is already waiting. Whichever
            // engine is last owns the reset; borrowed engines never wait.
            if (arrival + 1u == kABarrierParticipants) {
                atomicExch(
                    state + Overlap::kDW2SubsetBarrierCount, 0u);
                atomicExch(
                    state + Overlap::kElasticDW2ProducerCursor, 0u);
                __threadfence();
                atomicAdd(
                    state + Overlap::kDW2SubsetBarrierSense, 1u);
            }
        }
    }

    if constexpr (kProduceB) {
        if (engine_thread == 0u) {
            const uint32_t old_sense = ptx::ld_acq(
                state + Overlap::kDW2SubsetBarrierSense);
            __threadfence();
            const uint32_t arrival = ptx::atomic_add_acq_rel(
                state + Overlap::kDW2SubsetBarrierCount, 1u);
            DG_DEVICE_ASSERT(arrival < kABarrierParticipants);
            if (arrival + 1u == kABarrierParticipants) {
                atomicExch(
                    state + Overlap::kDW2SubsetBarrierCount, 0u);
                atomicExch(
                    state + Overlap::kElasticDW2ProducerCursor, 0u);
                __threadfence();
                atomicAdd(
                    state + Overlap::kDW2SubsetBarrierSense, 1u);
            } else {
                while (ptx::ld_acq(
                           state + Overlap::kDW2SubsetBarrierSense) ==
                       old_sense) {
                    __nanosleep(64);
                }
            }
            // The combined A retirement is observed through the generic
            // proxy, while the first B panel overwrites the aliased arena
            // through TMA. Bridge that acquire into the async proxy before
            // this engine is allowed to issue any B load/store transaction.
            asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        k3_mxfp8_dw2_w13_engine_sync(
            quant_scratch, participant_phase);
        k3_mxfp8_produce_dw2_phase_during_w13<
            kHidden, kIntermediateHidden, kNumExperts,
            kBlockM, kNumSMs, false, false, kProduceB>(
                quant_scratch, engine_thread,
                participant_phase, load_phase);
    }

    k3_mxfp8_dw2_w13_engine_sync(
        quant_scratch, participant_phase);
    if (engine_thread == 0u) {
        auto* const load_barrier = reinterpret_cast<Barrier*>(
            quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset);
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(load_barrier));
    }
    // The participant barrier remains valid until its owner gathers all
    // engine lanes. Invalidating it here can race a peer's final wait.
}

/** Finish dW2 preparation with eight engines in a retired parent CTA.
 *
 * The caller selects a pair-aligned, compile-time-bounded CTA subset. All
 * eight 4-warp engines reuse `[0, 81920)` of the retired body scratch, resume
 * A at the cursor left by live W13 engines, cross the combined engine-only
 * alias barrier, and produce B. A CTA joins the unified UMMA body only after
 * its own eight engines finish; other complete CTA pairs enter immediately.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kNumProducerCTAs>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_finish_dw2_with_persistent_producer_cta(
        const int* expert_counts,
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t warp_idx,
        const uint32_t lane_idx) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kThreadsPerEngine =
        kK3MxFp8DW2W13QuantThreads;
    constexpr uint32_t kProducerEngines =
        kNumProducerCTAs * kK3MxFp8DW2PersistentEnginesPerCTA;
    constexpr uint32_t kABarrierParticipants =
        kNumSMs + kProducerEngines;
    static_assert(
        kNumThreads == 1024u &&
            kK3MxFp8DW2PersistentEnginesPerCTA * kThreadsPerEngine ==
                kNumThreads,
        "The persistent dW2 producer must use all eight warpgroups");
    static_assert(kNumProducerCTAs > 0u && kNumProducerCTAs % 2u == 0u &&
                      kNumProducerCTAs < kNumSMs,
                  "Persistent producers must reserve complete CTA pairs");
    static_assert(
        kK3MxFp8DW2PersistentScratchBytes <=
            kK3MxFp8DW13QuantBodySmemBytes,
        "Persistent dW2 engines must fit below the suffix handoff");

    const uint32_t engine = threadIdx.x / kThreadsPerEngine;
    const uint32_t engine_thread = threadIdx.x % kThreadsPerEngine;
    uint8_t* const quant_scratch =
        smem_buffer + engine * kK3MxFp8DW2W13QuantScratchBytes;
    if (engine_thread == 0u) {
        initialize_k3_mxfp8_dw2_quant_scratch_leader<
            kHidden, kIntermediateHidden, kBlockM>(
                quant_scratch, expert_counts, backward_ranges,
                k_capacity, scale_arena_source, state,
                tensor_map_pack);
    }
    __syncthreads();

    const uint32_t first_warp =
        engine * kK3MxFp8DW2W13QuantWarps;
    k3_mxfp8_stream_dw2_operands_during_w13<
        kHidden, kIntermediateHidden, kNumExperts, kBlockM,
        kNumSMs, kNumThreads, true, true, false, false,
        kABarrierParticipants>(
            quant_scratch, warp_idx, lane_idx, first_warp);
    __syncthreads();

    if (engine_thread == 0u) {
        auto* const participant_barrier = reinterpret_cast<Barrier*>(
            quant_scratch + kK3MxFp8DW2W13QuantBarrierOffset) + 1;
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(
                participant_barrier));
    }
    __syncthreads();
}

/** Produce one transposed primary/residual operand without an allocation.
 *
 * A CTA handles complete 128x128 tiles.  Its first 128 threads issue one
 * 256-byte bulk row load each, then all 1024 threads quantize the Cartesian
 * product of four token-group32 blocks and 128 feature columns.  Each
 * group/feature pair is split between lanes 0/16 of one warp, preserving the
 * exact group-32 scale while shortening the per-lane dependency chain.
 * The primary pass overwrites each consumed BF16 source element with its
 * BF16-rounded residual.  That scratch reuse preserves the exact three-term
 * decomposition while avoiding a second primary FP8 conversion in the
 * residual pass.  Both FP8 planes are TMA stored in 128-byte-swizzled form.
 * In legacy mode, one-byte scales use a race-free global group-major scratch
 * plane and are compacted after all tiles retire.  Direct-packed mode instead
 * maps every group-32 scale byte to its expert-local uint32 scale row while it
 * quantizes and fills the unused bytes of a partial final row with 0x7f.  Each
 * byte has exactly one writer, including for empty, variable-length, and final
 * 64-row-tail layouts, so the fast path needs neither atomics nor new state.
 * Expert-local mode interprets `total_k` relative to `producer_expert`, maps
 * its source without a prefix search, and adds that expert's value prefix to
 * every destination coordinate.  This lets a two-CTA cluster publish one
 * expert without flattening its internal 64-row tail into the next expert.
 * Primary and residual destination strides are independent of each other and
 * of `source_width`. dW2-B can therefore retain the dead BF16 A source's 2H
 * byte pitch while placing its two I-wide planes at offsets zero and I. That
 * row-preserving alias keeps different experts byte-disjoint across producer
 * clusters; dW13-A-primary later consumes the 2I feature extent with the same
 * 2H physical pitch. A partial tile is scattered with its plane's stride, so
 * an expert-local 64-row tail never stores a 128-row TMA box into its neighbor.
 */
template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kMaxRanges, uint32_t kNumSMs,
          uint32_t kNumThreads, bool kDirectPackedScales,
          bool kExpertLocal, bool kFeatureMajor = false>
CUTLASS_DEVICE __noinline__ void k3_mxfp8_produce_operand(
        const cutlass::bfloat16_t* source,
        const uint32_t source_width,
        const uint32_t primary_destination_row_stride,
        const uint32_t residual_destination_row_stride,
        const uint32_t k_capacity,
        const uint32_t total_k,
        const uint32_t producer_expert,
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const uint32_t* value_prefix,
        const uint32_t* physical_range_prefix,
        uint8_t* primary_values,
        uint8_t* residual_values,
        uint8_t* primary_scale_output,
        uint8_t* residual_scale_output,
        const cute::TmaDescriptor& primary_map,
        const cute::TmaDescriptor& residual_map,
        const uint32_t producer_cta_idx,
        const uint32_t num_producer_ctas,
        uint8_t* smem_buffer,
        const uint32_t feature_panel_base = 0u,
        uint32_t* feature_ready_masks = nullptr) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    constexpr uint32_t kTile = 128;
    constexpr uint32_t kGroupsPerTile = kTile / 32u;
    constexpr uint32_t kLanesPerGroupFeature = 2u;
    constexpr uint32_t kQuantThreads =
        kGroupsPerTile * kTile * kLanesPerGroupFeature;
    static_assert(
        kNumThreads == kQuantThreads,
        "K3 MXFP8 operand production needs two lanes per group/feature pair");
    DG_DEVICE_ASSERT(
        num_producer_ctas > 0u && producer_cta_idx < num_producer_ctas);
    DG_DEVICE_ASSERT(
        primary_destination_row_stride >= source_width &&
        residual_destination_row_stride >= source_width);
    if constexpr (kExpertLocal)
        DG_DEVICE_ASSERT(producer_expert < kNumExperts);
    if constexpr (kFeatureMajor) {
        static_assert(kExpertLocal,
                      "Feature-major publication is expert-local only");
        DG_DEVICE_ASSERT(feature_ready_masks != nullptr);
    }
    const uint32_t value_row_begin = kExpertLocal
        ? value_prefix[producer_expert]
        : 0u;
    constexpr uint32_t kSourceBytes =
        kTile * kTile * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kQuantBytes = kTile * kTile;
    auto* source_smem =
        reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto* primary_smem = smem_buffer + kSourceBytes;
    auto* residual_smem = primary_smem + kQuantBytes;
    auto* load_barrier = reinterpret_cast<Barrier*>(
        residual_smem + kQuantBytes);

    if (threadIdx.x == 0) {
        load_barrier->init(1);
        cutlass::arch::fence_barrier_init();
        cute::prefetch_tma_descriptor(&primary_map);
        cute::prefetch_tma_descriptor(&residual_map);
    }
    __syncthreads();

    const uint32_t num_row_tiles = math::ceil_div(total_k, kTile);
    const uint32_t num_feature_tiles = source_width / kTile;
    const uint32_t num_tasks = num_row_tiles * num_feature_tiles;
    const uint32_t num_owned_feature_tiles =
        producer_cta_idx < num_feature_tiles
        ? math::ceil_div(
              num_feature_tiles - producer_cta_idx,
              num_producer_ctas)
        : 0u;
    const uint32_t num_scheduled_tasks = kFeatureMajor
        ? num_owned_feature_tiles * num_row_tiles : num_tasks;
    const uint32_t first_scheduled_task =
        kFeatureMajor ? 0u : producer_cta_idx;
    const uint32_t scheduled_task_stride =
        kFeatureMajor ? 1u : num_producer_ctas;
    uint32_t load_phase = 0;
    for (uint32_t task = first_scheduled_task;
         task < num_scheduled_tasks;
         task += scheduled_task_stride) {
        const uint32_t owned_feature_tile = kFeatureMajor
            ? task / num_row_tiles : 0u;
        const uint32_t row_tile = kFeatureMajor
            ? task - owned_feature_tile * num_row_tiles
            : task / num_feature_tiles;
        const uint32_t feature_tile = kFeatureMajor
            ? producer_cta_idx +
                  owned_feature_tile * num_producer_ctas
            : task - row_tile * num_feature_tiles;
        const uint32_t row_begin = row_tile * kTile;
        const uint32_t feature_begin = feature_tile * kTile;
        const uint32_t valid_rows = cute::min(kTile, total_k - row_begin);

        // Only the final partial tile needs an explicit zero tail.  The TMA
        // descriptors retain an aligned K capacity so its full 128-row store
        // remains in bounds and the GEMM's masked final load sees zeros.
        if (valid_rows != kTile) {
            constexpr uint32_t kWordsPerRow =
                kTile * sizeof(cutlass::bfloat16_t) / sizeof(uint32_t);
            auto* source_words = reinterpret_cast<uint32_t*>(source_smem);
            for (uint32_t idx =
                     valid_rows * kWordsPerRow + threadIdx.x;
                 idx < kTile * kWordsPerRow; idx += kNumThreads)
                source_words[idx] = 0u;
        }
        __syncthreads();

        if (threadIdx.x < valid_rows) {
            const uint32_t local_k = row_begin + threadIdx.x;
            uint32_t source_row = 0u;
            if constexpr (kExpertLocal) {
                source_row = k3_mxfp8_expert_source_pool_row<
                    kNumExperts, kBlockM, kMaxRanges>(
                        producer_expert, local_k, expert_counts,
                        backward_ranges, physical_range_prefix);
            } else {
                source_row = k3_mxfp8_source_pool_row<
                    kNumExperts, kBlockM, kMaxRanges>(
                        local_k, expert_counts, backward_ranges,
                        value_prefix, physical_range_prefix);
            }
            ptx::tma_load_1d(
                source_smem + threadIdx.x * kTile,
                source + static_cast<uint64_t>(source_row) * source_width +
                    feature_begin,
                load_barrier,
                kTile * sizeof(cutlass::bfloat16_t));
        }
        __syncthreads();
        if (threadIdx.x == 0)
            ptx::mbarrier_arrive_and_set_tx(
                load_barrier,
                valid_rows * kTile * sizeof(cutlass::bfloat16_t));
        __syncthreads();
        load_barrier->wait(load_phase);

        {
            const uint32_t warp = threadIdx.x >> 5;
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t token_half = lane >> 4;
            const uint32_t feature_slot = lane & 15u;
            const uint32_t group = warp >> 3;
            const uint32_t group_warp = warp & 7u;
            // Interleave the eight warps of a group over the two 64-column
            // halves.  This avoids the severe shared-bank concentration of
            // assigning adjacent lane pairs to adjacent features.
            const uint32_t feature =
                (group_warp >> 2) * 64u + 4u * feature_slot +
                (group_warp & 3u);
            const uint32_t first_token =
                group * 32u + token_half * 16u;
            constexpr uint32_t kWarpMask = 0xffffffffu;

            // Direct mode resolves the expert once per warp, then broadcasts
            // the final byte address metadata to its 16 feature owners.  The
            // guard excludes the two synthetic groups of a final 64-row tile.
            const uint32_t local_group_k = row_begin + group * 32u;
            const uint32_t global_group_k =
                value_row_begin + local_group_k;
            const bool valid_scale_group = local_group_k < total_k;
            uint32_t packed_scale_row = 0u;
            uint32_t packed_scale_control = 0u;
            if constexpr (kDirectPackedScales) {
                if (lane == 0u && valid_scale_group) {
                    uint32_t expert = producer_expert;
                    if constexpr (!kExpertLocal) {
                        expert = k3_mxfp8_find_prefix_owner<
                            kNumExperts, kMaxRanges>(
                                value_prefix, global_group_k);
                    }
                    const uint32_t local_group =
                        (global_group_k - value_prefix[expert]) / 32u;
                    const uint32_t num_expert_groups =
                        (value_prefix[expert + 1u] -
                         value_prefix[expert]) / 32u;
                    // ScalePrefix immediately follows ValuePrefix in the
                    // immutable K3 metadata ABI.  Derive it here instead of
                    // extending the hot producer call with one more pointer.
                    const auto* scale_prefix =
                        value_prefix + kNumExperts + 1u;
                    packed_scale_row =
                        scale_prefix[expert] + local_group / 4u;
                    packed_scale_control = (local_group & 3u) |
                        static_cast<uint32_t>(
                            local_group + 1u == num_expert_groups) << 2u;
                }
                packed_scale_row = __shfl_sync(
                    kWarpMask, packed_scale_row, 0);
                packed_scale_control = __shfl_sync(
                    kWarpMask, packed_scale_control, 0);
            }
            const uint32_t packed_scale_byte =
                packed_scale_control & 3u;
            const bool final_expert_group =
                (packed_scale_control & 4u) != 0u;

            float primary_amax = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 16; ++i) {
                const float value = static_cast<float>(
                    source_smem[(first_token + i) * kTile + feature]);
                primary_amax = cute::max(primary_amax, cute::abs(value));
            }
            primary_amax = cute::max(
                primary_amax,
                __shfl_xor_sync(kWarpMask, primary_amax, 16));
            float primary_scale = 1.0f;
            float primary_scale_inv = 1.0f;
            uint32_t primary_scale_byte = 0x7fu;
            if (token_half == 0u)
                k3_mxfp8_scale_pair(
                    primary_amax, primary_scale, primary_scale_inv,
                    primary_scale_byte);
            primary_scale = __shfl_sync(
                kWarpMask, primary_scale, feature_slot);
            primary_scale_inv = __shfl_sync(
                kWarpMask, primary_scale_inv, feature_slot);
            if (token_half == 0u) {
                if constexpr (kDirectPackedScales) {
                    if (valid_scale_group) {
                        const uint32_t packed_scale_feature =
                            k3_mxfp8_utccp_scale_feature(
                                feature_begin + feature);
                        const uint64_t packed_scale_offset =
                            (static_cast<uint64_t>(packed_scale_row) *
                                 source_width + packed_scale_feature) *
                            sizeof(uint32_t);
                        primary_scale_output[
                            packed_scale_offset + packed_scale_byte] =
                                static_cast<uint8_t>(primary_scale_byte);
                        if (final_expert_group) {
                            for (uint32_t byte = packed_scale_byte + 1u;
                                 byte < 4u; ++byte) {
                                primary_scale_output[
                                    packed_scale_offset + byte] = 0x7fu;
                            }
                        }
                    }
                } else {
                    const uint64_t raw_scale_offset =
                        static_cast<uint64_t>(
                            (value_row_begin + row_begin) / 32u + group) *
                            source_width + feature_begin + feature;
                    primary_scale_output[raw_scale_offset] =
                        static_cast<uint8_t>(primary_scale_byte);
                }
            }

            float residual_amax = 0.0f;
            #pragma unroll 1
            for (uint32_t chunk = 0; chunk < 4; ++chunk) {
                const uint32_t token = first_token + chunk * 4u;
                float values[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i)
                    values[i] = static_cast<float>(
                        source_smem[(token + i) * kTile + feature]);
                const auto primary = __nv_fp8x4_e4m3(make_float4(
                    values[0] * primary_scale_inv,
                    values[1] * primary_scale_inv,
                    values[2] * primary_scale_inv,
                    values[3] * primary_scale_inv));
                const float4 primary_float = static_cast<float4>(primary);
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    const float primary_value =
                        i == 0 ? primary_float.x :
                        i == 1 ? primary_float.y :
                        i == 2 ? primary_float.z : primary_float.w;
                    const cutlass::bfloat16_t rounded_residual(
                        values[i] - primary_value * primary_scale);
                    source_smem[(token + i) * kTile + feature] =
                        rounded_residual;
                    residual_amax = cute::max(
                        residual_amax,
                        cute::abs(static_cast<float>(rounded_residual)));
                }
                const uint32_t primary_bits = primary.__x;
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    primary_smem[k3_mxfp8_swizzle_128b_offset(
                        token + i, feature)] =
                        static_cast<uint8_t>(primary_bits >> (i * 8u));
                }
            }

            residual_amax = cute::max(
                residual_amax,
                __shfl_xor_sync(kWarpMask, residual_amax, 16));
            float residual_scale = 1.0f;
            float residual_scale_inv = 1.0f;
            uint32_t residual_scale_byte = 0x7fu;
            if (token_half == 0u)
                k3_mxfp8_scale_pair(
                    residual_amax, residual_scale, residual_scale_inv,
                    residual_scale_byte);
            residual_scale_inv = __shfl_sync(
                kWarpMask, residual_scale_inv, feature_slot);
            #pragma unroll 1
            for (uint32_t chunk = 0; chunk < 4; ++chunk) {
                const uint32_t token = first_token + chunk * 4u;
                float residual_values[4];
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i)
                    residual_values[i] = static_cast<float>(
                        source_smem[(token + i) * kTile + feature]);
                const auto residual = __nv_fp8x4_e4m3(make_float4(
                    residual_values[0] * residual_scale_inv,
                    residual_values[1] * residual_scale_inv,
                    residual_values[2] * residual_scale_inv,
                    residual_values[3] * residual_scale_inv));
                const uint32_t residual_bits = residual.__x;
                #pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    residual_smem[k3_mxfp8_swizzle_128b_offset(
                        token + i, feature)] =
                        static_cast<uint8_t>(residual_bits >> (i * 8u));
                }
            }

            if (token_half == 0u) {
                if constexpr (kDirectPackedScales) {
                    if (valid_scale_group) {
                        const uint32_t packed_scale_feature =
                            k3_mxfp8_utccp_scale_feature(
                                feature_begin + feature);
                        const uint64_t packed_scale_offset =
                            (static_cast<uint64_t>(packed_scale_row) *
                                 source_width + packed_scale_feature) *
                            sizeof(uint32_t);
                        residual_scale_output[
                            packed_scale_offset + packed_scale_byte] =
                                static_cast<uint8_t>(residual_scale_byte);
                        if (final_expert_group) {
                            for (uint32_t byte = packed_scale_byte + 1u;
                                 byte < 4u; ++byte) {
                                residual_scale_output[
                                    packed_scale_offset + byte] = 0x7fu;
                            }
                        }
                    }
                } else {
                    const uint64_t raw_scale_offset =
                        static_cast<uint64_t>(
                            (value_row_begin + row_begin) / 32u + group) *
                            source_width + feature_begin + feature;
                    residual_scale_output[raw_scale_offset] =
                        static_cast<uint8_t>(residual_scale_byte);
                }
            }
        }
        __syncthreads();

        if (valid_rows == kTile) {
            if (threadIdx.x == 0) {
                cute::tma_store_fence();
                cute::SM90_TMA_STORE_2D::copy(
                    &primary_map, primary_smem, feature_begin,
                    value_row_begin + row_begin);
                cute::SM90_TMA_STORE_2D::copy(
                    &residual_map, residual_smem, feature_begin,
                    value_row_begin + row_begin);
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            }
        } else {
            // A TMA store always transfers the descriptor's complete 128-row
            // box.  The parent intentionally reuses two exact-size FP8
            // planes inside one BF16 arena, so a pool capacity that is 64 mod
            // 128 has no padding row to absorb that final box.  Scatter only
            // the valid logical rows for this one rare tail tile.  Ordinary
            // stores are published by the producer-wide proxy fence and grid
            // barrier before grouped wgrad begins.
            const uint32_t num_tail_values = valid_rows * kTile;
            for (uint32_t idx = threadIdx.x; idx < num_tail_values;
                 idx += kNumThreads) {
                const uint32_t row = idx / kTile;
                const uint32_t feature = idx - row * kTile;
                const uint64_t dst =
                    static_cast<uint64_t>(
                        value_row_begin + row_begin + row) *
                        primary_destination_row_stride +
                    feature_begin + feature;
                const uint64_t residual_dst =
                    static_cast<uint64_t>(
                        value_row_begin + row_begin + row) *
                        residual_destination_row_stride +
                    feature_begin + feature;
                const uint32_t src =
                    k3_mxfp8_swizzle_128b_offset(row, feature);
                primary_values[dst] = primary_smem[src];
                residual_values[residual_dst] = residual_smem[src];
            }
        }
        __syncthreads();
        if constexpr (kFeatureMajor) {
            if (row_tile + 1u == num_row_tiles) {
                // Every scale writer publishes its own generic stores, while
                // the async-proxy fence covers the completed TMA value stores.
                // Only after the CTA-wide join may one lane release-OR the
                // panel bit consumed by the dynamic scheduler.
                asm volatile("fence.proxy.async.global;" ::: "memory");
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0u) {
                    const uint32_t panel =
                        feature_panel_base + feature_tile;
                    ptx::red_or_rel_gpu(
                        feature_ready_masks + panel / 32u,
                        1u << (panel % 32u));
                }
                __syncthreads();
            }
        }
        load_phase ^= 1;
    }

    if (threadIdx.x == 0) {
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(load_barrier));
    }
    __syncthreads();
}

/** Quantize dW13 operands on the exact body's otherwise-idle suffix warps.
 *
 * Warps 12..31 in every persistent CTA cooperatively advance one active
 * expert at a time.  A warp owns sixteen feature columns and reproduces the
 * serial producer's two-lanes-per-feature group-32 reduction, including the
 * BF16 rounding between the primary and residual FP8 products.  Four group
 * scales are packed directly into the descriptor-ready uint32 scale plane,
 * so this path neither touches the temporary raw-scale rows nor borrows the
 * grouped body's shared-memory/TMEM pipelines.
 *
 * Each writer publishes its generic stores to the async proxy before a
 * CTA-local named-barrier join.  One lane per CTA then contributes an
 * acquire/release arrival; the final CTA adds the operand-ready credit to the
 * existing per-expert W13-dgrad counter.  The grouped provider may therefore
 * consume expert E while these warps quantize E+1 without another allocation
 * or a grid-wide phase barrier.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_quantize_dw13_operands_background(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const cutlass::bfloat16_t* source_b,
        const uint32_t k_capacity,
        uint8_t* a_primary_values,
        uint8_t* a_residual_values,
        uint8_t* b_primary_values,
        uint8_t* b_residual_values,
        uint32_t* a_primary_scales,
        uint32_t* a_residual_scales,
        uint32_t* b_primary_scales,
        uint32_t* b_residual_scales,
        uint32_t* state,
        const uint32_t warp_idx,
        const uint32_t lane_idx) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kFirstQuantWarp = 12u;
    constexpr uint32_t kQuantWarpsPerCTA = 20u;
    constexpr uint32_t kQuantThreadsPerCTA =
        kQuantWarpsPerCTA * 32u;
    // User barrier five maps to hardware barrier thirteen after CUTLASS's
    // reserved prefix.  The parent's initial-dequant phase has retired it
    // before exact wgrad starts, and this grouped body otherwise uses only
    // mbarriers; avoid raw barrier fifteen used by reverse dispatch.
    constexpr uint32_t kQuantNamedBarrier = 5u;
    constexpr uint32_t kFeatureChunk = 16u;
    constexpr uint32_t kGroupsPerPackedRow = 4u;
    constexpr uint32_t kRowsPerGroup = 32u;
    constexpr uint32_t kRowsPerLane = 16u;
    constexpr uint32_t kAWidth = 2u * kIntermediateHidden;
    constexpr uint32_t kBWidth = kHidden;
    static_assert(
        kNumThreads == 1024u && kNumSMs % 2u == 0u &&
            kAWidth % kFeatureChunk == 0u &&
            kBWidth % kFeatureChunk == 0u,
        "K3 dW13 background quantizer geometry changed");

    if (warp_idx < kFirstQuantWarp ||
        warp_idx >= kFirstQuantWarp + kQuantWarpsPerCTA)
        return;

    const uint32_t local_quant_warp = warp_idx - kFirstQuantWarp;
    const uint32_t global_quant_warp =
        static_cast<uint32_t>(blockIdx.x) * kQuantWarpsPerCTA +
        local_quant_warp;
    constexpr uint32_t kGlobalQuantWarps =
        kNumSMs * kQuantWarpsPerCTA;
    constexpr uint32_t kAFeatureChunks = kAWidth / kFeatureChunk;
    constexpr uint32_t kBFeatureChunks = kBWidth / kFeatureChunk;
    constexpr uint32_t kFeatureChunksPerScaleRow =
        kAFeatureChunks + kBFeatureChunks;
    const uint32_t token_half = lane_idx >> 4;
    const uint32_t feature_slot = lane_idx & 15u;
    constexpr uint32_t kWarpMask = 0xffffffffu;

    const auto* const value_prefix =
        state + Prefix::kValuePrefix;
    const auto* const scale_prefix =
        state + Prefix::kScalePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);

    #pragma unroll 1
    for (uint32_t active = 0u; active < active_count; ++active) {
        const uint32_t expert = state[Overlap::kActiveExperts + active];
        const uint32_t expert_k =
            value_prefix[expert + 1u] - value_prefix[expert];
        DG_DEVICE_ASSERT(expert_k % kRowsPerGroup == 0u);
        const uint32_t num_groups = expert_k / kRowsPerGroup;
        const uint32_t num_scale_rows =
            scale_prefix[expert + 1u] - scale_prefix[expert];
        const uint64_t num_tasks =
            static_cast<uint64_t>(num_scale_rows) *
            kFeatureChunksPerScaleRow;

        for (uint64_t task = global_quant_warp;
             task < num_tasks; task += kGlobalQuantWarps) {
            const uint32_t local_scale_row =
                task / kFeatureChunksPerScaleRow;
            const uint32_t feature_task =
                task - static_cast<uint64_t>(local_scale_row) *
                    kFeatureChunksPerScaleRow;
            const bool is_a = feature_task < kAFeatureChunks;
            const uint32_t feature_chunk = is_a
                ? feature_task
                : feature_task - kAFeatureChunks;
            const uint32_t width = is_a ? kAWidth : kBWidth;
            const uint32_t feature =
                feature_chunk * kFeatureChunk + feature_slot;
            const auto* const source = is_a ? source_a : source_b;
            auto* const primary_values =
                is_a ? a_primary_values : b_primary_values;
            auto* const residual_values =
                is_a ? a_residual_values : b_residual_values;
            auto* const primary_scales =
                is_a ? a_primary_scales : b_primary_scales;
            auto* const residual_scales =
                is_a ? a_residual_scales : b_residual_scales;
            uint32_t primary_scale_word = 0u;
            uint32_t residual_scale_word = 0u;

            #pragma unroll
            for (uint32_t group = 0u;
                 group < kGroupsPerPackedRow; ++group) {
                const uint32_t local_group =
                    local_scale_row * kGroupsPerPackedRow + group;
                const bool valid_group = local_group < num_groups;
                const uint32_t local_row_begin =
                    local_group * kRowsPerGroup +
                    token_half * kRowsPerLane;
                // Keep the sixteen source values in eight packed BF16
                // registers.  Reusing these registers for the BF16-rounded
                // residual avoids the two 16-float local arrays that spill
                // under the parent's 64-register grouped-body allocation.
                uint32_t value_pairs[kRowsPerLane / 2u];
                #pragma unroll
                for (uint32_t pair = 0u;
                     pair < kRowsPerLane / 2u; ++pair) {
                    uint32_t packed_values = 0u;
                    #pragma unroll
                    for (uint32_t element = 0u; element < 2u; ++element) {
                        const uint32_t row = pair * 2u + element;
                        uint32_t source_row = static_cast<uint32_t>(-1);
                        if (valid_group && feature_slot == 0u) {
                            source_row =
                                k3_mxfp8_expert_source_pool_row<
                                    kNumExperts, kBlockM,
                                    kK3MaxBackwardRanges>(
                                        expert, local_row_begin + row,
                                        expert_counts, backward_ranges,
                                        physical_range_prefix);
                        }
                        source_row = __shfl_sync(
                            kWarpMask, source_row, token_half * 16u);
                        if (valid_group) {
                            const auto* const source_value =
                                source +
                                static_cast<uint64_t>(source_row) * width +
                                feature;
                            const uint32_t source_bits =
                                *reinterpret_cast<const uint16_t*>(
                                    source_value);
                            packed_values |=
                                source_bits << (element * 16u);
                        }
                    }
                    value_pairs[pair] = packed_values;
                }

                float primary_amax = 0.0f;
                #pragma unroll
                for (uint32_t pair = 0u;
                     pair < kRowsPerLane / 2u; ++pair) {
                    const float2 values = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(
                            &value_pairs[pair]));
                    primary_amax = cute::max(
                        primary_amax, cute::abs(values.x));
                    primary_amax = cute::max(
                        primary_amax, cute::abs(values.y));
                }
                primary_amax = cute::max(
                    primary_amax,
                    __shfl_xor_sync(kWarpMask, primary_amax, 16));
                float primary_scale = 1.0f;
                float primary_scale_inv = 1.0f;
                uint32_t primary_scale_byte = 0x7fu;
                if (token_half == 0u)
                    k3_mxfp8_scale_pair(
                        primary_amax, primary_scale,
                        primary_scale_inv, primary_scale_byte);
                primary_scale = __shfl_sync(
                    kWarpMask, primary_scale, feature_slot);
                primary_scale_inv = __shfl_sync(
                    kWarpMask, primary_scale_inv, feature_slot);
                primary_scale_byte = __shfl_sync(
                    kWarpMask, primary_scale_byte, feature_slot);

                float residual_amax = 0.0f;
                #pragma unroll 1
                for (uint32_t chunk = 0u; chunk < 4u; ++chunk) {
                    const uint32_t row = chunk * 4u;
                    const float2 values01 = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(
                            &value_pairs[chunk * 2u]));
                    const float2 values23 = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(
                            &value_pairs[chunk * 2u + 1u]));
                    const auto primary = __nv_fp8x4_e4m3(make_float4(
                        values01.x * primary_scale_inv,
                        values01.y * primary_scale_inv,
                        values23.x * primary_scale_inv,
                        values23.y * primary_scale_inv));
                    const float4 primary_float =
                        static_cast<float4>(primary);
                    const __nv_bfloat162 residual01 =
                        __float22bfloat162_rn(make_float2(
                            values01.x - primary_float.x * primary_scale,
                            values01.y - primary_float.y * primary_scale));
                    const __nv_bfloat162 residual23 =
                        __float22bfloat162_rn(make_float2(
                            values23.x - primary_float.z * primary_scale,
                            values23.y - primary_float.w * primary_scale));
                    const float2 rounded_residual01 =
                        __bfloat1622float2(residual01);
                    const float2 rounded_residual23 =
                        __bfloat1622float2(residual23);
                    residual_amax = cute::max(
                        residual_amax, cute::abs(rounded_residual01.x));
                    residual_amax = cute::max(
                        residual_amax, cute::abs(rounded_residual01.y));
                    residual_amax = cute::max(
                        residual_amax, cute::abs(rounded_residual23.x));
                    residual_amax = cute::max(
                        residual_amax, cute::abs(rounded_residual23.y));
                    value_pairs[chunk * 2u] =
                        *reinterpret_cast<const uint32_t*>(&residual01);
                    value_pairs[chunk * 2u + 1u] =
                        *reinterpret_cast<const uint32_t*>(&residual23);
                    if (valid_group) {
                        const uint32_t primary_bits = primary.__x;
                        #pragma unroll
                        for (uint32_t element = 0u;
                             element < 4u; ++element) {
                            const uint32_t local_row =
                                local_row_begin + row + element;
                            primary_values[
                                static_cast<uint64_t>(
                                    value_prefix[expert] + local_row) *
                                    width + feature] =
                                static_cast<uint8_t>(
                                    primary_bits >> (element * 8u));
                        }
                    }
                }

                residual_amax = cute::max(
                    residual_amax,
                    __shfl_xor_sync(kWarpMask, residual_amax, 16));
                float residual_scale = 1.0f;
                float residual_scale_inv = 1.0f;
                uint32_t residual_scale_byte = 0x7fu;
                if (token_half == 0u)
                    k3_mxfp8_scale_pair(
                        residual_amax, residual_scale,
                        residual_scale_inv, residual_scale_byte);
                residual_scale_inv = __shfl_sync(
                    kWarpMask, residual_scale_inv, feature_slot);
                residual_scale_byte = __shfl_sync(
                    kWarpMask, residual_scale_byte, feature_slot);

                #pragma unroll 1
                for (uint32_t chunk = 0u; chunk < 4u; ++chunk) {
                    const uint32_t row = chunk * 4u;
                    const float2 residual01 = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(
                            &value_pairs[chunk * 2u]));
                    const float2 residual23 = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(
                            &value_pairs[chunk * 2u + 1u]));
                    const auto residual_quantized =
                        __nv_fp8x4_e4m3(make_float4(
                            residual01.x * residual_scale_inv,
                            residual01.y * residual_scale_inv,
                            residual23.x * residual_scale_inv,
                            residual23.y * residual_scale_inv));
                    if (valid_group) {
                        const uint32_t residual_bits =
                            residual_quantized.__x;
                        #pragma unroll
                        for (uint32_t element = 0u;
                             element < 4u; ++element) {
                            const uint32_t local_row =
                                local_row_begin + row + element;
                            residual_values[
                                static_cast<uint64_t>(
                                    value_prefix[expert] + local_row) *
                                    width + feature] =
                                static_cast<uint8_t>(
                                    residual_bits >> (element * 8u));
                        }
                    }
                }

                if (token_half == 0u) {
                    primary_scale_word |=
                        (primary_scale_byte & 0xffu) << (group * 8u);
                    residual_scale_word |=
                        (residual_scale_byte & 0xffu) << (group * 8u);
                }
            }

            if (token_half == 0u) {
                const uint32_t global_scale_row =
                    scale_prefix[expert] + local_scale_row;
                const uint32_t packed_scale_feature =
                    k3_mxfp8_utccp_scale_feature(feature);
                const uint64_t scale_offset =
                    static_cast<uint64_t>(global_scale_row) * width +
                    packed_scale_feature;
                primary_scales[scale_offset] = primary_scale_word;
                residual_scales[scale_offset] = residual_scale_word;
            }
        }

        // Every writer must publish its own generic stores before the async
        // TMA proxy can observe them; a fence in only the elected lane is not
        // sufficient.  The named barrier is private to warps 12..31.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        cutlass::arch::NamedBarrier::sync(
            kQuantThreadsPerCTA, kQuantNamedBarrier);
        if (local_quant_warp == 0u && lane_idx == 0u) {
            const uint32_t arrival = ptx::atomic_add_acq_rel(
                state + Overlap::kDW13QuantDone + expert, 1u);
            DG_DEVICE_ASSERT(arrival < kNumSMs);
            if (arrival + 1u == kNumSMs) {
                const uint32_t previous = ptx::atomic_add_acq_rel(
                    state + Overlap::kDW13CompositeReady + expert, 1u);
                const uint32_t pool_blocks =
                    state[Overlap::kPoolBlockPrefix + expert + 1u] -
                    state[Overlap::kPoolBlockPrefix + expert];
                DG_DEVICE_ASSERT(previous <= pool_blocks * 14u);
            }
        }
    }
}

/** Stream exact dW2 operands through the CTA-local background engines.
 *
 * Warps 12--27 are four independent 128-thread engines. A task owns one
 * expert, one complete 128-feature panel, and every packed group-32 scale row
 * for that panel. One warp bulk-loads the 32 contiguous BF16 rows in parallel
 * into shared memory, all 128 threads perform the exact primary E4M3 plus
 * BF16-rounded residual E4M3 decomposition, and bulk stores publish the two
 * row-major FP8 planes and packed UE8M0 scale words.
 *
 * dW2-B reuses the range-major BF16 rows read by dW2-A. A tasks are claimed
 * first. Completion of all 28 A panels release-publishes one bit per expert;
 * each B expert carries an immutable bit mask of the physical source experts
 * its logical rows alias. A B task is CAS-claimed only after acquire-loading
 * every required A-complete bit. This is a dependency edge, not a global
 * barrier: unrelated B experts can start while later A experts are still in
 * flight. Each value/scale panel release-ORs the existing scheduler readiness
 * mask, allowing the three-product UMMA consumer to start from its first
 * complete four-task quantum.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_DEVICE __noinline__ void
k3_mxfp8_stream_dw2_operands_background(
        const int* expert_counts,
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t k_capacity,
        const cutlass::bfloat16_t* source_a,
        const cutlass::bfloat16_t* source_b,
        const cutlass::bfloat16_t* x_pool_output,
        const cutlass::bfloat16_t* grad_y_unweighted_output,
        const cutlass::bfloat16_t* down_unweighted_output,
        cutlass::bfloat16_t* grad_ye_output,
        const cutlass::float_e4m3_t* scale_arena_source,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t warp_idx,
        const uint32_t lane_idx) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kFirstQuantWarp = 12u;
    constexpr uint32_t kQuantWarps =
        kK3MxFp8DW13QuantNumEngines *
        kK3MxFp8DW13QuantWarpsPerEngine;
    constexpr uint32_t kThreadsPerEngine =
        kK3MxFp8DW13QuantWarpsPerEngine * 32u;
    constexpr uint32_t kAWidth = kHidden;
    constexpr uint32_t kBWidth = kIntermediateHidden;
    constexpr uint32_t kAFeatureTiles =
        kAWidth / kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kBFeatureTiles =
        kBWidth / kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kFeatureTiles =
        kAFeatureTiles + kBFeatureTiles;
    constexpr uint32_t kSourceElements =
        kK3MxFp8DW13QuantRowsPerGroup *
        kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kSourceBytes =
        kSourceElements * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kValueBytes =
        kSourceElements * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kSourceRowBytes =
        kK3MxFp8DW13QuantFeatureTile * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kValueRowBytes =
        kK3MxFp8DW13QuantFeatureTile *
        sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kScaleBytes =
        kK3MxFp8DW13QuantFeatureTile * sizeof(uint32_t);
    constexpr uint32_t kNoTask = 0xffffffffu;
    constexpr uint32_t kStop = 0xfffffffeu;
    static_assert(
        kNumThreads == 1024u && kNumSMs % 2u == 0u &&
            kAFeatureTiles == 28u && kBFeatureTiles == 24u &&
            kFeatureTiles == Overlap::kDW2FeaturePanelsPerExpert &&
            kBlockM % kK3MxFp8DW13QuantRowsPerGroup == 0u,
        "K3 streaming dW2 quantizer geometry changed");

    if (warp_idx < kFirstQuantWarp ||
        warp_idx >= kFirstQuantWarp + kQuantWarps)
        return;

    const uint32_t quant_warp = warp_idx - kFirstQuantWarp;
    const uint32_t engine =
        quant_warp / kK3MxFp8DW13QuantWarpsPerEngine;
    const uint32_t engine_thread =
        (quant_warp % kK3MxFp8DW13QuantWarpsPerEngine) * 32u + lane_idx;
    const uint32_t named_barrier =
        kK3MxFp8DW13QuantFirstUserNamedBarrier + engine;
    auto engine_sync = [=]() {
        cutlass::arch::NamedBarrier::sync(
            kThreadsPerEngine, named_barrier);
    };

    uint8_t* const engine_smem =
        smem_buffer + kK3MxFp8DW13QuantScratchBegin +
        engine * kK3MxFp8DW13QuantEngineStride;
    auto* const source_smem =
        reinterpret_cast<cutlass::bfloat16_t*>(engine_smem);
    auto* const primary_smem = engine_smem + kSourceBytes;
    auto* const residual_smem = primary_smem + kValueBytes;
    auto* const primary_scale_smem = reinterpret_cast<uint32_t*>(
        residual_smem + kValueBytes);
    auto* const residual_scale_smem =
        primary_scale_smem + kK3MxFp8DW13QuantFeatureTile;
    auto* const load_barrier = reinterpret_cast<Barrier*>(
        residual_scale_smem + kK3MxFp8DW13QuantFeatureTile);
    auto* const control = reinterpret_cast<volatile uint32_t*>(
        load_barrier + 1);
    static_assert(
        kSourceBytes + 2u * kValueBytes + 2u * kScaleBytes +
            sizeof(Barrier) + 4u * sizeof(uint32_t) ==
        kK3MxFp8DW13QuantEnginePayloadBytes);

    const auto scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kAWidth, kBWidth, k_capacity, kBlockM);
    const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden, k_capacity, kBlockM);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            scale_layout, dw13_scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    K3MxFp8WgradScaleStorage scales[2];
    k3_mxfp8_bind_wgrad_scale_storage(
        scale_layout, scale_arena, scale_arena + scale_layout.raw_bytes,
        scales[0], scales[1]);

    auto* const safe_h_arena = reinterpret_cast<uint8_t*>(
        x_pool_output == grad_y_unweighted_output
        ? const_cast<cutlass::bfloat16_t*>(down_unweighted_output)
        : const_cast<cutlass::bfloat16_t*>(grad_y_unweighted_output));
    auto* const retired_grad_ye_arena =
        reinterpret_cast<uint8_t*>(grad_ye_output);
    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const scale_prefix = state + Prefix::kScalePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);

    if (engine_thread == 0u) {
        load_barrier->init(1u);
        cutlass::arch::fence_barrier_init();
    }
    engine_sync();

    uint32_t load_phase = 0u;
    uint32_t search_begin = active_count == 0u
        ? 0u
        : (static_cast<uint32_t>(blockIdx.x) *
               kK3MxFp8DW13QuantNumEngines + engine) % active_count;
    while (true) {
        if (engine_thread == 0u) {
            control[0] = kNoTask;
            if (ptx::ld_acq(
                    state + Overlap::kDW2SuffixQuantExpertsDone) >= active_count) {
                control[0] = kStop;
            } else {
                for (uint32_t probe = 0u;
                     probe < active_count; ++probe) {
                    const uint32_t active =
                        (search_begin + probe) % active_count;
                    const uint32_t expert =
                        state[Overlap::kActiveExperts + active];
                    if (ptx::ld_acq(
                            state + Overlap::kDW2SuffixQuantDone + expert) >=
                        kFeatureTiles) {
                        continue;
                    }
                    auto* const cursor =
                        state + Overlap::kDW2SuffixQuantCursor + expert;
                    uint32_t task = ptx::ld_acq(cursor);
                    bool claimed = false;
                    while (task < kFeatureTiles) {
                        if (task >= kAFeatureTiles) {
                            const auto* const dependency =
                                state + Overlap::kDW2SuffixBAliasDependencies +
                                expert * Overlap::kDW2SuffixExpertMaskWords;
                            bool aliases_retired = true;
                            #pragma unroll
                            for (uint32_t word = 0u;
                                 word < Overlap::kDW2SuffixExpertMaskWords;
                                 ++word) {
                                const uint32_t required = dependency[word];
                                const uint32_t complete = ptx::ld_acq(
                                    state + Overlap::kDW2SuffixACompleteMasks +
                                    word);
                                aliases_retired &=
                                    (complete & required) == required;
                            }
                            if (!aliases_retired)
                                break;
                        }
                        const uint32_t observed = atomicCAS(
                            cursor, task, task + 1u);
                        if (observed == task) {
                            claimed = true;
                            break;
                        }
                        task = observed;
                    }
                    if (!claimed)
                        continue;
                    control[0] = expert;
                    control[1] = task;
                    control[2] =
                        scale_prefix[expert + 1u] - scale_prefix[expert];
                    // Complete one expert early so its first UMMA quanta can
                    // run while other experts are still being quantized.
                    search_begin = active;
                    break;
                }
            }
        }
        engine_sync();

        const uint32_t expert = control[0];
        if (expert == kStop)
            break;
        if (expert == kNoTask) {
            if (engine_thread == 0u) {
                search_begin = active_count == 0u
                    ? 0u : (search_begin + 1u) % active_count;
                __nanosleep(64);
            }
            engine_sync();
            continue;
        }

        const uint32_t feature_task = control[1];
        const uint32_t scale_rows = control[2];
        const bool is_a = feature_task < kAFeatureTiles;
        const uint32_t feature_tile = is_a
            ? feature_task : feature_task - kAFeatureTiles;
        const uint32_t width = is_a ? kAWidth : kBWidth;
        const uint32_t feature_begin =
            feature_tile * kK3MxFp8DW13QuantFeatureTile;
        const uint32_t expert_k =
            value_prefix[expert + 1u] - value_prefix[expert];
        const uint32_t num_groups =
            expert_k / kK3MxFp8DW13QuantRowsPerGroup;
        DG_DEVICE_ASSERT(
            scale_rows != 0u && feature_task < kFeatureTiles &&
            expert_k % kK3MxFp8DW13QuantRowsPerGroup == 0u);

        const auto* const source = is_a ? source_a : source_b;
        auto* const primary_values = is_a
            ? safe_h_arena : retired_grad_ye_arena;
        auto* const residual_values = is_a
            ? safe_h_arena + static_cast<uint64_t>(k_capacity) * kHidden
            : retired_grad_ye_arena + kIntermediateHidden;
        const uint32_t destination_row_stride =
            is_a ? kHidden : 2u * kHidden;
        auto* const primary_scales = is_a
            ? scales[0].primary_packed : scales[1].primary_packed;
        auto* const residual_scales = is_a
            ? scales[0].residual_packed : scales[1].residual_packed;

        for (uint32_t local_scale_row = 0u;
             local_scale_row < scale_rows; ++local_scale_row) {
            uint32_t primary_scale_word = 0x7f7f7f7fu;
            uint32_t residual_scale_word = 0x7f7f7f7fu;
            bool value_store_pending = false;

            #pragma unroll
            for (uint32_t group_in_scale_row = 0u;
                 group_in_scale_row < 4u; ++group_in_scale_row) {
                const uint32_t local_group =
                    local_scale_row * 4u + group_in_scale_row;
                if (local_group >= num_groups)
                    continue;
                const uint32_t local_row =
                    local_group * kK3MxFp8DW13QuantRowsPerGroup;
                const uint32_t source_row =
                    k3_mxfp8_expert_source_pool_row<
                        kNumExperts, kBlockM, kK3MaxBackwardRanges>(
                            expert, local_row, expert_counts,
                            *backward_ranges, physical_range_prefix);
                DG_DEVICE_ASSERT(
                    source_row + kK3MxFp8DW13QuantRowsPerGroup <=
                    k_capacity);

                // Do not issue all 32 bulk copies from one elected lane.  A
                // lane-local bulk queue can back-pressure before the elected
                // lane reaches arrive.expect_tx, leaving every engine thread
                // parked on this generation forever.  The proven producer
                // contract is one independent issuer per row, followed by an
                // engine-wide join before one arrival publishes the aggregate
                // transaction byte count.
                if (engine_thread <
                    kK3MxFp8DW13QuantRowsPerGroup) {
                    const uint32_t row = engine_thread;
                    ptx::tma_load_1d(
                        source_smem +
                            row * kK3MxFp8DW13QuantFeatureTile,
                        source +
                            static_cast<uint64_t>(source_row + row) * width +
                            feature_begin,
                        load_barrier,
                        kSourceRowBytes);
                }
                engine_sync();
                if (engine_thread == 0u) {
                    ptx::mbarrier_arrive_and_set_tx(
                        load_barrier, kSourceBytes);
                }
                load_barrier->wait(load_phase);
                load_phase ^= 1u;

                if (value_store_pending && engine_thread == 0u)
                    ptx::tma_store_wait<0>();
                engine_sync();
                value_store_pending = false;

                float primary_amax = 0.0f;
                #pragma unroll
                for (uint32_t row = 0u;
                     row < kK3MxFp8DW13QuantRowsPerGroup; ++row) {
                    const float value = static_cast<float>(
                        source_smem[
                            row * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread]);
                    primary_amax = cute::max(
                        primary_amax, cute::abs(value));
                }
                float primary_scale = 1.0f;
                float primary_scale_inv = 1.0f;
                uint32_t primary_scale_byte = 0x7fu;
                k3_mxfp8_scale_pair(
                    primary_amax, primary_scale, primary_scale_inv,
                    primary_scale_byte);

                float residual_amax = 0.0f;
                #pragma unroll 1
                for (uint32_t row = 0u;
                     row < kK3MxFp8DW13QuantRowsPerGroup; row += 4u) {
                    float values[4];
                    #pragma unroll
                    for (uint32_t i = 0u; i < 4u; ++i) {
                        values[i] = static_cast<float>(source_smem[
                            (row + i) * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread]);
                    }
                    const auto primary = __nv_fp8x4_e4m3(make_float4(
                        values[0] * primary_scale_inv,
                        values[1] * primary_scale_inv,
                        values[2] * primary_scale_inv,
                        values[3] * primary_scale_inv));
                    const float4 primary_float =
                        static_cast<float4>(primary);
                    const uint32_t primary_bits = primary.__x;
                    #pragma unroll
                    for (uint32_t i = 0u; i < 4u; ++i) {
                        const float primary_value = i == 0u
                            ? primary_float.x : i == 1u
                            ? primary_float.y : i == 2u
                            ? primary_float.z : primary_float.w;
                        const cutlass::bfloat16_t rounded_residual(
                            values[i] - primary_value * primary_scale);
                        source_smem[
                            (row + i) * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread] = rounded_residual;
                        primary_smem[
                            (row + i) * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread] = static_cast<uint8_t>(
                                primary_bits >> (i * 8u));
                        residual_amax = cute::max(
                            residual_amax,
                            cute::abs(
                                static_cast<float>(rounded_residual)));
                    }
                }

                float residual_scale = 1.0f;
                float residual_scale_inv = 1.0f;
                uint32_t residual_scale_byte = 0x7fu;
                k3_mxfp8_scale_pair(
                    residual_amax, residual_scale, residual_scale_inv,
                    residual_scale_byte);
                #pragma unroll 1
                for (uint32_t row = 0u;
                     row < kK3MxFp8DW13QuantRowsPerGroup; row += 4u) {
                    float residual_input[4];
                    #pragma unroll
                    for (uint32_t i = 0u; i < 4u; ++i) {
                        residual_input[i] = static_cast<float>(source_smem[
                            (row + i) * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread]);
                    }
                    const auto residual = __nv_fp8x4_e4m3(make_float4(
                        residual_input[0] * residual_scale_inv,
                        residual_input[1] * residual_scale_inv,
                        residual_input[2] * residual_scale_inv,
                        residual_input[3] * residual_scale_inv));
                    const uint32_t residual_bits = residual.__x;
                    #pragma unroll
                    for (uint32_t i = 0u; i < 4u; ++i) {
                        residual_smem[
                            (row + i) * kK3MxFp8DW13QuantFeatureTile +
                            engine_thread] = static_cast<uint8_t>(
                                residual_bits >> (i * 8u));
                    }
                }

                const uint32_t shift = group_in_scale_row * 8u;
                primary_scale_word =
                    (primary_scale_word & ~(0xffu << shift)) |
                    ((primary_scale_byte & 0xffu) << shift);
                residual_scale_word =
                    (residual_scale_word & ~(0xffu << shift)) |
                    ((residual_scale_byte & 0xffu) << shift);
                cute::tma_store_fence();
                engine_sync();
                if (engine_thread == 0u) {
                    const uint32_t destination_row =
                        value_prefix[expert] + local_row;
                    #pragma unroll
                    for (uint32_t row_group = 0u;
                         row_group < 4u; ++row_group) {
                        #pragma unroll
                        for (uint32_t row_in_group = 0u;
                             row_in_group < 8u; ++row_in_group) {
                            const uint32_t row =
                                row_group * 8u + row_in_group;
                            const uint64_t destination_offset =
                                static_cast<uint64_t>(
                                    destination_row + row) *
                                    destination_row_stride +
                                feature_begin;
                            ptx::tma_store_1d(
                                primary_values + destination_offset,
                                primary_smem +
                                    row * kK3MxFp8DW13QuantFeatureTile,
                                kValueRowBytes);
                            ptx::tma_store_1d(
                                residual_values + destination_offset,
                                residual_smem +
                                    row * kK3MxFp8DW13QuantFeatureTile,
                                kValueRowBytes);
                        }
                        // Bound each committed bulk group to sixteen row
                        // transfers while leaving at most four groups in flight.
                        cute::tma_store_arrive();
                    }
                }
                engine_sync();
                value_store_pending = true;
            }

            const uint32_t native_scale_feature =
                (engine_thread % 32u) * 4u + engine_thread / 32u;
            primary_scale_smem[native_scale_feature] = primary_scale_word;
            residual_scale_smem[native_scale_feature] = residual_scale_word;
            cute::tma_store_fence();
            engine_sync();
            if (engine_thread == 0u) {
                // Four value groups are outstanding after the final token
                // group. Drain them before committing the packed-scale group,
                // keeping the architectural bulk-group window bounded.
                if (value_store_pending)
                    ptx::tma_store_wait<0>();
                const uint32_t global_scale_row =
                    scale_prefix[expert] + local_scale_row;
                const uint64_t scale_offset =
                    static_cast<uint64_t>(global_scale_row) * width +
                    feature_begin;
                ptx::tma_store_1d(
                    primary_scales + scale_offset,
                    primary_scale_smem, kScaleBytes);
                ptx::tma_store_1d(
                    residual_scales + scale_offset,
                    residual_scale_smem, kScaleBytes);
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();
            }
            engine_sync();
        }

        if (engine_thread == 0u) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
            ptx::red_or_rel_gpu(
                state + Overlap::kDW2FeatureReadyMasks +
                    expert * Overlap::kDW2FeatureReadyWordsPerExpert +
                    feature_task / 32u,
                1u << (feature_task % 32u));
            const uint32_t completed_panels = ptx::atomic_add_acq_rel(
                state + Overlap::kDW2SuffixQuantDone + expert, 1u);
            DG_DEVICE_ASSERT(completed_panels < kFeatureTiles);
            if (feature_task < kAFeatureTiles) {
                const uint32_t completed_a = ptx::atomic_add_acq_rel(
                    state + Overlap::kDW2SuffixAQuantDone + expert, 1u);
                DG_DEVICE_ASSERT(completed_a < kAFeatureTiles);
                if (completed_a + 1u == kAFeatureTiles) {
                    ptx::red_or_rel_gpu(
                        state + Overlap::kDW2SuffixACompleteMasks + expert / 32u,
                        1u << (expert % 32u));
                }
            }
            if (completed_panels + 1u == kFeatureTiles) {
                const uint32_t experts_done = ptx::atomic_add_acq_rel(
                    state + Overlap::kDW2SuffixQuantExpertsDone, 1u);
                DG_DEVICE_ASSERT(experts_done < active_count);
            }
        }
        engine_sync();
    }

    if (engine_thread == 0u) {
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(load_barrier));
    }
    engine_sync();
}

/** Test whether dW13-A may recycle both BF16-backed destinations.
 *
 * dW13-A stores its primary FP8 operand into range-major grad-ye and its
 * residual FP8 operand into range-major h-weighted. Those arenas remain the
 * BF16 sources of dW2-A and dW2-B, respectively. dW13 destinations use compact
 * expert-major rows, so the overwritten rows can belong to a different source
 * expert or range. For one pair of 128-byte destination panels, locate the
 * first intersecting owner with an upper-bound search over each range's
 * monotonic prefix, then acquire both corresponding 128-BF16-feature dW2-A/B
 * publication bits. The caller bridges these generic-proxy acquires into the
 * async proxy before issuing either overwriting TMA store.
 */
template <uint32_t kHidden, uint32_t kNumExperts,
          uint32_t kNumSMs>
CUTLASS_DEVICE __noinline__ bool
k3_mxfp8_dw2_source_panels_ready_for_dw13_a(
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t* state,
        const uint32_t destination_expert,
        const uint32_t source_panel) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kDW2AFeaturePanels = kHidden / 128u;
    DG_DEVICE_ASSERT(source_panel < 24u);

    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t destination_begin =
        value_prefix[destination_expert];
    const uint32_t destination_end =
        value_prefix[destination_expert + 1u];
    if (destination_begin == destination_end)
        return true;
    const uint32_t source_a_ready_panel = source_panel;
    const uint32_t source_b_ready_panel =
        kDW2AFeaturePanels + source_panel;
    const uint32_t source_a_ready_word = source_a_ready_panel / 32u;
    const uint32_t source_b_ready_word = source_b_ready_panel / 32u;
    const uint32_t source_a_ready_mask =
        1u << (source_a_ready_panel % 32u);
    const uint32_t source_b_ready_mask =
        1u << (source_b_ready_panel % 32u);

    #pragma unroll 1
    for (uint32_t range_idx = 0u;
         range_idx < backward_ranges->num_ranges; ++range_idx) {
        const auto* const range_prefix =
            physical_range_prefix + range_idx * (kNumExperts + 1u);
        if (destination_end <= range_prefix[0] ||
            destination_begin >= range_prefix[kNumExperts])
            continue;
        // Find the first owner whose half-open source interval ends after the
        // destination begins. This is upper_bound(destination_begin) over
        // range_prefix[1..kNumExperts], expressed on expert indices so empty
        // experts and duplicate prefix entries remain correct.
        uint32_t first = 0u;
        uint32_t last = kNumExperts;
        while (first < last) {
            const uint32_t middle = (first + last) >> 1;
            if (range_prefix[middle + 1u] <= destination_begin)
                first = middle + 1u;
            else
                last = middle;
        }

        #pragma unroll 1
        for (uint32_t source_expert = first;
             source_expert < kNumExperts; ++source_expert) {
            const uint32_t source_begin = range_prefix[source_expert];
            const uint32_t source_end = range_prefix[source_expert + 1u];
            if (source_begin >= destination_end)
                break;
            // Duplicate prefix entries represent empty experts. They own no
            // BF16 source rows and therefore contribute no readiness edge.
            if (source_begin == source_end)
                continue;
            const auto* const ready =
                state + Overlap::kDW2FeatureReadyMasks +
                source_expert *
                    Overlap::kDW2FeatureReadyWordsPerExpert;
            if ((ptx::ld_acq(ready + source_a_ready_word) &
                     source_a_ready_mask) == 0u ||
                (ptx::ld_acq(ready + source_b_ready_word) &
                     source_b_ready_mask) == 0u)
                return false;
        }
    }
    return true;
}

/** Stream exact dW13 operands through one CTA-local TMA quantization engine.
 *
 * Warps 12--15 form one 128-thread engine.  Each claimed
 * task owns one expert-local packed scale row, one logical operand, and one
 * 128-feature tile.  A task iterates at most four group-32 token tiles: TMA
 * loads 32x128 BF16 values, every lane quantizes one complete feature column,
 * and TMA stores the primary/residual FP8 planes.  The four UE8M0 scale bytes
 * are packed in registers and published with two 512-byte bulk stores.
 *
 * An expert becomes writable only after all of its dW2 UMMA input reads have
 * retired.  Claim and completion counters are independent, so the engine that
 * completes the final task observes every preceding release.  It then adds
 * exactly one operand-ready credit to the existing dW13 composite counter and
 * never touches that counter again; the ready-first UMMA scheduler may safely
 * reuse it as its task cursor. `kExitOnLocalDW2Terminal` releases ordinary
 * dW2 CTAs at their terminal mailbox. Disabling it lets a bounded dW13 cohort
 * reuse the same exact producer until the global expert-completion count is
 * final, without changing storage or publication semantics.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          bool kExitOnLocalDW2Terminal = true,
          uint32_t kFirstQuantWarp_ = 12u,
          uint32_t kNumQuantEngines_ =
              kK3MxFp8DW13QuantNumEngines,
          uint32_t kQuantScratchBegin_ =
              kK3MxFp8DW13QuantScratchBegin,
          uint32_t kLocalProducerCompleteMask_ = 0u,
          bool kSkipAProduction_ = false>
CUTLASS_DEVICE void
k3_mxfp8_stream_dw13_operands_background(
        const int* expert_counts,
        const K3BackwardRangeSet* backward_ranges,
        const uint32_t k_capacity,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t warp_idx,
        const uint32_t lane_idx,
        const uint32_t* local_producer_mailbox = nullptr) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    constexpr uint32_t kFirstQuantWarp = kFirstQuantWarp_;
    constexpr uint32_t kQuantWarps =
        kNumQuantEngines_ *
        kK3MxFp8DW13QuantWarpsPerEngine;
    constexpr uint32_t kThreadsPerEngine =
        kK3MxFp8DW13QuantWarpsPerEngine * 32u;
    constexpr uint32_t kAWidth = 2u * kIntermediateHidden;
    constexpr uint32_t kBWidth = kHidden;
    constexpr uint32_t kAFeatureTiles =
        kAWidth / kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kBFeatureTiles =
        kBWidth / kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kFeatureTilesPerScaleRow =
        kAFeatureTiles + kBFeatureTiles;
    constexpr uint32_t kProducedFeatureTiles = kSkipAProduction_
        ? kBFeatureTiles : kFeatureTilesPerScaleRow;
    constexpr uint32_t kSourceElements =
        kK3MxFp8DW13QuantRowsPerGroup *
        kK3MxFp8DW13QuantFeatureTile;
    constexpr uint32_t kSourceBytes =
        kSourceElements * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kValueBytes =
        kSourceElements * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kScaleBytes =
        kK3MxFp8DW13QuantFeatureTile * sizeof(uint32_t);
    constexpr uint32_t kNoTask = 0xffffffffu;
    constexpr uint32_t kStop = 0xfffffffeu;
    static_assert(
        kNumThreads == 1024u && kNumSMs % 2u == 0u &&
            kNumQuantEngines_ > 0u &&
            kNumQuantEngines_ <= kK3MxFp8DW13QuantNumEngines &&
            kFirstQuantWarp + kQuantWarps <= kNumThreads / 32u &&
            kAWidth % kK3MxFp8DW13QuantFeatureTile == 0u &&
            kBWidth % kK3MxFp8DW13QuantFeatureTile == 0u &&
            kBlockM % kK3MxFp8DW13QuantRowsPerGroup == 0u,
        "K3 streaming dW13 quantizer geometry changed");

    if (warp_idx < kFirstQuantWarp ||
        warp_idx >= kFirstQuantWarp + kQuantWarps)
        return;

    const uint32_t quant_warp = warp_idx - kFirstQuantWarp;
    const uint32_t engine =
        quant_warp / kK3MxFp8DW13QuantWarpsPerEngine;
    const uint32_t engine_thread =
        (quant_warp % kK3MxFp8DW13QuantWarpsPerEngine) * 32u + lane_idx;
    const uint32_t named_barrier =
        kK3MxFp8DW13QuantFirstUserNamedBarrier + engine;
    auto engine_sync = [=]() {
        cutlass::arch::NamedBarrier::sync(
            kThreadsPerEngine, named_barrier);
    };

    uint8_t* const engine_smem =
        smem_buffer + kQuantScratchBegin_ +
        engine * kK3MxFp8DW13QuantEngineStride;
    auto* const source_smem =
        reinterpret_cast<cutlass::bfloat16_t*>(engine_smem);
    auto* const primary_smem = engine_smem + kSourceBytes;
    auto* const residual_smem = primary_smem + kValueBytes;
    auto* const primary_scale_smem = reinterpret_cast<uint32_t*>(
        residual_smem + kValueBytes);
    auto* const residual_scale_smem =
        primary_scale_smem + kK3MxFp8DW13QuantFeatureTile;
    auto* const load_barrier = reinterpret_cast<Barrier*>(
        residual_scale_smem + kK3MxFp8DW13QuantFeatureTile);
    auto* const control = reinterpret_cast<volatile uint32_t*>(
        load_barrier + 1);
    static_assert(
        kSourceBytes + 2u * kValueBytes + 2u * kScaleBytes +
            sizeof(Barrier) + 4u * sizeof(uint32_t) ==
        kK3MxFp8DW13QuantEnginePayloadBytes);

    const auto dw2_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden, k_capacity, kBlockM);
    const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kAWidth, kBWidth, k_capacity, kBlockM);
    const uint64_t scale_phase_offset =
        k3_mxfp8_wgrad_next_scale_phase_offset(dw2_scale_layout);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            dw2_scale_layout, dw13_scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    auto* const phase_scale_arena = scale_arena + scale_phase_offset;
    K3MxFp8WgradScaleStorage scales[2];
    k3_mxfp8_bind_wgrad_scale_storage(
        dw13_scale_layout, phase_scale_arena,
        phase_scale_arena + dw13_scale_layout.raw_bytes,
        scales[0], scales[1]);

    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const scale_prefix = state + Prefix::kScalePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);
    const auto* const maps = tensor_map_pack->maps;
    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    const auto* const dw2_mailbox =
        state + Overlap::kDW2Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    // Every cluster leaves this dW2-side producer once all thirteen local
    // scheduler roles consume the terminal mailbox token.  The same CTA pair
    // immediately enters dW13, whose specialization compile-eliminates the
    // local-mailbox exit and keeps producing until global completion.  This
    // continuous handoff needs no cluster to remain behind as a shepherd.
    if (engine_thread == 0u) {
        load_barrier->init(1u);
        cutlass::arch::fence_barrier_init();
        if constexpr (!kSkipAProduction_)
            cute::prefetch_tma_descriptor(
                maps + kK3MxFp8DW13ProducerSourceAMap);
        cute::prefetch_tma_descriptor(
            maps + kK3MxFp8DW13ProducerSourceBMap);
        if constexpr (!kSkipAProduction_) {
            cute::prefetch_tma_descriptor(
                maps + kK3MxFp8DW13ProducerValueAPrimaryMap);
            cute::prefetch_tma_descriptor(
                maps + kK3MxFp8DW13ProducerValueAResidualMap);
        }
        cute::prefetch_tma_descriptor(
            maps + kK3MxFp8DW13ProducerValueBPrimaryMap);
        cute::prefetch_tma_descriptor(
            maps + kK3MxFp8DW13ProducerValueBResidualMap);
    }
    engine_sync();

    uint32_t load_phase = 0u;
    uint32_t search_begin = active_count == 0u
        ? 0u
        : (static_cast<uint32_t>(blockIdx.x) *
               kNumQuantEngines_ + engine) % active_count;
    while (true) {
        if (engine_thread == 0u) {
            control[0] = kNoTask;
            const bool all_experts_done = ptx::ld_acq(
                state + Overlap::kDW13QuantExpertsDone) >= active_count;
            bool local_dw2_terminal = false;
            if constexpr (kExitOnLocalDW2Terminal) {
                local_dw2_terminal = local_producer_mailbox != nullptr
                    ? (kLocalProducerCompleteMask_ != 0u &&
                       ptx::ld_acq(local_producer_mailbox + 3u) != 0u &&
                       ptx::ld_acq(local_producer_mailbox + 2u) == 0u &&
                       ptx::ld_acq(local_producer_mailbox) ==
                           kLocalProducerCompleteMask_)
                    : (ptx::ld_acq(dw2_mailbox + 3u) != 0u &&
                       ptx::ld_acq(dw2_mailbox + 2u) == 0u &&
                       ptx::ld_acq(dw2_mailbox) ==
                           Overlap::kExactSchedulerRoleMask);
            }
            if (all_experts_done || local_dw2_terminal) {
                control[0] = kStop;
            } else {
                for (uint32_t probe = 0u;
                     probe < active_count; ++probe) {
                    const uint32_t active =
                        (search_begin + probe) % active_count;
                    const uint32_t expert = state[
                        Overlap::kActiveExperts + active];
                    const uint32_t scale_rows =
                        scale_prefix[expert + 1u] - scale_prefix[expert];
                    if (ptx::ld_acq(
                            state + Overlap::kDW13QuantDone + expert) >=
                        kProducedFeatureTiles)
                        continue;
                    auto* const cursor =
                        state + Overlap::kDW13QuantCursor + expert;
                    uint32_t task = ptx::ld_acq(cursor);
                    bool claimed = false;
                    while (task < kFeatureTilesPerScaleRow) {
                        const uint32_t feature_panel =
                            sched::external_k_grouped_first_consumer_feature_panel(
                                task, kAFeatureTiles, kBFeatureTiles);
                        // dW13-A overwrites one of dW2-B's primary/residual
                        // panels; dW13-B overwrites both planes of dW2-A.  A
                        // panel becomes writable only after every dW2 task
                        // that reads its corresponding pair has retired.
                        const bool is_a_panel =
                            feature_panel < kAFeatureTiles;
                        if constexpr (kSkipAProduction_) {
                            // SiTU has already release-published dW13-A into
                            // the ticketed ring. Advance the legacy ordinal
                            // cursor across A panels without rereading BF16 or
                            // writing the compact value arena; the consumer's
                            // per-panel ticket acquire supplies readiness.
                            if (is_a_panel) {
                                const uint32_t observed = atomicCAS(
                                    cursor, task, task + 1u);
                                task = observed == task
                                    ? task + 1u : observed;
                                continue;
                            }
                        }
                        const uint32_t pair = is_a_panel
                            ? (feature_panel %
                                   (kIntermediateHidden / 128u)) / 2u
                            : (feature_panel - kAFeatureTiles) / 2u;
                        const uint32_t pair_index = is_a_panel
                            ? (kHidden / 256u) + pair : pair;
                        const uint32_t pair_target =
                            is_a_panel ? (kHidden / 256u)
                                       : (kIntermediateHidden / 256u);
                        const uint32_t dgrad_target =
                            (state[Overlap::kPoolBlockPrefix + expert + 1u] -
                             state[Overlap::kPoolBlockPrefix + expert]) *
                            14u;
                        // Most probes arrive before this expert's cheap local
                        // dgrad/pair prerequisites. Do not traverse physical
                        // range prefixes until both acquire checks pass.
                        if (ptx::ld_acq(
                                state + Overlap::kDW13CompositeReady + expert) <
                                dgrad_target ||
                            ptx::ld_acq(
                                state + Overlap::kDW2InputPairRetired +
                                expert *
                                    Overlap::kDW2InputPairCountersPerExpert +
                                pair_index) < pair_target) {
                            break;
                        }
                        if (is_a_panel &&
                            !k3_mxfp8_dw2_source_panels_ready_for_dw13_a<
                                kHidden, kNumExperts, kNumSMs>(
                                    backward_ranges, state, expert,
                                    feature_panel / 2u)) {
                            break;
                        }
                        const uint32_t observed = atomicCAS(
                            cursor, task, task + 1u);
                        if (observed == task) {
                            // Pair retirement is acquired through the generic
                            // proxy, but this task later overwrites the retired
                            // dW2 arena with TMA stores. Carry the alias edge
                            // into the async proxy before publishing the work
                            // record to the other engine lanes.
                            asm volatile(
                                "fence.proxy.async.global;"
                                ::: "memory");
                            claimed = true;
                            break;
                        }
                        task = observed;
                    }
                    if (!claimed)
                        continue;
                    control[0] = expert;
                    control[1] = task;
                    control[2] = scale_rows;
                    // Keep each engine on the expert it just helped until
                    // that expert's cursor is exhausted.  dW13 cannot claim
                    // any UMMA work until the final quantization task adds
                    // the expert's single operand-ready credit; rotating
                    // after every 32x128 task maximizes partial experts and
                    // leaves the ready-first consumer idle.
                    search_begin = active;
                    break;
                }
            }
        }
        engine_sync();

        const uint32_t expert = control[0];
        if (expert == kStop)
            break;
        if (expert == kNoTask) {
            if (engine_thread == 0u) {
                search_begin = active_count == 0u
                    ? 0u : (search_begin + 1u) % active_count;
                __nanosleep(64);
            }
            engine_sync();
            continue;
        }

        const uint32_t feature_ordinal = control[1];
        const uint32_t scale_rows = control[2];
        DG_DEVICE_ASSERT(
            scale_rows != 0u &&
            feature_ordinal < kFeatureTilesPerScaleRow);
        const uint32_t feature_task =
            sched::external_k_grouped_first_consumer_feature_panel(
                feature_ordinal, kAFeatureTiles, kBFeatureTiles);
        const bool is_a = feature_task < kAFeatureTiles;
        const uint32_t feature_tile = is_a
            ? feature_task : feature_task - kAFeatureTiles;
        const uint32_t width = is_a ? kAWidth : kBWidth;
        const uint32_t feature_begin =
            feature_tile * kK3MxFp8DW13QuantFeatureTile;
        const uint32_t expert_k =
            value_prefix[expert + 1u] - value_prefix[expert];
        const uint32_t num_groups =
            expert_k / kK3MxFp8DW13QuantRowsPerGroup;
        DG_DEVICE_ASSERT(
            expert_k % kK3MxFp8DW13QuantRowsPerGroup == 0u);
        for (uint32_t local_scale_row = 0u;
             local_scale_row < scale_rows; ++local_scale_row) {
        uint32_t primary_scale_word = 0x7f7f7f7fu;
        uint32_t residual_scale_word = 0x7f7f7f7fu;
        bool value_store_pending = false;

        #pragma unroll
        for (uint32_t group_in_scale_row = 0u;
             group_in_scale_row < 4u; ++group_in_scale_row) {
            const uint32_t local_group =
                local_scale_row * 4u + group_in_scale_row;
            if (local_group >= num_groups)
                continue;
            const uint32_t local_row =
                local_group * kK3MxFp8DW13QuantRowsPerGroup;
            const uint32_t source_row =
                k3_mxfp8_expert_source_pool_row<
                    kNumExperts, kBlockM, kK3MaxBackwardRanges>(
                        expert, local_row, expert_counts,
                        *backward_ranges, physical_range_prefix);
            DG_DEVICE_ASSERT(source_row +
                kK3MxFp8DW13QuantRowsPerGroup <= k_capacity);

            if (engine_thread == 0u) {
                const auto* const source_map = maps + (is_a
                    ? kK3MxFp8DW13ProducerSourceAMap
                    : kK3MxFp8DW13ProducerSourceBMap);
                tma::copy<
                    kK3MxFp8DW13QuantFeatureTile,
                    kK3MxFp8DW13QuantRowsPerGroup, 0,
                    cutlass::bfloat16_t>(
                        source_map, load_barrier, source_smem,
                        feature_begin, source_row);
                load_barrier->arrive_and_expect_tx(kSourceBytes);
            }
            load_barrier->wait(load_phase);
            load_phase ^= 1u;

            // The source tile and value-store tiles are disjoint.  Let the
            // just-issued source load overlap the preceding primary/residual
            // store group, then drain that group only when these output
            // buffers are about to be reused.  This is the same stage-reuse
            // discipline as the MegaMoE forward producer and needs no second
            // shared-memory stage.
            if (value_store_pending && engine_thread == 0u)
                ptx::tma_store_wait<0>();
            engine_sync();
            value_store_pending = false;

            float primary_amax = 0.0f;
            #pragma unroll
            for (uint32_t row = 0u;
                 row < kK3MxFp8DW13QuantRowsPerGroup; ++row) {
                const float value = static_cast<float>(
                    source_smem[
                        row * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread]);
                primary_amax = cute::max(
                    primary_amax, cute::abs(value));
            }
            float primary_scale = 1.0f;
            float primary_scale_inv = 1.0f;
            uint32_t primary_scale_byte = 0x7fu;
            k3_mxfp8_scale_pair(
                primary_amax, primary_scale, primary_scale_inv,
                primary_scale_byte);

            float residual_amax = 0.0f;
            #pragma unroll 1
            for (uint32_t row = 0u;
                 row < kK3MxFp8DW13QuantRowsPerGroup; row += 4u) {
                float values[4];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    values[i] = static_cast<float>(source_smem[
                        (row + i) * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread]);
                }
                const auto primary = __nv_fp8x4_e4m3(make_float4(
                    values[0] * primary_scale_inv,
                    values[1] * primary_scale_inv,
                    values[2] * primary_scale_inv,
                    values[3] * primary_scale_inv));
                const float4 primary_float = static_cast<float4>(primary);
                const uint32_t primary_bits = primary.__x;
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    const float primary_value = i == 0u
                        ? primary_float.x : i == 1u
                        ? primary_float.y : i == 2u
                        ? primary_float.z : primary_float.w;
                    const cutlass::bfloat16_t rounded_residual(
                        values[i] - primary_value * primary_scale);
                    source_smem[
                        (row + i) * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread] = rounded_residual;
                    primary_smem[
                        (row + i) * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread] = static_cast<uint8_t>(
                            primary_bits >> (i * 8u));
                    residual_amax = cute::max(
                        residual_amax,
                        cute::abs(static_cast<float>(rounded_residual)));
                }
            }

            float residual_scale = 1.0f;
            float residual_scale_inv = 1.0f;
            uint32_t residual_scale_byte = 0x7fu;
            k3_mxfp8_scale_pair(
                residual_amax, residual_scale, residual_scale_inv,
                residual_scale_byte);
            #pragma unroll 1
            for (uint32_t row = 0u;
                 row < kK3MxFp8DW13QuantRowsPerGroup; row += 4u) {
                float residual_values[4];
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    residual_values[i] = static_cast<float>(source_smem[
                        (row + i) * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread]);
                }
                const auto residual = __nv_fp8x4_e4m3(make_float4(
                    residual_values[0] * residual_scale_inv,
                    residual_values[1] * residual_scale_inv,
                    residual_values[2] * residual_scale_inv,
                    residual_values[3] * residual_scale_inv));
                const uint32_t residual_bits = residual.__x;
                #pragma unroll
                for (uint32_t i = 0u; i < 4u; ++i) {
                    residual_smem[
                        (row + i) * kK3MxFp8DW13QuantFeatureTile +
                        engine_thread] = static_cast<uint8_t>(
                            residual_bits >> (i * 8u));
                }
            }

            const uint32_t shift = group_in_scale_row * 8u;
            primary_scale_word =
                (primary_scale_word & ~(0xffu << shift)) |
                ((primary_scale_byte & 0xffu) << shift);
            residual_scale_word =
                (residual_scale_word & ~(0xffu << shift)) |
                ((residual_scale_byte & 0xffu) << shift);
            // Every generic-proxy writer must publish its own shared-memory
            // bytes before the elected lane asks the async proxy to read the
            // tile.  An elected-lane-only fence does not cover peer lanes.
            cute::tma_store_fence();
            engine_sync();
            if (engine_thread == 0u) {
                const uint32_t destination_row =
                    value_prefix[expert] + local_row;
                const auto* const primary_map = maps + (is_a
                    ? kK3MxFp8DW13ProducerValueAPrimaryMap
                    : kK3MxFp8DW13ProducerValueBPrimaryMap);
                const auto* const residual_map = maps + (is_a
                    ? kK3MxFp8DW13ProducerValueAResidualMap
                    : kK3MxFp8DW13ProducerValueBResidualMap);
                cute::SM90_TMA_STORE_2D::copy(
                    primary_map, primary_smem,
                    feature_begin, destination_row);
                cute::SM90_TMA_STORE_2D::copy(
                    residual_map, residual_smem,
                    feature_begin, destination_row);
                cute::tma_store_arrive();
            }
            engine_sync();
            value_store_pending = true;
        }

        const uint32_t native_scale_feature =
            (engine_thread % 32u) * 4u + engine_thread / 32u;
        primary_scale_smem[native_scale_feature] = primary_scale_word;
        residual_scale_smem[native_scale_feature] = residual_scale_word;
        cute::tma_store_fence();
        engine_sync();
        if (engine_thread == 0u) {
            const uint32_t global_scale_row =
                scale_prefix[expert] + local_scale_row;
            auto* const primary_scales = is_a
                ? scales[0].primary_packed : scales[1].primary_packed;
            auto* const residual_scales = is_a
                ? scales[0].residual_packed : scales[1].residual_packed;
            const uint64_t scale_offset =
                static_cast<uint64_t>(global_scale_row) * width +
                feature_begin;
            ptx::tma_store_1d(
                primary_scales + scale_offset,
                primary_scale_smem, kScaleBytes);
            ptx::tma_store_1d(
                residual_scales + scale_offset,
                residual_scale_smem, kScaleBytes);
            cute::tma_store_arrive();
            ptx::tma_store_wait<0>();
        }
        engine_sync();
        }

        if (engine_thread == 0u) {
            // The unique owner has retired every value and packed-scale TMA
            // store for this panel. Bridge async to generic visibility before
            // publishing the exact scale-row target checked by the scheduler.
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __threadfence();
            auto* const feature_done =
                state + Overlap::kDW13FeatureDone +
                expert * Overlap::kDW13FeaturePanelsPerExpert +
                feature_task;
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(feature_done), "r"(scale_rows)
                : "memory");
            const uint32_t completed_panels = ptx::atomic_add_acq_rel(
                state + Overlap::kDW13QuantDone + expert, 1u);
            DG_DEVICE_ASSERT(
                completed_panels < kProducedFeatureTiles);
            if (completed_panels + 1u ==
                kProducedFeatureTiles) {
                if constexpr (kLocalProducerCompleteMask_ != 0u) {
                    // Only the hybrid ready-wgrad specialization uses the
                    // shared composite counter as a ready-first task cursor.
                    // Ordinary one-range exact wgrad retains the legacy
                    // counter contract and must not receive this extra credit.
                    const uint32_t pool_blocks =
                        state[Overlap::kPoolBlockPrefix + expert + 1u] -
                        state[Overlap::kPoolBlockPrefix + expert];
                    const uint32_t dgrad_target = pool_blocks * 14u;
                    const uint32_t previous = ptx::atomic_add_acq_rel(
                        state + Overlap::kDW13CompositeReady + expert, 1u);
                    DG_DEVICE_ASSERT(previous == dgrad_target);
                }
                const uint32_t experts_done = ptx::atomic_add_acq_rel(
                    state + Overlap::kDW13QuantExpertsDone, 1u);
                DG_DEVICE_ASSERT(experts_done < active_count);
            }
        }
        engine_sync();
    }

    if (engine_thread == 0u) {
        Barrier::invalidate(
            reinterpret_cast<Barrier::ValueType const*>(load_barrier));
    }
    engine_sync();
}

}  // namespace detail

#if DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD

/** Build the immutable expert/value/scale prefixes for one exact generation.
 *
 * This metadata-only operation deliberately has no internal barrier.  The
 * parent calls it while the persistent grid is still converged and publishes
 * the result through its final uniform release edge.  The serial compatibility
 * wrapper below supplies the same edge before entering the ordinary producer.
 */
template <uint32_t kNumExperts, uint32_t kBlockM>
CUTLASS_DEVICE __noinline__ void
build_k3_mxfp8_three_term_wgrad_metadata(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        int* prefix_storage,
        const uint32_t initializer_cta_idx) {
    using Prefix = K3MxFp8WgradPrefixLayout<
        kNumExperts, kK3MaxBackwardRanges>;
    auto* grouped_layout = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kGroupedLayout);
    auto* value_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kValuePrefix);
    auto* scale_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kScalePrefix);
    auto* physical_range_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kPhysicalRangePrefix);

    if (initializer_cta_idx != 0u || threadIdx.x != 0u)
        return;

    uint32_t total_k = 0u;
    uint32_t total_scale_rows = 0u;
    value_prefix[0] = 0u;
    scale_prefix[0] = 0u;
    for (uint32_t expert = 0u; expert < kNumExperts; ++expert) {
        uint32_t expert_k = 0u;
        for (uint32_t reverse_iteration = 0u;
             reverse_iteration < backward_ranges.num_ranges;
             ++reverse_iteration) {
            const uint32_t range_idx =
                backward_ranges.reverse_range_index(reverse_iteration);
            const uint32_t count = static_cast<uint32_t>(__ldg(
                expert_counts + backward_ranges.expert_counts_begin(
                    range_idx, kNumExperts) + expert));
            expert_k += math::ceil_div(count, kBlockM) * kBlockM;
        }
        grouped_layout[expert] = expert_k;
        total_k += expert_k;
        total_scale_rows += k3_mxfp8_wgrad_scale_rows(expert_k);
        value_prefix[expert + 1u] = total_k;
        scale_prefix[expert + 1u] = total_scale_rows;
    }
    for (uint32_t range_idx = 0u;
         range_idx < backward_ranges.num_ranges; ++range_idx) {
        const auto& range = backward_ranges.ranges[range_idx];
        uint32_t physical_row = range.pool_row_begin;
        auto* range_prefix = physical_range_prefix +
            range_idx * (kNumExperts + 1u);
        range_prefix[0] = physical_row;
        for (uint32_t expert = 0u; expert < kNumExperts; ++expert) {
            const uint32_t count = static_cast<uint32_t>(__ldg(
                expert_counts + backward_ranges.expert_counts_begin(
                    range_idx, kNumExperts) + expert));
            physical_row += math::ceil_div(count, kBlockM) * kBlockM;
            range_prefix[expert + 1u] = physical_row;
        }
    }
    // Preserve the fixed ABI when fewer than kK3MaxBackwardRanges are active.
    for (uint32_t range_idx = backward_ranges.num_ranges;
         range_idx < kK3MaxBackwardRanges; ++range_idx) {
        auto* range_prefix = physical_range_prefix +
            range_idx * (kNumExperts + 1u);
        for (uint32_t expert = 0u; expert <= kNumExperts; ++expert)
            range_prefix[expert] = 0u;
    }
}

template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          bool kDirectPackedScales,
          typename GridBarrier>
CUTLASS_DEVICE __noinline__ uint32_t
prepare_k3_mxfp8_three_term_wgrad_operands_subset(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        uint32_t source_a_width,
        uint32_t destination_a_primary_row_stride,
        uint32_t destination_a_residual_row_stride,
        const cutlass::bfloat16_t* source_b,
        uint32_t source_b_width,
        uint32_t destination_b_primary_row_stride,
        uint32_t destination_b_residual_row_stride,
        const K3MxFp8WgradScaleStorage& a_scale_storage,
        const K3MxFp8WgradScaleStorage& b_scale_storage,
        uint32_t k_capacity,
        int* prefix_storage,
        uint8_t* a_primary_values,
        uint8_t* a_residual_values,
        uint8_t* b_primary_values,
        uint8_t* b_residual_values,
        const cute::TmaDescriptor& a_primary_map,
        const cute::TmaDescriptor& a_residual_map,
        const cute::TmaDescriptor& b_primary_map,
        const cute::TmaDescriptor& b_residual_map,
        uint8_t* smem_buffer,
        uint32_t producer_cta_idx,
        uint32_t num_producer_ctas,
        GridBarrier grid_barrier);

/** Initialize one allocation-free dW2 overlap generation before divergence.
 *
 * This outlined helper builds the immutable prefixes and resets the dynamic
 * scheduler state in retired W2 storage.  TensorMap publication is deferred
 * to the bounded suffix producer set: carrying four 128-byte descriptors
 * through this middle-of-parent call exceeds the 1024-thread kernel's fixed
 * 64-register launch budget.  The full-grid release edge below publishes all
 * metadata before complete CTA clusters are allowed to diverge.
 */
template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          typename GridBarrier>
CUTLASS_DEVICE __noinline__ void
prepare_k3_mxfp8_dw2_overlap_state(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        uint32_t* state,
        GridBarrier grid_barrier) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;

    build_k3_mxfp8_three_term_wgrad_metadata<kNumExperts, kBlockM>(
        expert_counts, backward_ranges, reinterpret_cast<int*>(state),
        static_cast<uint32_t>(blockIdx.x));

    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        auto* const grouped = state;
        const auto* const value_prefix = state + Prefix::kValuePrefix;
        const auto* const physical_range_prefix =
            state + Prefix::kPhysicalRangePrefix;
        auto* const pool_prefix = state + Overlap::kPoolBlockPrefix;
        auto* const active_experts = state + Overlap::kActiveExperts;
        uint32_t pool_blocks = 0u;
        uint32_t active_count = 0u;
        pool_prefix[0] = 0u;
        for (uint32_t expert = 0u; expert < kNumExperts; ++expert) {
            const uint32_t expert_k = grouped[expert];
            DG_DEVICE_ASSERT(expert_k % kBlockM == 0u);
            if (expert_k != 0u)
                active_experts[active_count++] = expert;
            pool_blocks += expert_k / kBlockM;
            pool_prefix[expert + 1u] = pool_blocks;
        }

        for (uint32_t word = Overlap::kControl;
             word < Overlap::kDW2OperandReady; ++word)
            state[word] = 0u;
        for (uint32_t word = 0u;
             word < Overlap::kDW2SuffixExpertMaskWords; ++word) {
            state[Overlap::kDW2SuffixACompleteMasks + word] = 0u;
        }
        for (uint32_t expert = 0u; expert < kNumExperts; ++expert) {
            state[Overlap::kDW2OperandReady + expert] = 0u;
            state[Overlap::kDW2InputRetired + expert] = 0u;
            state[Overlap::kDW13CompositeReady + expert] = 0u;
            state[Overlap::kDW13QuantCursor + expert] = 0u;
            state[Overlap::kDW13QuantDone + expert] = 0u;
            state[Overlap::kDW2SuffixQuantCursor + expert] = 0u;
            state[Overlap::kDW2SuffixQuantDone + expert] = 0u;
            state[Overlap::kDW2SuffixAQuantDone + expert] = 0u;
            for (uint32_t word = 0u;
                 word < Overlap::kDW2SuffixExpertMaskWords; ++word) {
                state[Overlap::kDW2SuffixBAliasDependencies +
                      expert * Overlap::kDW2SuffixExpertMaskWords + word] =
                    0u;
            }
            for (uint32_t word = 0u;
                 word < Overlap::kDW2FeatureReadyWordsPerExpert; ++word) {
                state[Overlap::kDW2FeatureReadyMasks +
                      expert * Overlap::kDW2FeatureReadyWordsPerExpert +
                      word] = 0u;
            }
        }

        // dW2-B writes compact expert-major rows into the same physical arena
        // that still backs dW2-A's range-major BF16 source.  Record every
        // source expert whose half-open physical interval intersects one B
        // destination.  The suffix producer later checks this immutable mask
        // before claiming B, avoiding both a full-grid A/B barrier and the
        // historical two-range read-after-overwrite race.
        for (uint32_t destination_expert = 0u;
             destination_expert < kNumExperts; ++destination_expert) {
            const uint32_t destination_begin =
                value_prefix[destination_expert];
            const uint32_t destination_end =
                value_prefix[destination_expert + 1u];
            auto* const dependency =
                state + Overlap::kDW2SuffixBAliasDependencies +
                destination_expert * Overlap::kDW2SuffixExpertMaskWords;
            for (uint32_t range_idx = 0u;
                 range_idx < backward_ranges.num_ranges; ++range_idx) {
                const auto* const range_prefix = physical_range_prefix +
                    range_idx * (kNumExperts + 1u);
                for (uint32_t source_expert = 0u;
                     source_expert < kNumExperts; ++source_expert) {
                    if (k3_mxfp8_wgrad_row_intervals_overlap(
                            destination_begin, destination_end,
                            range_prefix[source_expert],
                            range_prefix[source_expert + 1u])) {
                        dependency[source_expert / 32u] |=
                            1u << (source_expert % 32u);
                    }
                }
            }
        }
        for (uint32_t cluster = 0u;
             cluster < Overlap::kNumClusters; ++cluster) {
            auto* const dw2_mailbox = state + Overlap::kDW2Mailboxes +
                cluster * Overlap::kMailboxWordsPerCluster;
            auto* const dw13_mailbox = state + Overlap::kDW13Mailboxes +
                cluster * Overlap::kMailboxWordsPerCluster;
            dw2_mailbox[0] = Overlap::kExactSchedulerRoleMask;
            dw2_mailbox[1] = 0u;
            dw2_mailbox[2] = 0u;
            dw2_mailbox[3] = 0u;
            dw13_mailbox[0] = Overlap::kExactSchedulerRoleMask;
            dw13_mailbox[1] = 0u;
            dw13_mailbox[2] = 0u;
            dw13_mailbox[3] = 0u;
        }
        state[Overlap::kMagic] = 0x4b334d58u; // "K3MX"
        state[Overlap::kActiveCount] = active_count;
        state[Overlap::kTotalPoolBlocks] = pool_blocks;
        state[Overlap::kDW2Tasks] =
            active_count * Overlap::kDW2ClusterTasksPerExpert;
        state[Overlap::kDW13Tasks] =
            active_count * Overlap::kDW13ClusterTasksPerExpert;
    }
    const uint32_t global_thread =
        static_cast<uint32_t>(blockIdx.x) * kNumThreads + threadIdx.x;
    constexpr uint32_t kGlobalThreads = kNumSMs * kNumThreads;
    constexpr uint32_t kNumPairCounters =
        kNumExperts * Overlap::kDW2InputPairCountersPerExpert;
    for (uint32_t word = global_thread;
         word < kNumPairCounters; word += kGlobalThreads) {
        state[Overlap::kDW2InputPairRetired + word] = 0u;
    }
    constexpr uint32_t kNumFeatureCounters =
        kNumExperts * Overlap::kDW13FeaturePanelsPerExpert;
    for (uint32_t word = global_thread;
         word < kNumFeatureCounters; word += kGlobalThreads) {
        state[Overlap::kDW13FeatureDone + word] = 0u;
    }
    grid_barrier();
}

/** Stream multirange exact dW2 operands from one suffix-CTA subset.
 *
 * Packed multirange storage is range-major while the exact operand arena is
 * expert-major, so a dW2-B destination row can alias an unread dW2-A source
 * row owned by another expert/range. The subset therefore produces A globally
 * and retains exactly one A-to-B retirement edge. After that edge, A panels
 * are release-published and complete CTA pairs claim experts, produce B in
 * feature-major order, and release-publish each B panel independently. This
 * lets the ready-first UMMA scheduler overlap one expert's dW2 with later B
 * production instead of waiting for a second subset-wide barrier.
 *
 * `kDW2Cursor` and mailbox word one are temporary producer claim state. The
 * ready-first consumer must claim output tasks through `kDW2OperandReady`,
 * which remains zero until the consumer owns it. No operand/state allocation
 * is added by this hybrid producer.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kNumProducerCTAs>
CUTLASS_DEVICE __noinline__ uint32_t
produce_k3_mxfp8_dw2_overlap_operands(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const cutlass::bfloat16_t* source_b,
        const uint32_t k_capacity,
        const cutlass::bfloat16_t* x_pool_output,
        const cutlass::bfloat16_t* grad_y_unweighted_output,
        const cutlass::bfloat16_t* down_unweighted_output,
        cutlass::bfloat16_t* grad_ye_output,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t producer_cta_idx) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    static_assert(kNumThreads == 1024u);
    static_assert(kNumSMs % 2u == 0u);
    static_assert(kNumProducerCTAs % 2u == 0u);
    static_assert(kHidden / 128u == 28u);
    static_assert(kIntermediateHidden / 128u == 24u);
    static_assert(
        kHidden / 128u + kIntermediateHidden / 128u ==
            Overlap::kDW2FeaturePanelsPerExpert &&
            Overlap::kDW2FeatureReadyWordsPerExpert == 2u,
        "K3 dW2 feature-mask geometry changed");
    const auto scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden, k_capacity, kBlockM);
    const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden, k_capacity, kBlockM);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            scale_layout, dw13_scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    K3MxFp8WgradScaleStorage scales[2];
    k3_mxfp8_bind_wgrad_scale_storage(
        scale_layout, scale_arena, scale_arena + scale_layout.raw_bytes,
        scales[0], scales[1]);

    auto* const safe_h_arena = reinterpret_cast<uint8_t*>(
        x_pool_output == grad_y_unweighted_output
        ? const_cast<cutlass::bfloat16_t*>(down_unweighted_output)
        : const_cast<cutlass::bfloat16_t*>(grad_y_unweighted_output));
    auto* const retired_grad_ye_arena =
        reinterpret_cast<uint8_t*>(grad_ye_output);
    const auto* const maps =
        tensor_map_pack->maps + kK3MxFp8DW2ValueAPrimaryMap;
    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t total_k = value_prefix[kNumExperts];
    DG_DEVICE_ASSERT(total_k <= k_capacity);
    K3MxFp8WgradSubsetBarrier<kNumProducerCTAs> subset_barrier{
        state + Overlap::kDW2SubsetBarrierCount,
        state + Overlap::kDW2SubsetBarrierSense};

    detail::k3_mxfp8_produce_operand<
        kNumExperts, kBlockM, kK3MaxBackwardRanges,
        kNumSMs, kNumThreads, true, false>(
            source_a, kHidden, kHidden, kHidden,
            k_capacity, total_k, 0u,
            expert_counts, backward_ranges, value_prefix,
            physical_range_prefix,
            safe_h_arena,
            safe_h_arena + static_cast<uint64_t>(k_capacity) * kHidden,
            reinterpret_cast<uint8_t*>(scales[0].primary_packed),
            reinterpret_cast<uint8_t*>(scales[0].residual_packed),
            maps[0], maps[1], producer_cta_idx, kNumProducerCTAs,
            smem_buffer);

    // B aliases range-major A input rows. Retire every producer CTA's A loads
    // before any cluster starts B, while preserving all completed A stores.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence();
    subset_barrier();

    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);
    if (threadIdx.x == 0u) {
        constexpr uint32_t kAReady =
            0xffffffffu >> (32u - kHidden / 128u);
        for (uint32_t active = producer_cta_idx;
             active < active_count; active += kNumProducerCTAs) {
            const uint32_t expert =
                state[Overlap::kActiveExperts + active];
            auto* const ready = state + Overlap::kDW2FeatureReadyMasks +
                expert * Overlap::kDW2FeatureReadyWordsPerExpert;
            // B panels zero through three share this word's upper bits and
            // may publish concurrently; an atomic OR cannot erase them.
            ptx::red_or_rel_gpu(ready, kAReady);
        }
    }

    const uint32_t cluster_rank =
        static_cast<uint32_t>(cute::block_rank_in_cluster());
    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    auto* const claim_mailbox = state + Overlap::kDW2Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    auto* const shared_ticket = reinterpret_cast<uint32_t*>(smem_buffer);
    DG_DEVICE_ASSERT(cluster_rank < 2u);

    while (true) {
        if (cluster_rank == 0u && threadIdx.x == 0u) {
            const uint32_t ticket = ptx::atomic_add_acq_rel(
                state + Overlap::kDW2Cursor, 1u);
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(claim_mailbox + 1u), "r"(ticket)
                : "memory");
        }
        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();
        if (threadIdx.x == 0u)
            *shared_ticket = ptx::ld_acq(claim_mailbox + 1u);
        __syncthreads();

        const uint32_t ticket = *shared_ticket;
        if (ticket >= active_count)
            break;
        const uint32_t expert = state[
            Overlap::kActiveExperts + ticket];
        const uint32_t expert_k =
            value_prefix[expert + 1u] - value_prefix[expert];
        DG_DEVICE_ASSERT(expert_k != 0u && expert_k % kBlockM == 0u);

        detail::k3_mxfp8_produce_operand<
            kNumExperts, kBlockM, kK3MaxBackwardRanges,
            kNumSMs, kNumThreads, true, true, true>(
                source_b, kIntermediateHidden,
                2u * kHidden, 2u * kHidden,
                k_capacity, expert_k, expert,
                expert_counts, backward_ranges, value_prefix,
                physical_range_prefix,
                retired_grad_ye_arena,
                retired_grad_ye_arena + kIntermediateHidden,
                reinterpret_cast<uint8_t*>(scales[1].primary_packed),
                reinterpret_cast<uint8_t*>(scales[1].residual_packed),
                maps[2], maps[3], cluster_rank, 2u, smem_buffer,
                kHidden / 128u,
                state + Overlap::kDW2FeatureReadyMasks + expert * 2u);

        // The feature-major producer has already release-published each B
        // panel. Keep the pair in lockstep before reusing its claim mailbox.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();
    }

    return total_k;
}

/** Stream exact dW2 operands one expert at a time from complete CTA pairs.
 *
 * Cluster rank zero claims one active expert through `kDW2Cursor` and
 * release-broadcasts that ticket in mailbox word one.  Both CTAs acquire the
 * ticket, then stride the expert-local A and B tile sets by cluster rank.  The
 * A cluster edge protects the retired BF16 source aliased by B's destination;
 * the B edge publishes all value and packed-scale stores before rank zero
 * release-stores the expert's ready-first scheduler seed.  Mailbox words zero
 * and three remain owned by the later dynamic scheduler, which overwrites the
 * temporary word-one ticket only before release-publishing its sequence.
 *
 * Empty experts never enter `kActiveExperts`.  Expert-relative production
 * preserves variable-length and internal 64-row tails without a global subset
 * barrier, allocation, or extra state word. The caller uses this complete A/B
 * variant only for one-range exact mode; multirange uses the hybrid global-A,
 * expert-local progressive-B producer above.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads>
CUTLASS_DEVICE __noinline__ uint32_t
produce_k3_mxfp8_dw2_expert_sticky_operands(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const cutlass::bfloat16_t* source_b,
        const uint32_t k_capacity,
        const cutlass::bfloat16_t* x_pool_output,
        const cutlass::bfloat16_t* grad_y_unweighted_output,
        const cutlass::bfloat16_t* down_unweighted_output,
        cutlass::bfloat16_t* grad_ye_output,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        uint32_t* state,
        uint8_t* smem_buffer) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    using Prefix = typename Overlap::Prefix;
    static_assert(kNumThreads == 1024u);
    static_assert(kNumSMs % 2u == 0u);
    static_assert(
        kHidden / 128u + kIntermediateHidden / 128u ==
            Overlap::kDW2FeaturePanelsPerExpert,
        "K3 dW2 feature-mask geometry changed");
    DG_DEVICE_ASSERT(backward_ranges.num_ranges == 1u);

    const auto scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden, k_capacity, kBlockM);
    const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden, k_capacity, kBlockM);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            scale_layout, dw13_scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    K3MxFp8WgradScaleStorage scales[2];
    k3_mxfp8_bind_wgrad_scale_storage(
        scale_layout, scale_arena, scale_arena + scale_layout.raw_bytes,
        scales[0], scales[1]);

    auto* const safe_h_arena = reinterpret_cast<uint8_t*>(
        x_pool_output == grad_y_unweighted_output
        ? const_cast<cutlass::bfloat16_t*>(down_unweighted_output)
        : const_cast<cutlass::bfloat16_t*>(grad_y_unweighted_output));
    auto* const retired_grad_ye_arena =
        reinterpret_cast<uint8_t*>(grad_ye_output);
    const auto* const maps =
        tensor_map_pack->maps + kK3MxFp8DW2ValueAPrimaryMap;
    const auto* const value_prefix = state + Prefix::kValuePrefix;
    const auto* const physical_range_prefix =
        state + Prefix::kPhysicalRangePrefix;
    const uint32_t active_count = ptx::ld_acq(
        state + Overlap::kActiveCount);
    const uint32_t cluster_rank =
        static_cast<uint32_t>(cute::block_rank_in_cluster());
    const uint32_t cluster_idx =
        static_cast<uint32_t>(blockIdx.x) / 2u;
    auto* const claim_mailbox = state + Overlap::kDW2Mailboxes +
        cluster_idx * Overlap::kMailboxWordsPerCluster;
    auto* const shared_ticket = reinterpret_cast<uint32_t*>(smem_buffer);
    DG_DEVICE_ASSERT(cluster_rank < 2u);

    while (true) {
        if (cluster_rank == 0u && threadIdx.x == 0u) {
            const uint32_t ticket = ptx::atomic_add_acq_rel(
                state + Overlap::kDW2Cursor, 1u);
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :: "l"(claim_mailbox + 1u), "r"(ticket)
                : "memory");
        }
        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();
        if (threadIdx.x == 0u)
            *shared_ticket = ptx::ld_acq(claim_mailbox + 1u);
        __syncthreads();

        const uint32_t ticket = *shared_ticket;
        if (ticket >= active_count)
            break;
        const uint32_t expert = state[
            Overlap::kActiveExperts + ticket];
        const uint32_t expert_k =
            value_prefix[expert + 1u] - value_prefix[expert];
        DG_DEVICE_ASSERT(expert_k != 0u && expert_k % kBlockM == 0u);

        detail::k3_mxfp8_produce_operand<
            kNumExperts, kBlockM, kK3MaxBackwardRanges,
            kNumSMs, kNumThreads, true, true, true>(
                source_a, kHidden, kHidden, kHidden,
                k_capacity, expert_k, expert,
                expert_counts, backward_ranges, value_prefix,
                physical_range_prefix,
                safe_h_arena,
                safe_h_arena +
                    static_cast<uint64_t>(k_capacity) * kHidden,
                reinterpret_cast<uint8_t*>(scales[0].primary_packed),
                reinterpret_cast<uint8_t*>(scales[0].residual_packed),
                maps[0], maps[1], cluster_rank, 2u, smem_buffer,
                0u,
                state + Overlap::kDW2FeatureReadyMasks + expert * 2u);

        // Every writer publishes its own generic/async proxy operations before
        // the complete cluster crosses the A-source/B-destination alias edge.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();

        detail::k3_mxfp8_produce_operand<
            kNumExperts, kBlockM, kK3MaxBackwardRanges,
            kNumSMs, kNumThreads, true, true, true>(
                source_b, kIntermediateHidden,
                2u * kHidden, 2u * kHidden,
                k_capacity, expert_k, expert,
                expert_counts, backward_ranges, value_prefix,
                physical_range_prefix,
                retired_grad_ye_arena,
                retired_grad_ye_arena + kIntermediateHidden,
                reinterpret_cast<uint8_t*>(scales[1].primary_packed),
                reinterpret_cast<uint8_t*>(scales[1].residual_packed),
                maps[2], maps[3], cluster_rank, 2u, smem_buffer,
                kHidden / 128u,
                state + Overlap::kDW2FeatureReadyMasks + expert * 2u);

        // Keep the producer pair in expert lockstep before mailbox reuse.  The
        // two per-expert mask words already release-published every panel;
        // kDW2OperandReady is now exclusively the consumer's zero-based CAS
        // cursor and must never be overwritten by a late producer.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        __syncthreads();
        comm::cluster_sync_with_relaxed_arrive();
    }

    return value_prefix[kNumExperts];
}

/** Produce dW13 operands after every aliased dW2 input tile has retired.
 *
 * dW2-B stores primary/residual values in two I-wide planes of the retired
 * dW2-A arena.  Both planes retain that arena's physical `2H` byte row pitch;
 * dW13-A-primary consumes a logical `2I` feature extent with the same `2H`
 * pitch.  This preserves expert-local byte disjointness while dW2 clusters
 * progress independently.  The compatibility producer below still
 * acquire-waits every expert because it launches one flattened full-grid
 * batch; the expert-local pipeline may instead start an expert at its exact
 * `kDW2InputRetired` count without touching any neighbor's rows.  Scale bytes
 * live in a separate second-phase slice for the same reason.
 */
template <uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kNumProducerCTAs>
CUTLASS_DEVICE __noinline__ uint32_t
produce_k3_mxfp8_dw13_overlap_operands(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const cutlass::bfloat16_t* source_b,
        const uint32_t k_capacity,
        const cutlass::bfloat16_t* x_pool_output,
        const cutlass::bfloat16_t* grad_y_unweighted_output,
        const cutlass::bfloat16_t* down_unweighted_output,
        cutlass::bfloat16_t* grad_ye_output,
        cutlass::bfloat16_t* h_weighted_output,
        const cutlass::float_e4m3_t* scale_arena_source,
        const K3MxFp8WgradTensorMapPack<cute::TmaDescriptor>*
            tensor_map_pack,
        uint32_t* state,
        uint8_t* smem_buffer,
        const uint32_t producer_cta_idx,
        const uint32_t launch_epoch) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    K3MxFp8WgradSubsetBarrier<kNumProducerCTAs> subset_barrier{
        state + Overlap::kDW13SubsetBarrierCount,
        state + Overlap::kDW13SubsetBarrierSense};

    if (producer_cta_idx == 0u && threadIdx.x == 0u) {
        const uint32_t active_count = ptx::ld_acq(
            state + Overlap::kActiveCount);
        for (uint32_t active = 0u; active < active_count; ++active) {
            const uint32_t expert = state[
                Overlap::kActiveExperts + active];
            while (ptx::ld_acq(
                       state + Overlap::kDW2InputRetired + expert) <
                   Overlap::kDW2ClusterTasksPerExpert) {
                __nanosleep(64);
            }
        }
    }
    subset_barrier();

    const auto dw2_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        kHidden, kIntermediateHidden, k_capacity, kBlockM);
    const auto scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
        2u * kIntermediateHidden, kHidden, k_capacity, kBlockM);
    const uint64_t phase_offset =
        k3_mxfp8_wgrad_next_scale_phase_offset(dw2_scale_layout);
    auto* const scale_arena = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(scale_arena_source));
    DG_DEVICE_ASSERT(
        k3_mxfp8_wgrad_two_phase_scale_bytes(
            dw2_scale_layout, scale_layout) <=
        static_cast<uint64_t>(k_capacity) * kHidden);
    auto* const phase_scale_arena = scale_arena + phase_offset;
    K3MxFp8WgradScaleStorage scales[2];
    k3_mxfp8_bind_wgrad_scale_storage(
        scale_layout, phase_scale_arena,
        phase_scale_arena + scale_layout.raw_bytes,
        scales[0], scales[1]);

    auto* const safe_h_arena = reinterpret_cast<uint8_t*>(
        x_pool_output == grad_y_unweighted_output
        ? const_cast<cutlass::bfloat16_t*>(down_unweighted_output)
        : const_cast<cutlass::bfloat16_t*>(grad_y_unweighted_output));
    auto* const retired_grad_ye_arena =
        reinterpret_cast<uint8_t*>(grad_ye_output);
    auto* const retired_h_arena =
        reinterpret_cast<uint8_t*>(h_weighted_output);
    const auto* const maps =
        tensor_map_pack->maps + kK3MxFp8DW13ValueAPrimaryMap;
    const uint32_t total_k =
        prepare_k3_mxfp8_three_term_wgrad_operands_subset<
            kNumExperts, kBlockM, kNumSMs, kNumThreads, false>(
                expert_counts, backward_ranges,
                source_a, 2u * kIntermediateHidden,
                2u * kHidden, 2u * kIntermediateHidden,
                source_b, kHidden, kHidden, kHidden,
                scales[0], scales[1], k_capacity,
                reinterpret_cast<int*>(state),
                retired_grad_ye_arena, retired_h_arena,
                safe_h_arena,
                safe_h_arena +
                    static_cast<uint64_t>(k_capacity) * kHidden,
                maps[0], maps[1], maps[2], maps[3], smem_buffer,
                producer_cta_idx, kNumProducerCTAs, subset_barrier);

    if (producer_cta_idx == 0u && threadIdx.x == 0u) {
        const uint32_t active_count = state[Overlap::kActiveCount];
        for (uint32_t active = 0u; active < active_count; ++active) {
            const uint32_t expert = state[
                Overlap::kActiveExperts + active];
            const uint32_t previous = ptx::atomic_add_acq_rel(
                state + Overlap::kDW13CompositeReady + expert, 1u);
            const uint32_t pool_blocks =
                state[Overlap::kPoolBlockPrefix + expert + 1u] -
                state[Overlap::kPoolBlockPrefix + expert];
            DG_DEVICE_ASSERT(
                previous <= pool_blocks * 14u);
        }
        asm volatile(
            "st.release.gpu.global.u32 [%0], %1;"
            :: "l"(state + Overlap::kDW13Epoch), "r"(launch_epoch)
            : "memory");
    }
    return total_k;
}

/** Open the per-expert dW13 generation for background operand production.
 *
 * The row-preserving value alias (logical `2I`, physical `2H` pitch) and
 * disjoint scale slice permit per-expert production. This compatibility edge
 * deliberately retains the conservative all-expert wait until the caller
 * switches to the expert-local producer. Once the selected subset observes
 * that condition, the distinct dW13 epoch lets all persistent CTAs enter the
 * grouped body; suffix warps publish readiness expert by expert under the
 * existing composite counter.
 */
template <uint32_t kNumExperts, uint32_t kNumSMs,
          uint32_t kNumProducerCTAs>
CUTLASS_DEVICE __noinline__ void
publish_k3_mxfp8_dw13_background_generation(
        uint32_t* state,
        const uint32_t producer_cta_idx,
        const uint32_t launch_epoch) {
    using Overlap = K3MxFp8WgradOverlapStateLayout<
        kNumExperts, kNumSMs, kK3MaxBackwardRanges>;
    K3MxFp8WgradSubsetBarrier<kNumProducerCTAs> subset_barrier{
        state + Overlap::kDW13SubsetBarrierCount,
        state + Overlap::kDW13SubsetBarrierSense};

    if (producer_cta_idx == 0u && threadIdx.x == 0u) {
        const uint32_t active_count = ptx::ld_acq(
            state + Overlap::kActiveCount);
        for (uint32_t active = 0u; active < active_count; ++active) {
            const uint32_t expert =
                state[Overlap::kActiveExperts + active];
            while (ptx::ld_acq(
                       state + Overlap::kDW2InputRetired + expert) <
                   Overlap::kDW2ClusterTasksPerExpert) {
                __nanosleep(64);
            }
        }
    }
    subset_barrier();

    if (producer_cta_idx == 0u && threadIdx.x == 0u) {
        asm volatile(
            "st.release.gpu.global.u32 [%0], %1;"
            :: "l"(state + Overlap::kDW13Epoch), "r"(launch_epoch)
            : "memory");
    }
}

/** Build two K3 wgrad operands and their compact scale planes in-place.
 *
 * Value descriptors, primary/residual destination row strides, and
 * scale-storage records are deliberately independent. dW2-B can therefore
 * write two I-wide planes with the retired dW2-A arena's `2H` physical pitch,
 * while dW13-A consumes a logical `2I` primary plane with that same pitch and
 * a compact `2I` residual plane. Prefix metadata must already be immutable.
 * The callback may
 * synchronize either the full grid or a selected CTA subset.  Its A edge must
 * remain because the retired A source aliases B's destination.  Its B edge is
 * the final value/scale publication edge in direct-packed mode; legacy raw
 * mode invokes one additional edge after scale compaction.  Thus the
 * subsequent TMA/UMMA body observes a complete batch in either mode.
 */
template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          bool kDirectPackedScales,
          typename GridBarrier>
CUTLASS_DEVICE __noinline__ uint32_t
prepare_k3_mxfp8_three_term_wgrad_operands_subset(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const uint32_t source_a_width,
        const uint32_t destination_a_primary_row_stride,
        const uint32_t destination_a_residual_row_stride,
        const cutlass::bfloat16_t* source_b,
        const uint32_t source_b_width,
        const uint32_t destination_b_primary_row_stride,
        const uint32_t destination_b_residual_row_stride,
        const K3MxFp8WgradScaleStorage& a_scale_storage,
        const K3MxFp8WgradScaleStorage& b_scale_storage,
        const uint32_t k_capacity,
        int* prefix_storage,
        uint8_t* a_primary_values,
        uint8_t* a_residual_values,
        uint8_t* b_primary_values,
        uint8_t* b_residual_values,
        const cute::TmaDescriptor& a_primary_map,
        const cute::TmaDescriptor& a_residual_map,
        const cute::TmaDescriptor& b_primary_map,
        const cute::TmaDescriptor& b_residual_map,
        uint8_t* smem_buffer,
        const uint32_t producer_cta_idx,
        const uint32_t num_producer_ctas,
        GridBarrier grid_barrier) {
    using Prefix = K3MxFp8WgradPrefixLayout<
        kNumExperts, kK3MaxBackwardRanges>;
    auto* grouped_layout = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kGroupedLayout);
    auto* value_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kValuePrefix);
    auto* scale_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kScalePrefix);
    auto* physical_range_prefix = reinterpret_cast<uint32_t*>(
        prefix_storage + Prefix::kPhysicalRangePrefix);

    DG_DEVICE_ASSERT(
        num_producer_ctas > 0u && producer_cta_idx < num_producer_ctas);
    const uint32_t total_k = value_prefix[kNumExperts];
    DG_DEVICE_ASSERT(total_k <= k_capacity);
    DG_DEVICE_ASSERT(
        total_k == 0u ||
        (a_primary_values != nullptr && a_residual_values != nullptr &&
         b_primary_values != nullptr && b_residual_values != nullptr));
    const uint32_t total_scale_rows = scale_prefix[kNumExperts];
    DG_DEVICE_ASSERT(
        a_scale_storage.packed_row_capacity >= total_scale_rows &&
        b_scale_storage.packed_row_capacity >= total_scale_rows);
    DG_DEVICE_ASSERT(
        total_scale_rows == 0u ||
        (a_scale_storage.primary_packed != nullptr &&
         a_scale_storage.residual_packed != nullptr &&
         b_scale_storage.primary_packed != nullptr &&
         b_scale_storage.residual_packed != nullptr));
    if constexpr (!kDirectPackedScales) {
        const uint32_t required_raw_rows =
            k3_mxfp8_wgrad_raw_scale_rows(total_k);
        DG_DEVICE_ASSERT(
            a_scale_storage.raw_row_capacity >= required_raw_rows &&
            b_scale_storage.raw_row_capacity >= required_raw_rows);
        DG_DEVICE_ASSERT(
            total_k == 0u ||
            (a_scale_storage.primary_raw != nullptr &&
             a_scale_storage.residual_raw != nullptr &&
             b_scale_storage.primary_raw != nullptr &&
             b_scale_storage.residual_raw != nullptr));
    }
    detail::k3_mxfp8_produce_operand<
        kNumExperts, kBlockM, kK3MaxBackwardRanges,
        kNumSMs, kNumThreads, kDirectPackedScales, false>(
            source_a, source_a_width,
            destination_a_primary_row_stride,
            destination_a_residual_row_stride,
            k_capacity, total_k, 0u,
            expert_counts, backward_ranges, value_prefix,
            physical_range_prefix,
            a_primary_values, a_residual_values,
            kDirectPackedScales
                ? reinterpret_cast<uint8_t*>(
                      a_scale_storage.primary_packed)
                : a_scale_storage.primary_raw,
            kDirectPackedScales
                ? reinterpret_cast<uint8_t*>(
                      a_scale_storage.residual_packed)
                : a_scale_storage.residual_raw,
            a_primary_map, a_residual_map,
            producer_cta_idx, num_producer_ctas, smem_buffer);
    // The fused parent deliberately reuses operand A's retired BF16 source
    // as operand B's FP8 destination.  Retire every CTA's A load and publish
    // its value/scale stores before any CTA starts B; a CTA-local return from
    // k3_mxfp8_produce_operand is insufficient because task counts differ.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence();
    grid_barrier();
    detail::k3_mxfp8_produce_operand<
        kNumExperts, kBlockM, kK3MaxBackwardRanges,
        kNumSMs, kNumThreads, kDirectPackedScales, false>(
            source_b, source_b_width,
            destination_b_primary_row_stride,
            destination_b_residual_row_stride,
            k_capacity, total_k, 0u,
            expert_counts, backward_ranges, value_prefix,
            physical_range_prefix,
            b_primary_values, b_residual_values,
            kDirectPackedScales
                ? reinterpret_cast<uint8_t*>(
                      b_scale_storage.primary_packed)
                : b_scale_storage.primary_raw,
            kDirectPackedScales
                ? reinterpret_cast<uint8_t*>(
                      b_scale_storage.residual_packed)
                : b_scale_storage.residual_raw,
            b_primary_map, b_residual_map,
            producer_cta_idx, num_producer_ctas, smem_buffer);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __threadfence();
    grid_barrier();

    if constexpr (kDirectPackedScales) {
        return total_k;
    } else {
        #pragma unroll
        for (uint32_t plane = 0; plane < 4u; ++plane) {
            const bool is_a = plane < 2u;
            const bool is_residual = (plane & 1u) != 0u;
            const uint32_t feature_width =
                is_a ? source_a_width : source_b_width;
            const auto& scale_storage =
                is_a ? a_scale_storage : b_scale_storage;
            const uint64_t tasks =
                static_cast<uint64_t>(total_scale_rows) * feature_width;
            for (uint64_t task =
                     static_cast<uint64_t>(producer_cta_idx) *
                         kNumThreads + threadIdx.x;
                 task < tasks;
                 task += static_cast<uint64_t>(num_producer_ctas) *
                     kNumThreads) {
                const uint32_t sf_row = task / feature_width;
                const uint32_t feature = task -
                    static_cast<uint64_t>(sf_row) * feature_width;
                const uint32_t expert =
                    detail::k3_mxfp8_find_prefix_owner<
                        kNumExperts, kK3MaxBackwardRanges>(
                            scale_prefix, sf_row);
                const uint32_t local_sf_row =
                    sf_row - scale_prefix[expert];
                const uint32_t num_groups =
                    math::ceil_div(grouped_layout[expert], 32u);
                const uint32_t raw_group_begin =
                    value_prefix[expert] / 32u;
                auto* raw = is_residual
                    ? scale_storage.residual_raw
                    : scale_storage.primary_raw;
                uint32_t packed = 0;
                #pragma unroll
                for (uint32_t byte = 0; byte < 4; ++byte) {
                    const uint32_t local_group =
                        local_sf_row * 4u + byte;
                    const uint8_t scale = local_group < num_groups
                        ? raw[(static_cast<uint64_t>(raw_group_begin) +
                               local_group) * feature_width + feature]
                        : static_cast<uint8_t>(0x7f);
                    packed |=
                        static_cast<uint32_t>(scale) << (byte * 8u);
                }
                auto* compact = is_residual
                    ? scale_storage.residual_packed
                    : scale_storage.primary_packed;
                const uint32_t packed_scale_feature =
                    k3_mxfp8_utccp_scale_feature(feature);
                compact[
                    static_cast<uint64_t>(sf_row) * feature_width +
                    packed_scale_feature] = packed;
            }
        }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        grid_barrier();
        return total_k;
    }
}

/** Full-grid compatibility wrapper for the serial fused/standalone path. */
template <uint32_t kNumExperts, uint32_t kBlockM,
          uint32_t kNumSMs, uint32_t kNumThreads,
          typename GridBarrier>
CUTLASS_DEVICE __noinline__ uint32_t
prepare_k3_mxfp8_three_term_wgrad_operands(
        const int* expert_counts,
        const K3BackwardRangeSet& backward_ranges,
        const cutlass::bfloat16_t* source_a,
        const uint32_t source_a_width,
        const uint32_t destination_a_primary_row_stride,
        const uint32_t destination_a_residual_row_stride,
        const cutlass::bfloat16_t* source_b,
        const uint32_t source_b_width,
        const uint32_t destination_b_primary_row_stride,
        const uint32_t destination_b_residual_row_stride,
        const K3MxFp8WgradScaleStorage& a_scale_storage,
        const K3MxFp8WgradScaleStorage& b_scale_storage,
        const uint32_t k_capacity,
        int* prefix_storage,
        uint8_t* a_primary_values,
        uint8_t* a_residual_values,
        uint8_t* b_primary_values,
        uint8_t* b_residual_values,
        const cute::TmaDescriptor& a_primary_map,
        const cute::TmaDescriptor& a_residual_map,
        const cute::TmaDescriptor& b_primary_map,
        const cute::TmaDescriptor& b_residual_map,
        uint8_t* smem_buffer,
        GridBarrier grid_barrier) {
    build_k3_mxfp8_three_term_wgrad_metadata<kNumExperts, kBlockM>(
        expert_counts, backward_ranges, prefix_storage,
        static_cast<uint32_t>(blockIdx.x));
    grid_barrier();
    return prepare_k3_mxfp8_three_term_wgrad_operands_subset<
        kNumExperts, kBlockM, kNumSMs, kNumThreads, false>(
            expert_counts, backward_ranges,
            source_a, source_a_width,
            destination_a_primary_row_stride,
            destination_a_residual_row_stride,
            source_b, source_b_width,
            destination_b_primary_row_stride,
            destination_b_residual_row_stride,
            a_scale_storage, b_scale_storage,
            k_capacity, prefix_storage,
            a_primary_values, a_residual_values,
            b_primary_values, b_residual_values,
            a_primary_map, a_residual_map,
            b_primary_map, b_residual_map,
            smem_buffer, static_cast<uint32_t>(blockIdx.x), kNumSMs,
            grid_barrier);
}

#endif  // DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD

/** Fixed cluster-2 geometry shared by K3 dW2 and paired dW13.
 *
 * `grouped_layout` supplies the externally padded K extent of each expert.
 * Extents must be group-32 aligned; K3's native 192-row expert blocks satisfy
 * this directly.  Value-plane prefixes remain compact, while each expert's
 * scale plane independently reserves ceil(K / 128) packed uint32 rows.  The
 * producer is responsible for that scale-row padding and for zeroing unused
 * scale bytes; this body never allocates or repacks global storage.
 */
template <uint32_t kShapeM_, uint32_t kShapeN_, uint32_t kNumGroups_,
          uint32_t kNumSMs_, bool kWithAccumulation_ = false>
struct Sm100K3MxFp8ThreeTermWgradConfig {
    static constexpr uint32_t kShapeM = kShapeM_;
    static constexpr uint32_t kShapeN = kShapeN_;
    static constexpr uint32_t kNumGroups = kNumGroups_;
    static constexpr uint32_t kNumSMs = kNumSMs_;
    static constexpr bool kWithAccumulation = kWithAccumulation_;

    static constexpr uint32_t kBlockM = 256;
    static constexpr uint32_t kBlockN = 128;
    static constexpr uint32_t kBlockK = 128;
    static constexpr uint32_t kNumOutputPanels = 2;
    static constexpr uint32_t kStoreBlockM =
        kK3MxFp8WgradStoreBlockM;
    static constexpr uint32_t kNumTmaStoreStages =
        kK3MxFp8WgradNumTmaStoreStages;
    static constexpr uint32_t kKAlignment = 32;
    static constexpr uint32_t kScaleKSpan = 128;
    // Preserve two logical P00/P10/P01 triplets in flight while compacting
    // their operands to two physical slots per triplet. P00/P01 share the
    // primary-A slot and P00/P10 share the primary-B slot.
    static constexpr uint32_t kNumStages = 6;
    static constexpr uint32_t kNumOperandStages = 4;
    static constexpr uint32_t kNumMulticast = 2;
    static constexpr bool kIsMulticastOnA = true;
    static constexpr bool kSwapAB = true;
    static constexpr bool kEnsureZeroPadding = true;
    // A 128x128 output panel contributes 64 A rows per CTA.  A 128-byte
    // swizzle has a 128-element FP8 TMA atom and would therefore issue zero
    // transactions for that half-panel.  Use the 64-byte layout for every
    // exact value descriptor: A then issues one 64x128 transaction while B
    // issues two adjacent transactions into the same 128x128 stage.  Keeping
    // A and B on one layout also preserves the allocation-free dW13-B alias
    // of the dW2-A TensorMaps.
    static constexpr uint32_t kSwizzleA = 64;
    static constexpr uint32_t kSwizzleB = 64;
    static constexpr uint32_t kSwizzleCD = 128;
    static constexpr uint32_t kNumNonEpilogueThreads = 128;
    static constexpr uint32_t kNumEpilogueThreads = 128;
    static constexpr uint32_t kNumThreads =
        kNumNonEpilogueThreads + kNumEpilogueThreads;

    static_assert(kShapeM % kBlockM == 0,
                  "K3 wgrad M must be tiled exactly by 256");
    static_assert(kBlockM % kNumOutputPanels == 0,
                  "K3 wgrad output panels must exactly partition M");
    static_assert(kShapeN % kBlockN == 0,
                  "K3 wgrad N must be tiled exactly by 128");
    static_assert(kNumSMs % kNumMulticast == 0,
                  "Cluster-2 wgrad requires an even SM count");
};

using Sm100K3Ep8DW2ThreeTermWgradConfig =
    Sm100K3MxFp8ThreeTermWgradConfig<3584, 3072, 112, 148>;
using Sm100K3Ep8DW13ThreeTermWgradConfig =
    Sm100K3MxFp8ThreeTermWgradConfig<6144, 3584, 112, 148>;

/** Independent mbarrier and TMEM ownership for one embedded wgrad batch.
 *
 * CTA-specific scheduler lengths leave mbarrier generations different after
 * dW2, so a descriptor switch may never retain the live barrier objects while
 * resetting local stage counters to zero.  K3 instead invalidates dW2's
 * drained barriers, retains only its empty base-zero TMEM allocation, then
 * initializes fresh phase-zero barriers for dW13 before finally freeing TMEM.
 * Keeping allocation/free separate from init/invalidate makes that contract a
 * compile-time policy rather than an implicit caller convention.
 */
template <bool kInitializeBatchResources_, bool kReleaseBatchResources_,
          bool kSynchronizeAfterRelease_ =
              (kReleaseBatchResources_ && !kInitializeBatchResources_),
          bool kAllocateTmem_ = kInitializeBatchResources_,
          bool kFreeTmem_ = kReleaseBatchResources_>
struct Sm100K3MxFp8WgradBatchResourceHooks {
    static constexpr bool kInitializeBatchResources =
        kInitializeBatchResources_;
    static constexpr bool kReleaseBatchResources =
        kReleaseBatchResources_;
    static constexpr bool kSynchronizeAfterRelease =
        kSynchronizeAfterRelease_;
    static constexpr bool kAllocateTmem = kAllocateTmem_;
    static constexpr bool kFreeTmem = kFreeTmem_;
    static_assert(!kSynchronizeAfterRelease || kReleaseBatchResources,
                  "A post-release join requires resource release");
    static_assert(!kFreeTmem || kReleaseBatchResources,
                  "TMEM free requires drained barrier release");
};

using Sm100K3MxFp8WgradDefaultBatchResourceHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<true, true, true>;
using Sm100K3MxFp8WgradCallerManagedBatchResourceHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<false, false, false>;
template <bool kInitializeBatchResources = false>
using Sm100K3MxFp8WgradEmbeddedReleaseBatchResourceHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<
        kInitializeBatchResources, true, true>;
using Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<
        true, true, true, true, false>;
using Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<
        true, true, true, false, true>;
using Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks =
    Sm100K3MxFp8WgradBatchResourceHooks<
        true, true, true, false, false>;

static_assert(
    Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks::
        kInitializeBatchResources &&
    Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks::
        kReleaseBatchResources &&
    Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks::kAllocateTmem &&
    !Sm100K3MxFp8WgradReleaseBarriersRetainTmemHooks::kFreeTmem);
static_assert(
    Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks::
        kInitializeBatchResources &&
    Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks::
        kReleaseBatchResources &&
    !Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks::
        kAllocateTmem &&
    Sm100K3MxFp8WgradReinitializeBarriersReleaseTmemHooks::kFreeTmem);
static_assert(
    Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks::
        kInitializeBatchResources &&
    Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks::
        kReleaseBatchResources &&
    !Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks::
        kAllocateTmem &&
    !Sm100K3MxFp8WgradReinitializeBarriersRetainTmemHooks::kFreeTmem);

template <typename Config>
using Sm100K3MxFp8ThreeTermDefaultTaskProvider = sched::Scheduler<
    GemmType::KGroupedContiguous,
    Config::kBlockM, Config::kBlockN, Config::kNumGroups,
    Config::kNumMulticast, Config::kIsMulticastOnA, Config::kNumSMs,
    Config::kEnsureZeroPadding, Config::kKAlignment,
    Config::kScaleKSpan>;

#if DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD

/** Default source-retirement hook for standalone and serial callers. */
struct Sm100K3MxFp8NoInputTileRetired {
    CUTLASS_DEVICE void operator()(
            uint32_t, uint32_t, uint32_t) const {}
};

/** Default-disabled lifecycle for rolling, group-32 A-panel inputs.
 *
 * A selected lifecycle acquire-waits every physical panel named by one
 * compact K128 scheduler tile before its first A TMA read. Its retirement
 * hook runs once per output tile only after final P01 completion. The
 * lifecycle owns compact-to-physical coordinate resolution, so the ordinary
 * compact producer pays no instructions.
 */
struct Sm100K3MxFp8NoInputPanelLifecycle {
    static constexpr bool kEnabled = false;
    static constexpr bool kAEnabled = false;
    static constexpr bool kBEnabled = false;

    template <typename... Args>
    CUTLASS_DEVICE void load_a_k128_stage(Args&&...) const {}
    template <typename... Args>
    CUTLASS_DEVICE void load_b_k128_stage(Args&&...) const {}
    CUTLASS_DEVICE void retire_a_k128_after_p01(
            uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t) const {}
    CUTLASS_DEVICE void retire_b_k128_after_p01(
            uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t) const {}
};

template <typename Callback>
struct Sm100K3MxFp8InputTileRetiredEnabled {
    static constexpr bool value = true;
};

template <>
struct Sm100K3MxFp8InputTileRetiredEnabled<
        Sm100K3MxFp8NoInputTileRetired> {
    static constexpr bool value = false;
};

/** Execute one allocation-free, cluster-2 K3 three-term MXFP8 wgrad body.
 *
 * The eight input descriptors name externally prepared group-32 E4M3 value
 * and UE8M0 scale planes.  A descriptors have logical shape [M, K], B
 * descriptors [N, K], and scale descriptors use packed 1d1d words (four
 * adjacent K-group scales per uint32) with every 128-feature tile already in
 * UTCCP-native order.  Primary and residual planes must have identical
 * layouts and grouped K prefixes.
 *
 * The caller owns `smem_buffer` and must provide at least
 * `sm100_k3_mxfp8_three_term_wgrad_smem_bytes<Config>()` bytes aligned to
 * 1024.  No global variable, CUDA allocation, workspace counter, or hidden
 * output is created.  Resource ownership is selected by `BatchResourceHooks`:
 * the default policy initializes cluster mbarriers, allocates base-zero TMEM,
 * drains the final epilogue, frees TMEM, invalidates the barriers, and performs
 * a post-free cluster join.  The K3 pair uses the split policies above: dW2
 * drains and invalidates its CTA-specific barrier generations but retains the
 * empty TMEM allocation; dW13 reinitializes barriers at phase zero, reuses
 * that allocation, then invalidates/frees both resources. A fully
 * caller-managed policy performs neither boundary and requires the enclosing
 * kernel to own both. In every case both peer CTAs must enter together, and a
 * policy that allocates TMEM requires no other live base-zero TMEM owner.
 * `InputTileRetiredCallback` runs once per cluster output tile, on one elected
 * lane of the leader CTA, after the tile's final UMMA completion is visible.
 * It may therefore publish per-expert source retirement before a later phase
 * overwrites aliased operand storage; output TMA completion is not implied.
 */
template <typename Config,
          typename TaskProvider =
              Sm100K3MxFp8ThreeTermDefaultTaskProvider<Config>,
          typename BatchResourceHooks =
              Sm100K3MxFp8WgradDefaultBatchResourceHooks,
          typename BackgroundWorkCallback,
          typename InputTileRetiredCallback =
              Sm100K3MxFp8NoInputTileRetired,
          typename InputPanelLifecycle =
              Sm100K3MxFp8NoInputPanelLifecycle>
CUTLASS_DEVICE __noinline__ void
sm100_k3_mxfp8_three_term_grouped_wgrad_body(
        int* grouped_layout,
        uint32_t shape_k,
        const cute::TmaDescriptor& tensor_map_a_primary,
        const cute::TmaDescriptor& tensor_map_a_residual,
        const cute::TmaDescriptor& tensor_map_b_primary,
        const cute::TmaDescriptor& tensor_map_b_residual,
        const cute::TmaDescriptor& tensor_map_sfa_primary,
        const cute::TmaDescriptor& tensor_map_sfa_residual,
        const cute::TmaDescriptor& tensor_map_sfb_primary,
        const cute::TmaDescriptor& tensor_map_sfb_residual,
        const cute::TmaDescriptor& tensor_map_d,
        uint8_t* smem_buffer,
        bool wait_for_primary_kernel,
        BackgroundWorkCallback background_work,
        InputTileRetiredCallback input_tile_retired =
            InputTileRetiredCallback{},
        const cute::TmaDescriptor* phase_one_tensor_maps = nullptr,
        const cute::TmaDescriptor* phase_one_tensor_map_d = nullptr,
        const InputPanelLifecycle* input_panel_lifecycle = nullptr) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)) || \
    defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;
    using input_dtype_t = cutlass::float_e4m3_t;
    using output_dtype_t = cutlass::bfloat16_t;

    constexpr uint32_t kBlockM = Config::kBlockM;
    constexpr uint32_t kBlockN = Config::kBlockN;
    constexpr uint32_t kBlockK = Config::kBlockK;
    constexpr uint32_t kNumOutputPanels = Config::kNumOutputPanels;
    constexpr uint32_t kPanelBlockM = kBlockM / kNumOutputPanels;
    constexpr uint32_t kNumStages = Config::kNumStages;
    constexpr uint32_t kNumOperandStages = Config::kNumOperandStages;
    constexpr uint32_t kNumMulticast = Config::kNumMulticast;
    constexpr uint32_t kLayoutADM = 128;
    constexpr uint32_t kUmmaM = kLayoutADM * kNumMulticast;
    // Keep one full-width tensor-core instruction per K32 quantum.  The prior
    // split issue path performed two 256x128 UMMAs for the same logical
    // 256x128 scheduler task, doubling instruction issue and synchronization
    // overhead without exposing an earlier completion edge: both panel-full
    // barriers were published only after the final product and K tile.  The
    // epilogue remains split into two 128-row panels, but both read disjoint
    // halves of this one 256-column accumulator.
    constexpr uint32_t kUmmaN = kBlockM;
    constexpr uint32_t kUmmaK = 32;
    constexpr bool kPhaseTagged = []() constexpr {
        if constexpr (requires { TaskProvider::kTaskPhaseTagged; })
            return TaskProvider::kTaskPhaseTagged;
        return false;
    }();
    constexpr bool kReadyFirst = []() constexpr {
        if constexpr (requires {
                TaskProvider::kTaskReadyFirstTaskClaim;
            })
            return TaskProvider::kTaskReadyFirstTaskClaim;
        return false;
    }();
    if constexpr (kPhaseTagged) {
        DG_DEVICE_ASSERT(
            phase_one_tensor_maps != nullptr &&
            phase_one_tensor_map_d != nullptr);
    }
    static_assert(
        InputPanelLifecycle::kEnabled ==
            (InputPanelLifecycle::kAEnabled ||
             InputPanelLifecycle::kBEnabled),
        "Input-panel lifecycle enable bit must name at least one operand");
    if constexpr (InputPanelLifecycle::kEnabled)
        DG_DEVICE_ASSERT(input_panel_lifecycle != nullptr);

    // UTCCP-native producer layout removes the auxiliary scale-transpose
    // scheduler role.  Keep value grouping, compact-scale span, and physical
    // TMA tile width independent: K3's 192-row expert tails are group-32
    // aligned but not K128 aligned.
    if constexpr (requires {
            TaskProvider::kTaskNumAuxiliarySchedulerWarps;
            TaskProvider::kTaskScaleKSpan;
            TaskProvider::kTaskTmaTileK;
            TaskProvider::kCompleteAcquireMask;
        }) {
        static_assert(
            TaskProvider::kTaskNumAuxiliarySchedulerWarps == 0u,
            "UTCCP-native MXFP8 scheduling has no auxiliary warp");
        static_assert(
            TaskProvider::kCompleteAcquireMask == 0x7ffu,
            "UTCCP-native MXFP8 scheduler requires eleven roles");
        static_assert(
            TaskProvider::kTaskKAlignment == Config::kKAlignment &&
                TaskProvider::kTaskScaleKSpan == Config::kScaleKSpan &&
                TaskProvider::kTaskTmaTileK == Config::kBlockK,
            "Exact MXFP8 scheduler value/scale/TMA K geometry drifted");
    }
    constexpr uint32_t kLoadBlockM = kPanelBlockM / kNumMulticast;
    constexpr uint32_t kMmaLoadBlockM = kBlockM / kNumMulticast;
    constexpr uint32_t kLoadBlockN = kBlockN;
    constexpr uint32_t kNumProducts = 3;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kSFBlockM =
        math::constexpr_align(kPanelBlockM, kNumUTCCPAlignedElems);
    constexpr uint32_t kSFBlockN =
        math::constexpr_align(kBlockN, kNumUTCCPAlignedElems);

    // One full 256-column accumulator plus the unchanged 24 scale columns
    // fits in the same 512-column architectural allocation.  Two independent
    // full-width accumulators would not fit, so the two output barriers name
    // disjoint epilogue views rather than independent accumulator stages.
    constexpr uint32_t kNumEpilogueStages = kNumOutputPanels;
    constexpr uint32_t kNumTMAStoreStages =
        Config::kNumTmaStoreStages;
    // Two 32-row slices preserve the proven 128-column TMA store while
    // halving TMEM-load, store-barrier, and epilogue-loop iterations versus
    // the original 16-row slice.  The resulting exact-body shared-memory
    // footprint remains below the SM103 per-CTA limit.
    constexpr uint32_t kStoreBlockM = Config::kStoreBlockM;
    constexpr uint32_t kStoreBlockN = kBlockN;
    constexpr uint32_t kNumUMMAStoreThreads =
        Config::kNumEpilogueThreads;

    constexpr uint32_t kSmemCDSizePerStage =
        kStoreBlockM * kStoreBlockN * sizeof(output_dtype_t);
    constexpr uint32_t kSmemCDSize =
        kSmemCDSizePerStage * kNumTMAStoreStages;
    constexpr uint32_t kSmemASizePerStage =
        kLoadBlockM * kBlockK * sizeof(input_dtype_t);
    constexpr uint32_t kSmemBSizePerStage =
        kLoadBlockN * kBlockK * sizeof(input_dtype_t);
    constexpr uint32_t kSmemSFASizePerStage =
        kSFBlockM * sizeof(uint32_t);
    constexpr uint32_t kSmemSFBSizePerStage =
        kSFBlockN * sizeof(uint32_t);

    constexpr uint32_t kNumAOperandStages =
        kNumOperandStages * kNumOutputPanels;
    constexpr uint32_t kNumAccumTmemCols = kUmmaN;
    constexpr uint32_t kNumSFATmemCols = kSFBlockM / 32;
    constexpr uint32_t kNumSFBTmemCols = kSFBlockN / 32;
    // P10 reuses B0/SFB0 and P01 reuses A0/SFA0 from P00.  A scales are
    // panel-local; B scales are shared by both panels.  The resulting scale
    // footprint is still 2 * (2 * 4) + 2 * 4 = 24 columns, so the complete
    // logical footprint remains 280 columns and rounds to the same 512-column
    // architectural allocation as the former single-accumulator body.
    constexpr uint32_t kTmemStartColOfSFA0 = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB0 =
        kTmemStartColOfSFA0 +
        kNumOutputPanels * kNumSFATmemCols;
    constexpr uint32_t kTmemStartColOfSFA1 =
        kTmemStartColOfSFB0 + kNumSFBTmemCols;
    constexpr uint32_t kTmemStartColOfSFB1 =
        kTmemStartColOfSFA1 +
        kNumOutputPanels * kNumSFATmemCols;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<
            kTmemStartColOfSFB1 + kNumSFBTmemCols>();

    DG_STATIC_ASSERT(kNumMulticast == 2,
                     "K3 three-term wgrad is cluster-2 only");
    DG_STATIC_ASSERT(
        kNumOutputPanels == 2u && kPanelBlockM == 128u &&
            kNumEpilogueStages == 2u &&
            kNumAccumTmemCols == 256u,
        "K3 exact wgrad requires one 256-column accumulator with two "
        "epilogue panels");
    DG_STATIC_ASSERT(
        kNumOutputPanels * kSmemSFASizePerStage ==
            kBlockM * sizeof(uint32_t),
        "The two panel-local SFA slots must form one contiguous 256-row "
        "descriptor tile");
    DG_STATIC_ASSERT(
        Config::kSwizzleA == 64u && Config::kSwizzleB == 64u &&
            kLoadBlockM == 64u && kMmaLoadBlockM == 128u &&
            kLoadBlockN == 128u,
        "Wide-UMMA exact wgrad requires two contiguous 64-byte A atoms");
    DG_STATIC_ASSERT(
        kStoreBlockM == 32u && kNumTMAStoreStages == 2u &&
            kBlockM % kStoreBlockM == 0u,
        "K3 exact wgrad requires two 32-row output-store stages");
    DG_STATIC_ASSERT(Config::kSwapAB && Config::kIsMulticastOnA,
                     "K3 three-term wgrad requires swap-AB cluster geometry");
    DG_STATIC_ASSERT(kBlockK == 128 && kUmmaK == 32,
                     "K3 MXFP8 wgrad requires four group-32 UMMAs per K tile");
    DG_STATIC_ASSERT(Config::kKAlignment % kUmmaK == 0,
                     "K3 grouped prefixes must preserve group-32 MXFP8");
    DG_STATIC_ASSERT(Config::kScaleKSpan == 4 * kUmmaK,
                     "One packed scale row must cover four group-32 UMMAs");
    DG_STATIC_ASSERT(kNumStages >= 3 && kNumStages % kNumProducts == 0,
                     "The retained P00/P10/P01 ring needs stage triplets");
    DG_STATIC_ASSERT(
        kNumOperandStages == 2 * (kNumStages / kNumProducts),
        "Each in-flight product triplet needs two operand slots");
    DG_STATIC_ASSERT(kNumTmemCols <= 512,
                     "K3 wgrad TMEM footprint exceeds one cluster allocation");
    DG_STATIC_ASSERT(TaskProvider::kTaskGemmType ==
                         GemmType::KGroupedContiguous,
                     "K3 wgrad requires a K-grouped task provider");
    DG_STATIC_ASSERT(TaskProvider::kTaskBlockM == kBlockM &&
                         TaskProvider::kTaskBlockN == kBlockN,
                     "Task provider tile shape does not match K3 wgrad");
    DG_STATIC_ASSERT(TaskProvider::kTaskNumMulticast == 2 &&
                         TaskProvider::kTaskIsMulticastOnA,
                     "Task provider must retain the cluster-2 geometry");
    // Warps 0/1 remain TMA/MMA.  Warps 2/3 are intentionally idle after the
    // software transpose is removed; retaining four non-epilogue warps keeps
    // epilogue ownership at warps 4..7 and background ownership at warp 8+.
    DG_STATIC_ASSERT(
        Config::kNumNonEpilogueThreads == 4u * 32u &&
            Config::kNumEpilogueThreads == 4u * 32u,
        "Direct-scale wgrad must not reclassify idle warp 2 as epilogue");

    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<output_dtype_t*>(
            smem_buffer + i * kSmemCDSizePerStage);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<input_dtype_t*>(
            smem_buffer + kSmemCDSize + i * kSmemASizePerStage);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<input_dtype_t*>(
            smem_buffer + kSmemCDSize +
            kNumAOperandStages * kSmemASizePerStage +
            i * kSmemBSizePerStage);
    });
    auto* sf_start_ptr =
        reinterpret_cast<uint8_t*>(smem_b[kNumOperandStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(
            sf_start_ptr + i * kSmemSFASizePerStage);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(
            sf_start_ptr + kNumAOperandStages * kSmemSFASizePerStage +
            i * kSmemSFBSizePerStage);
    });

    auto* barrier_start_ptr =
        reinterpret_cast<Barrier*>(smem_sfb[kNumOperandStages]);
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + i;
    });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return barrier_start_ptr + kNumStages + i;
    });
    auto tmem_full_barriers =
        utils::PatternVisitor([=](const uint32_t& i) {
            return barrier_start_ptr + 2 * kNumStages + i;
        });
    auto tmem_empty_barriers =
        utils::PatternVisitor([=](const uint32_t& i) {
            return barrier_start_ptr + 2 * kNumStages +
                   kNumEpilogueStages + i;
        });
    auto* tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + 2 * kNumStages + 2 * kNumEpilogueStages);

    if constexpr (
        BatchResourceHooks::kInitializeBatchResources ||
        BatchResourceHooks::kAllocateTmem)
        comm::cluster_sync_with_relaxed_arrive();

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a_primary);
        cute::prefetch_tma_descriptor(&tensor_map_a_residual);
        cute::prefetch_tma_descriptor(&tensor_map_b_primary);
        cute::prefetch_tma_descriptor(&tensor_map_b_residual);
        cute::prefetch_tma_descriptor(&tensor_map_sfa_primary);
        cute::prefetch_tma_descriptor(&tensor_map_sfa_residual);
        cute::prefetch_tma_descriptor(&tensor_map_sfb_primary);
        cute::prefetch_tma_descriptor(&tensor_map_sfb_residual);
        cute::prefetch_tma_descriptor(&tensor_map_d);
        if constexpr (kPhaseTagged) {
            #pragma unroll
            for (uint32_t map = 0u; map < 8u; ++map)
                cute::prefetch_tma_descriptor(
                    phase_one_tensor_maps + map);
            cute::prefetch_tma_descriptor(phase_one_tensor_map_d);
        }
    }

    if constexpr (
        BatchResourceHooks::kInitializeBatchResources ||
        BatchResourceHooks::kAllocateTmem) {
        if constexpr (BatchResourceHooks::kInitializeBatchResources) {
          if (warp_idx == 1 && cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++i) {
                // Both CTAs contribute one scheduler/TMA role.  Cooperative
                // 2SM TMA completion is charged to the leader barrier, while
                // the peer contributes the second zero-byte arrival.
                full_barriers[i]->init(kNumMulticast);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                tmem_full_barriers[i]->init(1);
                tmem_empty_barriers[i]->init(
                    kNumMulticast * kNumUMMAStoreThreads);
            }
            cutlass::arch::fence_barrier_init();
          }
        }
        if constexpr (BatchResourceHooks::kAllocateTmem) {
          if (warp_idx == 2)
            Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
        }
        comm::cluster_sync_with_relaxed_arrive();
    }

#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
    if ((blockIdx.x == 132u || blockIdx.x == 133u) && threadIdx.x == 0u) {
        printf(
            "K3_EXACT_RING_BODY milestone=init_sync block=%u thread=%u "
            "leader=%u tmem=%u\n",
            static_cast<uint32_t>(blockIdx.x),
            static_cast<uint32_t>(threadIdx.x),
            static_cast<uint32_t>(is_leader_cta),
            ptx::ld_shared(tmem_ptr_in_smem));
    }
#endif

    // A standalone dependent launch waits for its primary kernel here.  An
    // embedded caller has already established operand readiness and must not
    // execute a grid-dependency wait from the middle of a persistent kernel.
    if (wait_for_primary_kernel)
        cudaGridDependencySynchronize();

    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    // Materialize these values in device-local storage.  Scheduler's legacy
    // constructor takes const references, which would otherwise ODR-use the
    // config's host-side constexpr members under NVCC.
    const uint32_t shape_m = Config::kShapeM;
    const uint32_t shape_n = Config::kShapeN;
    TaskProvider scheduler(shape_m, shape_n, shape_k, grouped_layout);

#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
    if ((blockIdx.x == 132u || blockIdx.x == 133u) && threadIdx.x == 0u) {
        if constexpr (requires {
                scheduler.task_cursor;
                scheduler.task_limit;
                scheduler.cluster_mailbox;
                scheduler.state_ready_epoch;
                scheduler.expected_state_epoch;
            }) {
            printf(
                "K3_EXACT_RING_BODY milestone=scheduler_enter block=%u "
                "thread=%u cursor=%u limit=%u mailbox_mask=0x%08x "
                "mailbox_first=%u mailbox_count=%u mailbox_sequence=%u "
                "state_epoch=0x%08x expected_epoch=0x%08x\n",
                static_cast<uint32_t>(blockIdx.x),
                static_cast<uint32_t>(threadIdx.x),
                ptx::ld_acq(scheduler.task_cursor), scheduler.task_limit,
                ptx::ld_acq(scheduler.cluster_mailbox),
                ptx::ld_acq(scheduler.cluster_mailbox + 1u),
                ptx::ld_acq(scheduler.cluster_mailbox + 2u),
                ptx::ld_acq(scheduler.cluster_mailbox + 3u),
                scheduler.state_ready_epoch == nullptr
                    ? 0u : ptx::ld_acq(scheduler.state_ready_epoch),
                scheduler.expected_state_epoch);
        } else {
            printf(
                "K3_EXACT_RING_BODY milestone=scheduler_enter block=%u "
                "thread=%u dynamic=0\n",
                static_cast<uint32_t>(blockIdx.x),
                static_cast<uint32_t>(threadIdx.x));
        }
    }
#endif

    uint32_t stage_idx = 0;
    uint32_t phase = 0;
    const auto advance_pipeline = [&]() {
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    if (warp_idx == 0 && cute::elect_one_sync()) {
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            if ((blockIdx.x == 132u || blockIdx.x == 133u) &&
                scheduler.current_iter == 0) {
                if constexpr (requires {
                        scheduler.batch_sequence;
                        scheduler.batch_first;
                        scheduler.batch_count;
                    }) {
                    printf(
                        "K3_EXACT_RING_BODY milestone=tma_task block=%u "
                        "thread=%u group=%u m=%u n=%u shape_k=%u "
                        "k_cumsum=%u batch_sequence=%u batch_first=%u "
                        "batch_count=%u\n",
                        static_cast<uint32_t>(blockIdx.x),
                        static_cast<uint32_t>(threadIdx.x),
                        scheduler.current_group_idx, m_block_idx, n_block_idx,
                        scheduler.current_shape_k, scheduler.current_k_cumsum,
                        scheduler.batch_sequence, scheduler.batch_first,
                        scheduler.batch_count);
                } else {
                    printf(
                        "K3_EXACT_RING_BODY milestone=tma_task block=%u "
                        "thread=%u group=%u m=%u n=%u shape_k=%u "
                        "k_cumsum=%u dynamic=0\n",
                        static_cast<uint32_t>(blockIdx.x),
                        static_cast<uint32_t>(threadIdx.x),
                        scheduler.current_group_idx, m_block_idx, n_block_idx,
                        scheduler.current_shape_k,
                        scheduler.current_k_cumsum);
                }
            }
#endif
            if constexpr (kPhaseTagged || kReadyFirst) {
                // The ready-first scheduler acquires a generic-proxy release
                // published only after the producer's global TMA stores have
                // completed and, for the 64-feature producer, after every
                // generic scale-word owner has release-fenced its stores.
                // Bridge that acquire into the async proxy before issuing
                // consumer TMA loads; otherwise a just-published value/scale
                // panel can still be observed through an older async view.
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
            const uint32_t num_k_blocks =
                math::ceil_div(scheduler.current_shape_k, kBlockK);
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks; ++k_block_idx) {
                #pragma unroll
                for (uint32_t product_idx = 0;
                     product_idx < kNumProducts; ++product_idx) {
                    // One base-stage completion proves that P00, P10, and P01
                    // have all retired: they update the same accumulator in
                    // issue order, and only P01 releases the triplet base.
                    // The two residual-only stages therefore need no separate
                    // loader wait before the triplet is reused.
                    if (product_idx == 0u)
                        empty_barriers[stage_idx]->wait(phase ^ 1);

                    const uint32_t base_m_idx =
                        scheduler.template get_global_idx<
                            false, sched::IndexType::MN>(
                                shape_m, kBlockM, m_block_idx);
                    // Both prepared operands are compact [MN, total_K]
                    // matrices. Expert selection lives solely in k_idx; an
                    // MN group offset would incorrectly require an E-times
                    // wider sparse descriptor.
                    const uint32_t n_idx =
                        scheduler.template get_global_idx<
                            false, sched::IndexType::MN>(
                                shape_n, kBlockN, n_block_idx,
                                m_block_idx);
                    const uint32_t k_idx =
                        scheduler.template get_global_idx<
                            true, sched::IndexType::K>(
                                shape_k, kBlockK, k_block_idx, m_block_idx);
                    const bool use_residual_a =
                        product_idx == static_cast<uint32_t>(
                            K3MxFp8WgradProduct::P10);
                    const bool use_residual_b =
                        product_idx == static_cast<uint32_t>(
                            K3MxFp8WgradProduct::P01);
                    const uint32_t a_operand_base_stage_idx =
                        k3_mxfp8_wgrad_operand_stage(
                            stage_idx, use_residual_a);
                    const uint32_t b_operand_stage_idx =
                        k3_mxfp8_wgrad_operand_stage(
                            stage_idx, use_residual_b);
                    const uint32_t sfb_n_idx = n_block_idx * kBlockN;
                    const uint32_t sf_k_idx =
                        scheduler.template get_global_idx<
                            true, sched::IndexType::SF_K>(
                                math::ceil_div(
                                    shape_k, Config::kScaleKSpan), 1,
                                k_block_idx, m_block_idx);
                    uint32_t expected_bytes = 0u;
                    if (!use_residual_b) {
                        const auto* map_a = use_residual_a
                            ? &tensor_map_a_residual
                            : &tensor_map_a_primary;
                        const auto* map_sfa = use_residual_a
                            ? &tensor_map_sfa_residual
                            : &tensor_map_sfa_primary;
                        if constexpr (kPhaseTagged) {
                            if (scheduler.current_wgrad_phase != 0u) {
                                map_a = phase_one_tensor_maps +
                                    (use_residual_a ? 1u : 0u);
                                map_sfa = phase_one_tensor_maps +
                                    (use_residual_a ? 5u : 4u);
                            }
                        }
                        // The host A-scale descriptor has one 256-element
                        // tile.  Load it once into the first of the two
                        // contiguous 128-element SFA slots; issuing two
                        // panel-local 128-element copies against that
                        // descriptor leaves the transaction/barrier byte
                        // contract undefined.  Value loads remain split to
                        // retain the cluster-2 64-row CTA geometry.
                        const uint32_t a_scale_stage_idx =
                            a_operand_base_stage_idx * kNumOutputPanels;
                        const uint32_t sfa_m_idx =
                            m_block_idx * kBlockM;
                        const uint32_t a_operand_stage_idx =
                            a_operand_base_stage_idx * kNumOutputPanels;
                        const uint32_t m_idx =
                            base_m_idx + cute::block_rank_in_cluster() *
                                kMmaLoadBlockM;
                        if constexpr (InputPanelLifecycle::kAEnabled) {
                            const uint32_t remaining_k =
                                scheduler.current_shape_k -
                                k_block_idx * kBlockK;
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                            if ((blockIdx.x == 132u ||
                                 blockIdx.x == 133u) &&
                                scheduler.current_iter == 0 &&
                                k_block_idx == 0u && product_idx == 0u) {
                                printf(
                                    "K3_EXACT_RING_BODY milestone=a_load_begin "
                                    "block=%u thread=%u group=%u m=%u n=%u "
                                    "k=%u valid_k=%u\n",
                                    static_cast<uint32_t>(blockIdx.x),
                                    static_cast<uint32_t>(threadIdx.x),
                                    scheduler.current_group_idx, m_idx,
                                    n_idx / kBlockN, k_idx,
                                    remaining_k < kBlockK
                                        ? remaining_k : kBlockK);
                            }
#endif
                            input_panel_lifecycle->load_a_k128_stage(
                                scheduler.current_group_idx,
                                m_idx, base_m_idx, n_idx / kBlockN,
                                k_idx,
                                remaining_k < kBlockK
                                    ? remaining_k : kBlockK,
                                use_residual_a,
                                map_a, map_sfa, sf_k_idx,
                                full_barriers[stage_idx],
                                reinterpret_cast<uint8_t*>(
                                    smem_a[a_operand_stage_idx]),
                                smem_sfa[a_scale_stage_idx],
                                expected_bytes);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                            if ((blockIdx.x == 132u ||
                                 blockIdx.x == 133u) &&
                                scheduler.current_iter == 0 &&
                                k_block_idx == 0u && product_idx == 0u) {
                                printf(
                                    "K3_EXACT_RING_BODY milestone=a_load_end "
                                    "block=%u thread=%u expected_bytes=%u\n",
                                    static_cast<uint32_t>(blockIdx.x),
                                    static_cast<uint32_t>(threadIdx.x),
                                    expected_bytes);
                            }
#endif
                        } else {
                            tma::copy<kBlockM, 1, 0>(
                                map_sfa, full_barriers[stage_idx],
                                smem_sfa[a_scale_stage_idx],
                                sfa_m_idx, sf_k_idx, kNumMulticast);
                            expected_bytes +=
                                kNumOutputPanels * kSmemSFASizePerStage;

                            // Keep the host's 64-byte descriptor atom, but
                            // stage the two atoms in CTA-major order.
                            tma::copy<
                                kMmaLoadBlockM, kBlockK,
                                Config::kSwizzleA, input_dtype_t>(
                                    map_a, full_barriers[stage_idx],
                                    smem_a[a_operand_stage_idx],
                                    m_idx, k_idx, kNumMulticast);
                            expected_bytes +=
                                kNumOutputPanels * kSmemASizePerStage;
                        }
                    }
                    if (!use_residual_a) {
                        const auto* map_b = use_residual_b
                            ? &tensor_map_b_residual
                            : &tensor_map_b_primary;
                        const auto* map_sfb = use_residual_b
                            ? &tensor_map_sfb_residual
                            : &tensor_map_sfb_primary;
                        if constexpr (kPhaseTagged) {
                            if (scheduler.current_wgrad_phase != 0u) {
                                map_b = phase_one_tensor_maps +
                                    (use_residual_b ? 3u : 2u);
                                map_sfb = phase_one_tensor_maps +
                                    (use_residual_b ? 7u : 6u);
                            }
                        }
                        if constexpr (InputPanelLifecycle::kBEnabled) {
                            const uint32_t remaining_k =
                                scheduler.current_shape_k -
                                k_block_idx * kBlockK;
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                            if ((blockIdx.x == 132u ||
                                 blockIdx.x == 133u) &&
                                scheduler.current_iter == 0 &&
                                k_block_idx == 0u && product_idx == 0u) {
                                printf(
                                    "K3_EXACT_RING_BODY milestone=b_load_begin "
                                    "block=%u thread=%u group=%u m=%u n=%u "
                                    "k=%u valid_k=%u\n",
                                    static_cast<uint32_t>(blockIdx.x),
                                    static_cast<uint32_t>(threadIdx.x),
                                    scheduler.current_group_idx, base_m_idx,
                                    n_idx, k_idx,
                                    remaining_k < kBlockK
                                        ? remaining_k : kBlockK);
                            }
#endif
                            input_panel_lifecycle->load_b_k128_stage(
                                scheduler.current_group_idx,
                                base_m_idx, n_idx, k_idx,
                                remaining_k < kBlockK
                                    ? remaining_k : kBlockK,
                                use_residual_b,
                                map_b, map_sfb, sf_k_idx,
                                full_barriers[stage_idx],
                                reinterpret_cast<uint8_t*>(
                                    smem_b[b_operand_stage_idx]),
                                smem_sfb[b_operand_stage_idx],
                                expected_bytes);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                            if ((blockIdx.x == 132u ||
                                 blockIdx.x == 133u) &&
                                scheduler.current_iter == 0 &&
                                k_block_idx == 0u && product_idx == 0u) {
                                printf(
                                    "K3_EXACT_RING_BODY milestone=b_load_end "
                                    "block=%u thread=%u expected_bytes=%u\n",
                                    static_cast<uint32_t>(blockIdx.x),
                                    static_cast<uint32_t>(threadIdx.x),
                                    expected_bytes);
                            }
#endif
                        } else {
                            tma::copy<
                                kLoadBlockN, kBlockK, Config::kSwizzleB,
                                input_dtype_t>(
                                    map_b, full_barriers[stage_idx],
                                    smem_b[b_operand_stage_idx], n_idx, k_idx,
                                    kNumMulticast);
                            tma::copy<kBlockN, 1, 0>(
                                map_sfb, full_barriers[stage_idx],
                                smem_sfb[b_operand_stage_idx],
                                sfb_n_idx, sf_k_idx, kNumMulticast);
                            expected_bytes +=
                                kSmemBSizePerStage + kSmemSFBSizePerStage;
                        }
                    }

                    // Preserve the exact P00/P10/P01 load set while joining
                    // both CTA-local copies at the leader barrier.  Per CTA:
                    // P00=34304 B, P10=17408 B, P01=16896 B.
                    const uint32_t expected_product_bytes =
                        product_idx == static_cast<uint32_t>(
                            K3MxFp8WgradProduct::P00)
                            ? 34304u
                            : product_idx == static_cast<uint32_t>(
                                  K3MxFp8WgradProduct::P10)
                            ? 17408u
                            : 16896u;
                    DG_DEVICE_ASSERT(expected_bytes == expected_product_bytes);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(
                            expected_bytes * kNumMulticast);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                    if ((blockIdx.x == 132u || blockIdx.x == 133u) &&
                        scheduler.current_iter == 0 &&
                        k_block_idx == 0u && product_idx == 0u) {
                        printf(
                            "K3_EXACT_RING_BODY milestone=tma_arrive "
                            "block=%u thread=%u stage=%u phase=%u bytes=%u\n",
                            static_cast<uint32_t>(blockIdx.x),
                            static_cast<uint32_t>(threadIdx.x), stage_idx,
                            phase, expected_bytes);
                    }
#endif
                    advance_pipeline();
                }
            }
        }
    } else if (warp_idx == 1 && is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            input_dtype_t, input_dtype_t, float,
            cutlass::float_ue8m0_t,
            kUmmaM, kUmmaN,
            cute::UMMA::Major::MN, cute::UMMA::Major::MN>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);
        auto a_desc = mma::sm100::make_umma_desc<
            cute::UMMA::Major::MN, kMmaLoadBlockM, kBlockK,
            Config::kSwizzleA>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<
            cute::UMMA::Major::MN, kLoadBlockN, kBlockK,
            Config::kSwizzleB>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = lane_idx < kNumOperandStages
            ? a_desc.lo +
                lane_idx * kNumOutputPanels * kSmemASizePerStage / 16
            : 0u;
        const uint32_t b_desc_lo = lane_idx < kNumOperandStages
            ? b_desc.lo + lane_idx * kSmemBSizePerStage / 16
            : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            if (blockIdx.x == 132u && scheduler.current_iter == 0 &&
                lane_idx == 0u) {
                printf(
                    "K3_EXACT_RING_BODY milestone=mma_task block=%u "
                    "thread=%u group=%u m=%u n=%u shape_k=%u\n",
                    static_cast<uint32_t>(blockIdx.x),
                    static_cast<uint32_t>(threadIdx.x),
                    scheduler.current_group_idx, m_block_idx, n_block_idx,
                    scheduler.current_shape_k);
            }
#endif
            // The wide UMMA overwrites both accumulator halves together, so
            // both split epilogues must retire before the next task starts.
            const uint32_t accum_phase = scheduler.current_iter & 1u;
            #pragma unroll
            for (uint32_t panel_idx = 0u;
                 panel_idx < kNumOutputPanels; ++panel_idx)
                tmem_empty_barriers[panel_idx]->wait(accum_phase ^ 1u);
            ptx::tcgen05_after_thread_sync();

            const uint32_t num_k_blocks =
                math::ceil_div(scheduler.current_shape_k, kBlockK);
            for (uint32_t k_block_idx = 0;
                 k_block_idx < num_k_blocks; ++k_block_idx) {
                #pragma unroll
                for (uint32_t product_idx = 0;
                     product_idx < kNumProducts; ++product_idx) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                    if (blockIdx.x == 132u &&
                        scheduler.current_iter == 0 && lane_idx == 0u &&
                        k_block_idx == 0u && product_idx == 0u) {
                        printf(
                            "K3_EXACT_RING_BODY milestone=mma_wait_begin "
                            "block=%u thread=%u stage=%u phase=%u\n",
                            static_cast<uint32_t>(blockIdx.x),
                            static_cast<uint32_t>(threadIdx.x), stage_idx,
                            phase);
                    }
#endif
                    full_barriers[stage_idx]->wait(phase);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                    if (blockIdx.x == 132u &&
                        scheduler.current_iter == 0 && lane_idx == 0u &&
                        k_block_idx == 0u && product_idx == 0u) {
                        printf(
                            "K3_EXACT_RING_BODY milestone=mma_wait_end "
                            "block=%u thread=%u stage=%u phase=%u\n",
                            static_cast<uint32_t>(blockIdx.x),
                            static_cast<uint32_t>(threadIdx.x), stage_idx,
                            phase);
                    }
#endif
                    ptx::tcgen05_after_thread_sync();

                    const uint32_t base_stage_idx =
                        stage_idx - product_idx;
                    const bool use_residual_a =
                        product_idx == static_cast<uint32_t>(
                            K3MxFp8WgradProduct::P10);
                    const bool use_residual_b =
                        product_idx == static_cast<uint32_t>(
                            K3MxFp8WgradProduct::P01);
                    const uint32_t a_base_stage_idx =
                        k3_mxfp8_wgrad_operand_stage(
                            stage_idx, use_residual_a);
                    const uint32_t b_stage_idx =
                        k3_mxfp8_wgrad_operand_stage(
                            stage_idx, use_residual_b);
                    const uint32_t b_desc_base_lo =
                        ptx::exchange(b_desc_lo, b_stage_idx);
                    const bool final_product =
                        product_idx + 1 == kNumProducts;
                    const bool final_k =
                        k_block_idx + 1 == num_k_blocks;
                    const uint32_t remaining_k =
                        scheduler.current_shape_k -
                        k_block_idx * kBlockK;
                    const uint32_t num_valid_umma = cute::min(
                        kBlockK / kUmmaK,
                        math::ceil_div(remaining_k, kUmmaK));

                    const uint32_t a_stage_idx =
                        a_base_stage_idx * kNumOutputPanels;
                    const uint32_t a_desc_base_lo =
                        ptx::exchange(a_desc_lo, a_base_stage_idx);
                    if (cute::elect_one_sync()) {
                        using UTCCP =
                            cute::SM100_UTCCP_4x32dp128bit_2cta;
                        const uint32_t tmem_sfa = use_residual_a
                            ? kTmemStartColOfSFA1
                            : kTmemStartColOfSFA0;
                        const uint32_t tmem_sfb = use_residual_b
                            ? kTmemStartColOfSFB1
                            : kTmemStartColOfSFB0;
                        if (!use_residual_b) {
                            // The descriptor-matched 256-element scale TMA
                            // occupies two contiguous 128-element slots. Copy
                            // both to the contiguous eight-column SFA matrix
                            // consumed by the wide instruction.
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < kNumOutputPanels; ++i) {
                                mma::sm100::replace_smem_desc_addr(
                                    sf_desc,
                                    smem_sfa[a_stage_idx] +
                                        i * kNumUTCCPAlignedElems);
                                UTCCP::copy(sf_desc, tmem_sfa + i * 4);
                            }
                        }
                        if (!use_residual_a) {
                            #pragma unroll
                            for (uint32_t i = 0;
                                 i < kSFBlockN /
                                         kNumUTCCPAlignedElems;
                                 ++i) {
                                mma::sm100::replace_smem_desc_addr(
                                    sf_desc,
                                    smem_sfb[b_stage_idx] +
                                        i * kNumUTCCPAlignedElems);
                                UTCCP::copy(
                                    sf_desc, tmem_sfb + i * 4);
                            }
                        }

                        using Mma =
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS;
                        const auto issue_umma =
                            [&]<uint32_t kUmmaKIdx>() {
                            constexpr uint32_t kOffset =
                                kUmmaKIdx * kUmmaK;
                            const auto runtime_instr_desc =
                                mma::sm100::
                                    make_runtime_instr_desc_with_sf_id(
                                        instr_desc,
                                        kUmmaKIdx, kUmmaKIdx);
                            a_desc.lo =
                                mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::MN,
                                    kMmaLoadBlockM,
                                    Config::kSwizzleA,
                                    input_dtype_t>(
                                        a_desc_base_lo, 0, kOffset);
                            b_desc.lo =
                                mma::sm100::advance_umma_desc_lo<
                                    cute::UMMA::Major::MN,
                                    kLoadBlockN,
                                    Config::kSwizzleB,
                                    input_dtype_t>(
                                        b_desc_base_lo, 0, kOffset);
                            const bool accumulate =
                                kUmmaKIdx > 0 || product_idx > 0 ||
                                k_block_idx > 0;
                            Mma::fma(
                                b_desc, a_desc, 0u,
                                accumulate, runtime_instr_desc,
                                tmem_sfb, tmem_sfa);
                        };
                        utils::for_each_static_prefix(
                            std::make_integer_sequence<
                                uint32_t, kBlockK / kUmmaK>(),
                            num_valid_umma, issue_umma);
                    }
                    __syncwarp();

                    if constexpr (kNumMulticast == 2) {
                        if (final_product && final_k) {
                            constexpr uint16_t kCTAMask = 0b11;
                            #pragma unroll
                            for (uint32_t panel_idx = 0u;
                                 panel_idx < kNumOutputPanels; ++panel_idx) {
                                cutlass::arch::
                                    umma_arrive_multicast_2x1SM(
                                        reinterpret_cast<uint64_t*>(
                                            tmem_full_barriers[panel_idx]),
                                        kCTAMask);
                            }
                        }
                    }
                    __syncwarp();

                    if constexpr (kNumMulticast == 2) {
                        constexpr uint16_t kCTAMask = 0b11;
                        // Commit every four UMMAs, but retire the three stage
                        // barriers in reverse order. The loader observes only
                        // the base barrier, which P01 releases after the two
                        // reused primary operands have had their final read.
                        const uint32_t release_stage_idx =
                            product_idx == 0u
                                ? base_stage_idx + 2u
                                : product_idx == 1u
                                ? base_stage_idx + 1u
                                : base_stage_idx;
                        cutlass::arch::umma_arrive_multicast_2x1SM(
                            reinterpret_cast<uint64_t*>(
                                empty_barriers[release_stage_idx]),
                            kCTAMask);
                    }
                    __syncwarp();
                    if constexpr (InputPanelLifecycle::kEnabled) {
                        // P01's empty-stage multicast is the first edge that
                        // proves both primary operands have had their final
                        // read for this K128 quantum. One cluster task is
                        // M256 x N256: both CTAs share the M256 A tile and own
                        // adjacent N128 B tiles. The sole leader therefore
                        // retires both logical N readers: A's same eight
                        // tickets receive two arrivals, while B retires eight
                        // distinct tickets. This keeps rolling reuse
                        // independent of accumulator/store lifetime.
                        if (product_idx == static_cast<uint32_t>(
                                K3MxFp8WgradProduct::P01) &&
                            cute::elect_one_sync()) {
                            const uint32_t base_m_idx =
                                scheduler.template get_global_idx<
                                    false, sched::IndexType::MN>(
                                        shape_m, kBlockM, m_block_idx);
                            const uint32_t base_n_idx =
                                n_block_idx * kBlockN;
                            const uint32_t compact_k_begin =
                                scheduler.template get_global_idx<
                                    true, sched::IndexType::K>(
                                        shape_k, kBlockK, k_block_idx,
                                        m_block_idx);
                            const uint32_t remaining_k =
                                scheduler.current_shape_k -
                                k_block_idx * kBlockK;
                            #pragma unroll
                            for (uint32_t cluster_rank = 0u;
                                 cluster_rank < kNumMulticast;
                                 ++cluster_rank) {
                                const uint32_t reader_base_n =
                                    base_n_idx + cluster_rank * kBlockN;
                                if constexpr (
                                    InputPanelLifecycle::kAEnabled) {
                                    input_panel_lifecycle
                                        ->retire_a_k128_after_p01(
                                            scheduler.current_group_idx,
                                            base_m_idx, reader_base_n,
                                            compact_k_begin,
                                            remaining_k < kBlockK
                                                ? remaining_k : kBlockK);
                                }
                                if constexpr (
                                    InputPanelLifecycle::kBEnabled) {
                                    input_panel_lifecycle
                                        ->retire_b_k128_after_p01(
                                            scheduler.current_group_idx,
                                            base_m_idx, reader_base_n,
                                            compact_k_begin,
                                            remaining_k < kBlockK
                                                ? remaining_k : kBlockK);
                                }
                            }
                        }
                    }
                    __syncwarp();
                    advance_pipeline();
                }
            }
        }
        // The final accumulator tile can still be owned by the epilogue after
        // the scheduler is exhausted.  Wait for its TMEM-empty generation
        // before any warp reaches the cluster teardown; a cluster join alone
        // does not order the epilogue's TMEM reads against Allocator::free().
        const int32_t final_iter =
            static_cast<int32_t>(scheduler.current_iter) - 1;
        if (final_iter >= 0) {
            const uint32_t final_phase =
                static_cast<uint32_t>(final_iter) & 1u;
            #pragma unroll
            for (uint32_t panel_idx = 0u;
                 panel_idx < kNumOutputPanels; ++panel_idx) {
                tmem_empty_barriers[panel_idx]->wait(final_phase);
            }
        }
    } else if (warp_idx >= Config::kNumNonEpilogueThreads / 32 &&
               warp_idx <
                   (Config::kNumNonEpilogueThreads +
                    kNumUMMAStoreThreads) / 32) {
        const uint32_t epilogue_warp_idx =
            warp_idx - Config::kNumNonEpilogueThreads / 32;
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);
        uint32_t tma_stage_idx = 0;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
            if ((blockIdx.x == 132u || blockIdx.x == 133u) &&
                scheduler.current_iter == 0 && lane_idx == 0u) {
                printf(
                    "K3_EXACT_RING_BODY milestone=epilogue_task block=%u "
                    "thread=%u warp=%u group=%u m=%u n=%u\n",
                    static_cast<uint32_t>(blockIdx.x),
                    static_cast<uint32_t>(threadIdx.x),
                    epilogue_warp_idx, scheduler.current_group_idx,
                    m_block_idx, n_block_idx);
            }
#endif
            const uint32_t accum_phase = scheduler.current_iter & 1u;
            const uint32_t base_m_idx =
                scheduler.template get_global_idx<
                    true, sched::IndexType::MN>(
                        shape_m, kBlockM, m_block_idx);
            const uint32_t base_n_idx = n_block_idx * kBlockN;
            DG_DEVICE_ASSERT(
                scheduler.get_aligned_effective_m_in_block(m_block_idx) ==
                kBlockM);
            const cute::TmaDescriptor* output_map = &tensor_map_d;
            if constexpr (kPhaseTagged) {
                if (scheduler.current_wgrad_phase != 0u)
                    output_map = phase_one_tensor_map_d;
            }

            #pragma unroll 1
            for (uint32_t panel_idx = 0u;
                 panel_idx < kNumOutputPanels; ++panel_idx) {
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                if ((blockIdx.x == 132u || blockIdx.x == 133u) &&
                    scheduler.current_iter == 0 && lane_idx == 0u &&
                    panel_idx == 0u) {
                    printf(
                        "K3_EXACT_RING_BODY milestone=epilogue_wait_begin "
                        "block=%u thread=%u warp=%u phase=%u\n",
                        static_cast<uint32_t>(blockIdx.x),
                        static_cast<uint32_t>(threadIdx.x),
                        epilogue_warp_idx, accum_phase);
                }
#endif
                tmem_full_barriers[panel_idx]->wait(accum_phase);
#if DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG
                if ((blockIdx.x == 132u || blockIdx.x == 133u) &&
                    scheduler.current_iter == 0 && lane_idx == 0u &&
                    panel_idx == 0u) {
                    printf(
                        "K3_EXACT_RING_BODY milestone=epilogue_wait_end "
                        "block=%u thread=%u warp=%u phase=%u\n",
                        static_cast<uint32_t>(blockIdx.x),
                        static_cast<uint32_t>(threadIdx.x),
                        epilogue_warp_idx, accum_phase);
                }
#endif
                ptx::tcgen05_after_thread_sync();

                // Both full barriers are ordered after the same wide-UMMA
                // completion. Input-panel retirement already happened at
                // P01's stage-release edge; retain only output/source-tile
                // callbacks here.
                if (panel_idx + 1u == kNumOutputPanels) {
                    if constexpr (
                        Sm100K3MxFp8InputTileRetiredEnabled<
                            InputTileRetiredCallback>::value) {
                        if (epilogue_warp_idx == 0 && is_leader_cta &&
                            cute::elect_one_sync()) {
                            if constexpr (requires {
                                input_tile_retired(
                                    scheduler.current_group_idx,
                                    m_block_idx, n_block_idx,
                                    scheduler.current_wgrad_phase);
                            }) {
                                input_tile_retired(
                                    scheduler.current_group_idx,
                                    m_block_idx, n_block_idx,
                                    scheduler.current_wgrad_phase);
                            } else {
                                input_tile_retired(
                                    scheduler.current_group_idx,
                                    m_block_idx, n_block_idx);
                            }
                        }
                    }
                }

                epilogue::sm100_store_cd_swap_ab<
                    kPanelBlockM, kBlockN,
                    kStoreBlockM, kStoreBlockN,
                    Config::kSwizzleCD, kNumTMAStoreStages,
                    kNumUMMAStoreThreads,
                    GemmType::KGroupedContiguous,
                    Config::kWithAccumulation,
                    output_dtype_t,
                    epilogue::transform::EpilogueIdentity>(
                        smem_cd, tma_stage_idx,
                        panel_idx * kPanelBlockM,
                        base_m_idx + panel_idx * kPanelBlockM,
                        base_n_idx,
                        scheduler.current_group_idx,
                        kPanelBlockM,
                        epilogue_warp_idx, lane_idx,
                        tmem_empty_barriers[panel_idx],
                        *output_map);
            }
        }
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
        __syncwarp();
    } else if (warp_idx >=
               (Config::kNumNonEpilogueThreads +
                Config::kNumEpilogueThreads) / 32) {
        // K3's fused parent launches 32 warps while the tensor-core body uses
        // only eight.  Communication/publication work supplied by the parent
        // runs on the otherwise-idle suffix and is joined by the body-wide
        // cluster handoff below.
        background_work(warp_idx, lane_idx);
    }

    comm::cluster_sync_with_relaxed_arrive();

    if constexpr (BatchResourceHooks::kReleaseBatchResources) {
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
            for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        tmem_full_barriers[i]));
                Barrier::invalidate(
                    reinterpret_cast<Barrier::ValueType const*>(
                        tmem_empty_barriers[i]));
            }
        }
        if constexpr (BatchResourceHooks::kFreeTmem) {
            if (warp_idx == 0)
                Allocator().free(0, kNumTmemCols);
        }
        if constexpr (BatchResourceHooks::kSynchronizeAfterRelease)
            comm::cluster_sync_with_relaxed_arrive();
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
        DG_DEVICE_ASSERT(false && "K3 three-term wgrad requires SM100+");
#endif
}

constexpr uint32_t k3_mxfp8_wgrad_producer_smem_bytes() {
    return 128u * 128u * sizeof(cutlass::bfloat16_t) +
        2u * 128u * 128u * sizeof(cutlass::float_e4m3_t) +
        sizeof(cutlass::arch::ClusterTransactionBarrier);
}

template <typename Config>
constexpr uint32_t
sm100_k3_mxfp8_three_term_wgrad_barrier_offset() {
    constexpr uint32_t kCD =
        Config::kNumTmaStoreStages * Config::kStoreBlockM *
        Config::kBlockN * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t kA =
        Config::kNumOperandStages * Config::kNumOutputPanels *
        (Config::kBlockM / Config::kNumOutputPanels /
         Config::kNumMulticast) *
        Config::kBlockK * sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kB =
        Config::kNumOperandStages * Config::kBlockN * Config::kBlockK *
        sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t kSFA =
        Config::kNumOperandStages * Config::kNumOutputPanels *
        math::constexpr_align(
            Config::kBlockM / Config::kNumOutputPanels, 128u) *
        sizeof(uint32_t);
    constexpr uint32_t kSFB =
        Config::kNumOperandStages *
        math::constexpr_align(Config::kBlockN, 128u) * sizeof(uint32_t);
    return kCD + kA + kB + kSFA + kSFB;
}

template <typename Config>
constexpr uint32_t sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset() {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    return sm100_k3_mxfp8_three_term_wgrad_barrier_offset<Config>() +
        (2u * Config::kNumStages +
         2u * Config::kNumOutputPanels) * sizeof(Barrier);
}

template <typename Config>
constexpr uint32_t sm100_k3_mxfp8_three_term_wgrad_smem_bytes() {
    return sm100_k3_mxfp8_three_term_wgrad_tmem_ptr_offset<Config>() +
        sizeof(uint32_t);
}

/** Thin standalone wrapper used only by compile/numeric harnesses.
 *
 * Production integration should call the body from the fused backward kernel
 * after retiring its previous mbarriers and TMEM allocation.
 */
template <typename Config,
          typename TaskProvider =
              Sm100K3MxFp8ThreeTermDefaultTaskProvider<Config>>
CUTLASS_GLOBAL void __launch_bounds__(Config::kNumThreads, 1)
sm100_k3_mxfp8_three_term_grouped_wgrad_impl(
        int* grouped_layout,
        uint32_t shape_k,
        const __grid_constant__ cute::TmaDescriptor tensor_map_a_primary,
        const __grid_constant__ cute::TmaDescriptor tensor_map_a_residual,
        const __grid_constant__ cute::TmaDescriptor tensor_map_b_primary,
        const __grid_constant__ cute::TmaDescriptor tensor_map_b_residual,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa_primary,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa_residual,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb_primary,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb_residual,
        const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const auto no_background_work = [] (uint32_t, uint32_t) {};
    sm100_k3_mxfp8_three_term_grouped_wgrad_body<Config, TaskProvider>(
        grouped_layout, shape_k,
        tensor_map_a_primary, tensor_map_a_residual,
        tensor_map_b_primary, tensor_map_b_residual,
        tensor_map_sfa_primary, tensor_map_sfa_residual,
        tensor_map_sfb_primary, tensor_map_sfb_residual,
        tensor_map_d, smem_buffer, true, no_background_work);
}

#endif  // DG_ENABLE_K3_MXFP8_THREE_TERM_WGRAD

}  // namespace deep_gemm

#pragma clang diagnostic pop
