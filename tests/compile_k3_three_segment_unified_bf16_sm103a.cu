#define DG_EXPERIMENTAL_K3_READY_WGRAD 1
#define DG_EXPERIMENTAL_K3_THREE_SEGMENT_BF16_PROGRESSIVE_WGRAD 1

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

/** Compile the real EP8 three-range unified BF16 MegaMoE backward.
 *
 * The specialization retains MegaMoE's symmetric-memory communication and
 * one persistent parent kernel. Its terminal phase interleaves exact dW2 and
 * dW13 tasks in one TMA/UMMA resource lifetime and reduces true-varlen dX from
 * the full-rank publication edge on two communication warps. The range-aware
 * reducer remains inside the kernel and overlaps the unified wgrad body.
 */
void instantiate_k3_three_segment_unified_bf16_sm103a() {
    auto* kernel = &sm100_fp8_fp4_mega_moe_backward_wave_impl<
        3584, 3072, 112,
        192, 128, 128,
        256, 128,
        6, 148, 8,
        true,
        false, false, false,
        false, false, false,
        true, true,
        ActivationType::SiTU,
        4.0f, 25.0f,
        false,
        RouteWeightMode::PostDown,
        CombineOrderMode::FixedTopK,
        false, false,
        true, false, true,
        true, false, true,
        false, false, false,
        true>;
    (void)kernel;
}
