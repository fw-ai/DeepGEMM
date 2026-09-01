#define DG_EXPERIMENTAL_K3_READY_WGRAD 1
#define DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS 8
#define DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS 32
#define DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID 1

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

/** Compile the K3 EP8 three-range BF16-dW2/MXFP8-dW13 specialization. */
void instantiate_k3_mxfp8_dw13_hybrid_parent_sm103a() {
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
        false, true, false,
        true>;
    (void)kernel;
}
