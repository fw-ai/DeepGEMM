#define DG_EXPERIMENTAL_K3_READY_WGRAD 1
#define DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS 8
#define DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS 32
#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL 1
#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL 1

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

/** Compile the exact two-range terminal-dynamic4 K3 EP8 specialization.
 *
 * Multi-range preparation retains symmetric-memory communication and publishes
 * two absolute physical prefixes plus their union. Terminal dW2/dW13 then use
 * the existing 128x256x64 BF16 UMMA/TMA bodies with four-task claims.
 */
void instantiate_k3_terminal_dynamic4_two_range_bf16_sm103a() {
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
