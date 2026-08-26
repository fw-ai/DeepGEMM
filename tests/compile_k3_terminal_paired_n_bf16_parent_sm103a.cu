#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL 1
#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL 1

#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

using PairedTerminalDW2 =
    sched::ExternalKGroupedTerminalDynamicRangeProvider<
        128, 256, 2, false, 148,
        3584, 3072, 4, 84,
        192, 64, 31, 144,
        4, 4, 16, true>;
using PairedTerminalDW13 =
    sched::ExternalKGroupedTerminalDynamicRangeProvider<
        128, 256, 2, false, 148,
        6144, 3584, 4, 168,
        192, 64, 31, 144,
        4, 4, 16, true>;

static_assert(PairedTerminalDW2::kTaskPairedN);
static_assert(PairedTerminalDW13::kTaskPairedN);
static_assert(PairedTerminalDW2::Decoder::kNumClusterTasksPerGroup == 84u);
static_assert(PairedTerminalDW13::Decoder::kNumClusterTasksPerGroup == 168u);
static_assert(PairedTerminalDW2::kCompleteAcquireMask == 0x7ffu);

/** Compile the real current-parent K3 EP8 terminal-dynamic4 specialization.
 *
 * MegaMoE FP8/FP4 dgrad and symmetric-memory communication are unchanged.
 * Terminal dW2/dW13 use the existing 128x256x64 BF16 UMMA/TMA bodies and
 * one retained SMEM/TMEM lifecycle, with four-task post-readiness claims.
 */
void instantiate_k3_terminal_dynamic4_bf16_sm103a() {
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
        false>;
    (void)kernel;
}
