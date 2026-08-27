#pragma once

#include <cstring>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

// Map an activation name to its `deep_gemm::ActivationType` enumerator token
// (resolved inside the JIT-generated translation unit via `using namespace deep_gemm`).
static std::string get_activation_type_name(const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    if (activation == "situ")
        return "ActivationType::SiTU";
    DG_HOST_UNREACHABLE("Unsupported activation");
}

static std::string get_mega_moe_lora_mode_name(const std::string& mode) {
    if (mode == "disabled")
        return "MegaMoELoraMode::Disabled";
    if (mode == "payload_only")
        return "MegaMoELoraMode::PayloadOnly";
    if (mode == "fc1")
        return "MegaMoELoraMode::FC1";
    if (mode == "fc1_down")
        return "MegaMoELoraMode::FC1Down";
    DG_HOST_UNREACHABLE("Unsupported MegaMoE LoRA mode");
}

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        float situ_beta;
        float situ_linear_beta;
        bool fast_math;
        std::string activation;
        std::string lora_mode;
        int lora_rank;
        int num_lora_slots;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_lora_gate_a;
        CUtensorMap tensor_map_lora_up_a;
        CUtensorMap tensor_map_lora_gate_b;
        CUtensorMap tensor_map_lora_up_b;
        CUtensorMap tensor_map_lora_down_a;
        float lora_scaling;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {}, {}, {}
    >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_ring_tokens,
    args.config.num_sf_ring_tokens,
    args.config.num_stages,
    args.config.num_bytes_per_pull,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    get_activation_type_name(args.activation),
    to_string(args.situ_beta),
    to_string(args.situ_linear_beta),
    get_mega_moe_lora_mode_name(args.lora_mode),
    args.lora_rank,
    args.num_lora_slots);
    }

    template <MegaMoELoraMode kMode>
    static void launch_with_lora_maps(
        const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        layout::MegaMoELoraTensorMaps<kMode> lora_maps{};
        if constexpr (kMode == MegaMoELoraMode::FC1 or
                      kMode == MegaMoELoraMode::FC1Down) {
            static_assert(sizeof(layout::TmaDescriptorStorage) == sizeof(CUtensorMap));
            std::memcpy(&lora_maps.gate_a, &args.tensor_map_lora_gate_a, sizeof(CUtensorMap));
            std::memcpy(&lora_maps.up_a, &args.tensor_map_lora_up_a, sizeof(CUtensorMap));
            std::memcpy(&lora_maps.gate_b, &args.tensor_map_lora_gate_b, sizeof(CUtensorMap));
            std::memcpy(&lora_maps.up_b, &args.tensor_map_lora_up_b, sizeof(CUtensorMap));
        }
        if constexpr (kMode == MegaMoELoraMode::FC1Down) {
            std::memcpy(&lora_maps.down_a, &args.tensor_map_lora_down_a, sizeof(CUtensorMap));
            lora_maps.scaling = args.lora_scaling;
        }
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            lora_maps
        ));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // Keep descriptor-bearing FC1 launch data out of disabled and
        // payload-only kernel argument types.
        if (args.lora_mode == "disabled")
            launch_with_lora_maps<MegaMoELoraMode::Disabled>(kernel, config, args);
        else if (args.lora_mode == "payload_only")
            launch_with_lora_maps<MegaMoELoraMode::PayloadOnly>(kernel, config, args);
        else if (args.lora_mode == "fc1")
            launch_with_lora_maps<MegaMoELoraMode::FC1>(kernel, config, args);
        else
            launch_with_lora_maps<MegaMoELoraMode::FC1Down>(kernel, config, args);
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const std::string& activation,
    const float& activation_clamp,
    const float& situ_beta,
    const float& situ_linear_beta,
    const bool& fast_math,
    const std::string& lora_mode,
    const int& lora_rank,
    const int& num_lora_slots,
    const torch::Tensor& ring_lora_gate_up_acts,
    const std::optional<torch::Tensor>& lora_gate_b,
    const std::optional<torch::Tensor>& lora_up_b,
    const std::optional<torch::Tensor>& lora_down_a,
    const float& lora_scaling
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4,
        lora_mode == "fc1" or lora_mode == "fc1_down",
        lora_mode == "fc1_down");
    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    CUtensorMap tensor_map_lora_gate_a{}, tensor_map_lora_up_a{};
    CUtensorMap tensor_map_lora_gate_b{}, tensor_map_lora_up_b{};
    CUtensorMap tensor_map_lora_down_a{};
    if (lora_mode == "fc1" or lora_mode == "fc1_down") {
        constexpr int kLoraRank = 128;
        constexpr int kLoraKBlock = 64;
        constexpr int kLoraLoadRows = 8;
        const auto ring_lora_gate_a = ring_lora_gate_up_acts.select(1, 0);
        const auto ring_lora_up_a = ring_lora_gate_up_acts.select(1, 1);
        tensor_map_lora_gate_a = make_tma_2d_desc(
            ring_lora_gate_a, kLoraRank, num_ring_tokens,
            kLoraKBlock, kLoraLoadRows,
            static_cast<int>(ring_lora_gate_up_acts.stride(0)), 128);
        tensor_map_lora_up_a = make_tma_2d_desc(
            ring_lora_up_a, kLoraRank, num_ring_tokens,
            kLoraKBlock, kLoraLoadRows,
            static_cast<int>(ring_lora_gate_up_acts.stride(0)), 128);
        const auto lora_b_outer = num_lora_slots * num_experts_per_rank * intermediate_hidden;
        tensor_map_lora_gate_b = make_tma_2d_desc(
            lora_gate_b.value(), kLoraRank, lora_b_outer,
            kLoraKBlock, kLoraLoadRows,
            static_cast<int>(lora_gate_b->stride(-2)), 128);
        tensor_map_lora_up_b = make_tma_2d_desc(
            lora_up_b.value(), kLoraRank, lora_b_outer,
            kLoraKBlock, kLoraLoadRows,
            static_cast<int>(lora_up_b->stride(-2)), 128);
    }
    if (lora_mode == "fc1_down") {
        constexpr int kDownKBlock = 64;
        constexpr int kDownRank = 128;
        const auto down_outer =
            num_lora_slots * num_experts_per_rank * kDownRank;
        tensor_map_lora_down_a = make_tma_2d_desc(
            lora_down_a.value(), intermediate_hidden, down_outer,
            kDownKBlock, kDownRank,
            static_cast<int>(lora_down_a->stride(-2)), 128);
    }

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    const auto num_sms = device_runtime->get_num_sms();
    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .situ_beta = situ_beta,
        .situ_linear_beta = situ_linear_beta,
        .fast_math = fast_math,
        .activation = activation,
        .lora_mode = lora_mode,
        .lora_rank = lora_mode == "disabled" ? 0 : lora_rank,
        .num_lora_slots = lora_mode == "disabled" ? 0 : num_lora_slots,
        .config = config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .tensor_map_lora_gate_a = tensor_map_lora_gate_a,
        .tensor_map_lora_up_a = tensor_map_lora_up_a,
        .tensor_map_lora_gate_b = tensor_map_lora_gate_b,
        .tensor_map_lora_up_b = tensor_map_lora_up_b,
        .tensor_map_lora_down_a = tensor_map_lora_down_a,
        .lora_scaling = lora_scaling,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
