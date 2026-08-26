#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/impls/k3_mxfp8_exact_epilogue_ring.hpp>
#include <deep_gemm/impls/k3_multirange_backward.hpp>

namespace deep_gemm {

static std::string get_backward_route_weight_mode_name(
    const std::string& route_weight_mode) {
    if (route_weight_mode == "pre_down")
        return "RouteWeightMode::PreDown";
    if (route_weight_mode == "post_down")
        return "RouteWeightMode::PostDown";
    DG_HOST_UNREACHABLE("Unsupported route weight mode");
}

static std::string get_backward_activation_type_name(
    const std::string& activation) {
    if (activation == "swiglu")
        return "ActivationType::SwiGLU";
    if (activation == "geglu")
        return "ActivationType::GeGLU";
    if (activation == "situ")
        return "ActivationType::SiTU";
    DG_HOST_UNREACHABLE("Unsupported backward activation");
}

static std::string get_backward_combine_order_mode_name(
    const std::string& combine_order_mode) {
    if (combine_order_mode == "fixed_topk")
        return "CombineOrderMode::FixedTopK";
    if (combine_order_mode == "deepep")
        return "CombineOrderMode::DeepEP";
    if (combine_order_mode == "deepep_v1")
        return "CombineOrderMode::DeepEPV1";
    DG_HOST_UNREACHABLE("Unsupported combine order mode");
}

// Build the inline device ABI from a host-owned packed-range description.
// Every range contributes five values in physical arena order:
//   [active tokens, symmetric token capacity, pool rows, activation rows,
//    scale-factor rows].
// Token offsets advance by the rank-uniform capacity while pool/TMA offsets
// advance by their physical row counts. This distinction is required for
// variable-length ranks: a remote metadata token must map to the same
// symmetric row even when another rank has fewer active tokens.
static K3BackwardRangeSet make_k3_backward_range_set(
    const std::vector<int64_t>& packed_range_sizes,
    const uint32_t default_num_tokens,
    const uint32_t default_max_tokens_per_rank,
    const uint32_t default_num_pool_rows,
    const uint32_t default_num_acts_rows,
    const uint32_t default_num_sf_pool_rows,
    const uint32_t num_experts) {
    constexpr uint32_t kFieldsPerRange = 5u;
    if (packed_range_sizes.empty()) {
        return K3BackwardRangeSet::single_range(
            default_num_tokens, default_max_tokens_per_rank,
            default_num_pool_rows, default_num_acts_rows,
            default_num_sf_pool_rows);
    }

    DG_HOST_ASSERT(
        packed_range_sizes.size() % kFieldsPerRange == 0u);
    const uint32_t num_ranges = static_cast<uint32_t>(
        packed_range_sizes.size() / kFieldsPerRange);
    DG_HOST_ASSERT(
        num_ranges > 0u && num_ranges <= kK3MaxBackwardRanges);

    const auto read_u32 = [&](const uint32_t index) {
        const int64_t value = packed_range_sizes[index];
        DG_HOST_ASSERT(
            value >= 0 &&
            static_cast<uint64_t>(value) <=
                std::numeric_limits<uint32_t>::max());
        return static_cast<uint32_t>(value);
    };

    std::array<K3BackwardRangeShape, kK3MaxBackwardRanges> shapes{};
    uint64_t token_count = 0u;
    uint64_t token_capacity = 0u;
    uint64_t pool_rows = 0u;
    uint64_t acts_rows = 0u;
    uint64_t sf_pool_rows = 0u;
    for (uint32_t range_idx = 0u;
         range_idx < num_ranges; ++range_idx) {
        const uint32_t field_begin = range_idx * kFieldsPerRange;
        const uint32_t num_tokens = read_u32(field_begin + 0u);
        const uint32_t max_tokens_per_rank =
            read_u32(field_begin + 1u);
        const uint32_t num_pool_rows = read_u32(field_begin + 2u);
        const uint32_t num_acts_rows = read_u32(field_begin + 3u);
        const uint32_t num_sf_pool_rows = read_u32(field_begin + 4u);
        DG_HOST_ASSERT(num_tokens <= max_tokens_per_rank);
        shapes[range_idx] = {
            num_tokens, max_tokens_per_rank,
            num_pool_rows, num_acts_rows, num_sf_pool_rows};
        token_count += num_tokens;
        token_capacity += max_tokens_per_rank;
        pool_rows += num_pool_rows;
        acts_rows += num_acts_rows;
        sf_pool_rows += num_sf_pool_rows;
    }

    DG_HOST_ASSERT(token_count == default_num_tokens);
    DG_HOST_ASSERT(token_capacity == default_max_tokens_per_rank);
    DG_HOST_ASSERT(pool_rows == default_num_pool_rows);
    DG_HOST_ASSERT(acts_rows == default_num_acts_rows);
    DG_HOST_ASSERT(sf_pool_rows == default_num_sf_pool_rows);
    const auto result = pack_k3_backward_ranges(
        shapes.data(), num_ranges, num_experts);
    DG_HOST_ASSERT(result.is_packed(num_experts));
    return result;
}

class SM100MegaMoEBackwardCombineRuntime final
    : public LaunchRuntime<
          SM100MegaMoEBackwardCombineRuntime> {
public:
    struct Args {
        int num_ranks;
        int num_local_experts;
        std::string combine_order_mode = "fixed_topk";
        cutlass::bfloat16_t* grad_x_output;
        const cutlass::bfloat16_t* combine_buffer;
        const int64_t* topk_ids;
        uint32_t num_tokens;
        uint32_t num_max_tokens;
        uint32_t num_topk;
        uint32_t hidden;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mega_moe_backward_combine.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_mega_moe_backward_combine_grad_x<
            {}, {}, {}>);
}};
)",
            args.num_ranks, args.num_local_experts,
            get_backward_combine_order_mode_name(
                args.combine_order_mode));
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config, args.grad_x_output,
            args.combine_buffer, args.topk_ids,
            args.num_tokens, args.num_max_tokens,
            args.num_topk, args.hidden));
    }
};

class SM100BF16MegaMoEBackwardPostDownPreludeRuntime final
    : public LaunchRuntime<
          SM100BF16MegaMoEBackwardPostDownPreludeRuntime> {
public:
    struct Args {
        int hidden;
        int num_experts;
        int block_m;
        int num_sms;
        int num_ranks;
        std::string combine_order_mode = "fixed_topk";
        bool do_reverse_dispatch = true;
        bool compute_route_dot = true;
        bool write_weighted = true;
        bool synchronize_ranks = true;
        bool synchronize_after_dispatch = true;
        bool barrier_only = false;
        bool x_prepared = false;
        int route_prelude_threads = 256;
        const int* expert_counts;
        layout::Workspace backward_workspace;
        layout::SymBuffer<> backward_sym_buffer;
        const cutlass::bfloat16_t* backward_grad_y;
        const cutlass::bfloat16_t* backward_x;
        const float* backward_topk_weights;
        float* backward_grad_route;
        const layout::TokenSrcMetadata* token_src_metadata;
        uint32_t num_topk;
        uint32_t num_pool_rows;
        cutlass::bfloat16_t* grad_y_unweighted_output;
        cutlass::bfloat16_t* grad_y_weighted_output;
        cutlass::bfloat16_t* x_pool_output;
        float* route_weights_output;
        const cutlass::bfloat16_t* down_unweighted;
        float* grad_route_output;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

// K3 in-kernel on-demand residual-weight cache revision 199.

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_bf16_mega_moe_backward_post_down_prelude<
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        >);
}};
)",
            args.hidden, args.num_experts, args.block_m,
            args.num_sms, args.num_ranks,
            get_backward_combine_order_mode_name(
                args.combine_order_mode),
            args.do_reverse_dispatch ? "true" : "false",
            args.compute_route_dot ? "true" : "false",
            args.write_weighted ? "true" : "false",
            "false",
            args.synchronize_ranks ? "true" : "false",
            args.synchronize_after_dispatch ? "true" : "false",
            args.barrier_only ? "true" : "false",
            args.x_prepared ? "true" : "false",
            args.route_prelude_threads);
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.expert_counts,
            args.backward_workspace,
            args.backward_sym_buffer,
            args.backward_grad_y,
            args.backward_x,
            args.backward_topk_weights,
            args.backward_grad_route,
            args.token_src_metadata,
            args.num_topk,
            args.num_pool_rows,
            args.grad_y_unweighted_output,
            args.grad_y_weighted_output,
            args.x_pool_output,
            args.route_weights_output,
            args.down_unweighted,
            args.grad_route_output));
    }
};

class SM100FP8FP4MegaMoEBackwardWaveRuntime final
    : public LaunchRuntime<SM100FP8FP4MegaMoEBackwardWaveRuntime> {
public:
    struct Args {
        int hidden;
        int intermediate_hidden;
        int num_experts;
        int num_pool_rows;
        int num_acts_rows;
        int num_sf_pool_rows;
        int block_m;
        int block_n;
        int block_k;
        int sf_block_m;
        int sf_block_n;
        int num_stages;
        int num_sms;
        int num_ranks;
        bool bf16_mode = false;
        bool inline_weight_dequant = false;
        bool phase_ordered_weight_dequant = false;
        bool branch_major_bf16_wgrad_tail = false;
        bool mxfp8_three_term_wgrad = false;
        bool inline_residual_mxfp8_dgrad = false;
        bool residual_mxfp8_dgrad = false;
        bool build_residual_mxfp8_weights = false;
        bool exact_source_x = false;
        bool gate_up_prepared = false;
        std::string activation = "swiglu";
        float situ_beta = 1.0f;
        float situ_linear_beta =
            std::numeric_limits<float>::infinity();
        bool fast_math = false;
        std::string route_weight_mode = "pre_down";
        std::string combine_order_mode = "fixed_topk";

        const int* expert_counts;
        layout::SymBuffer<> backward_sym_buffer;
        layout::Workspace backward_workspace;
        const cutlass::bfloat16_t* backward_grad_y;
        const cutlass::bfloat16_t* backward_x;
        const float* backward_topk_weights;
        float* backward_grad_route;
        cutlass::bfloat16_t* backward_grad_x_output = nullptr;
        uint32_t num_backward_tokens = 0;
        K3BackwardRangeSet backward_ranges{};
        const layout::TokenSrcMetadata* token_src_metadata;
        uint32_t num_topk;
        uint32_t acts_sf_stride;
        CUtensorMap tensor_map_acts;
        CUtensorMap tensor_map_acts_sf;
        CUtensorMap tensor_map_weights;
        CUtensorMap tensor_map_weights_sf;
        CUtensorMap tensor_map_output;
        CUtensorMap tensor_map_grad_ye;
        CUtensorMap tensor_map_w2_dequant;
        CUtensorMap tensor_map_w2_dgrad_weights;
        CUtensorMap tensor_map_w2_dgrad_weights_sf;
        CUtensorMap tensor_map_w2_weights;
        CUtensorMap tensor_map_w2_scales;
        CUtensorMap tensor_map_w13_dequant;
        CUtensorMap tensor_map_w13_dgrad_weights;
        CUtensorMap tensor_map_w13_dgrad_weights_sf;
        CUtensorMap tensor_map_w13_weights;
        CUtensorMap tensor_map_w13_scales;
        CUtensorMap tensor_map_grad_gate_up;
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w2_wgrad_slot_a{};
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w2_wgrad_slot_b{};
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w2_wgrad_slot_d{};
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w13_wgrad_slot_a{};
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w13_wgrad_slot_b{};
        K3MxFp8WgradAuxSlot<
            CUtensorMap, cutlass::bfloat16_t,
            cutlass::float_e4m3_t> tensor_map_w13_wgrad_slot_d{};
        K3MxFp8WgradTensorMapPack<CUtensorMap>
            k3_mxfp8_wgrad_tensor_maps{};
        const cutlass::float_e4m3_t* acts_ptr;
        const uint32_t* acts_sf_ptr;
        const int8_t* w2_weights;
        const float* w2_scales;
        cutlass::bfloat16_t* w2_dequant_scratch;
        const int8_t* w13_weights;
        const float* w13_scales;
        cutlass::bfloat16_t* w13_dequant_scratch;
        const cutlass::bfloat16_t* gate_up_output;
        cutlass::bfloat16_t* grad_ye_output;
        cutlass::bfloat16_t* grad_y_unweighted_output;
        cutlass::bfloat16_t* route_weights;
        float* route_weights_fp32;
        cutlass::bfloat16_t* grad_h_output;
        cutlass::bfloat16_t* grad_gate_up_output;
        cutlass::bfloat16_t* h_act_output;
        cutlass::bfloat16_t* h_weighted_output;
        cutlass::bfloat16_t* x_pool_output;
        cutlass::bfloat16_t* grad_x_pool_output;
        const cutlass::bfloat16_t* down_unweighted_output;
        float* grad_route_output;
        uint32_t* grid_sync_counter;
        float activation_limit;
        bool compute_w13_dgrad;
        bool direct_remote_grad_x;
        bool write_grad_x_pool;
        bool clear_wgrad_padding;
        bool clear_empty_wgrad_expert_outputs;
        bool inline_wgrad = false;
        bool accumulate_wgrad = false;
        bool compute_route_grad = false;
        bool trace_kernel = false;
        bool vectorized_grad_x_store = false;
        bool wide_grad_x_store = false;
        bool multi_range_backward = false;
        uint64_t* kernel_trace = nullptr;
        bool inputs_prepared = false;
        bool dispatch_inputs_prepared = false;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // The ready-driven schedule exists to join two physical activation
        // ranges without a host-visible intermediate. Keep ordinary one-range
        // launches on the lower-overhead legacy suffix; enabling readiness for
        // them only removes W13 dgrad producers and has no range boundary to
        // hide.
        const bool enable_k3_ready_wgrad =
            args.multi_range_backward &&
            args.hidden == 3584 &&
            args.intermediate_hidden == 3072 &&
            args.num_experts == 112 &&
            args.block_m == 192 &&
            args.num_sms == 148 &&
            args.num_ranks == 8 &&
            !args.bf16_mode &&
            !args.inline_weight_dequant &&
            !args.phase_ordered_weight_dequant &&
            !args.inline_residual_mxfp8_dgrad &&
            !args.residual_mxfp8_dgrad &&
            !args.build_residual_mxfp8_weights &&
            args.exact_source_x &&
            args.gate_up_prepared &&
            args.activation == "situ" &&
            args.route_weight_mode == "post_down" &&
            args.combine_order_mode == "fixed_topk" &&
            args.compute_w13_dgrad &&
            args.direct_remote_grad_x &&
            args.inline_wgrad;
        // Exactly two physical ranges use the same compact logical K axis as
        // the one-range exact engine. Keep this selector narrow so three-range
        // launches retain their separately validated hybrid handoff.
        const bool enable_k3_mxfp8_two_range_exact =
            args.mxfp8_three_term_wgrad &&
            args.multi_range_backward &&
            enable_k3_ready_wgrad &&
            args.backward_ranges.num_ranges == 2u &&
            args.num_topk == 16u &&
            args.clear_wgrad_padding &&
            !args.accumulate_wgrad &&
            args.situ_beta == 4.0f &&
            args.situ_linear_beta == 25.0f;
        const std::string ready_wgrad_defines =
            enable_k3_ready_wgrad &&
                !enable_k3_mxfp8_two_range_exact
            ? "#define DG_EXPERIMENTAL_K3_READY_WGRAD 1\n"
              "#define DG_EXPERIMENTAL_K3_READY_WGRAD_INITIAL_CLUSTERS 8\n"
              "#define DG_EXPERIMENTAL_K3_READY_WGRAD_BATCH_TASKS 32\n"
            : "";
        const std::string mxfp8_three_term_wgrad_define =
            args.mxfp8_three_term_wgrad &&
                (!args.multi_range_backward ||
                 enable_k3_mxfp8_two_range_exact)
            ? "#define DG_EXPERIMENTAL_K3_MXFP8_THREE_TERM_WGRAD 1\n"
              "#define DG_EXPERIMENTAL_K3_MXFP8_WGRAD_OVERLAP 1\n"
            : "";
        const std::string branch_major_bf16_wgrad_tail_define =
            args.branch_major_bf16_wgrad_tail
            ? "#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_WGRAD_TAIL 1\n"
            : "";
        const std::string branch_major_bf16_dynamic_tail_define =
            args.branch_major_bf16_wgrad_tail
            ? "#define DG_EXPERIMENTAL_K3_BRANCH_MAJOR_BF16_DYNAMIC_TAIL 1\n"
            : "";
        const std::string two_segment_bf16_progressive_define =
            args.mxfp8_three_term_wgrad &&
                args.multi_range_backward &&
                !enable_k3_mxfp8_two_range_exact &&
                enable_k3_ready_wgrad &&
                args.backward_ranges.num_ranges == 2u &&
                args.num_topk == 16u &&
                args.clear_wgrad_padding &&
                !args.accumulate_wgrad &&
                args.situ_beta == 4.0f &&
                args.situ_linear_beta == 25.0f
            ? "#define DG_EXPERIMENTAL_K3_TWO_SEGMENT_BF16_PROGRESSIVE_WGRAD 1\n"
            : "";
        const bool enable_k3_mxfp8_exact_epilogue_ring =
            args.mxfp8_three_term_wgrad &&
                args.multi_range_backward &&
                enable_k3_ready_wgrad &&
                args.backward_ranges.num_ranges ==
                    kK3MaxBackwardRanges &&
                args.num_topk == 16u &&
                args.clear_wgrad_padding &&
                !args.accumulate_wgrad &&
                args.situ_beta == 4.0f &&
                args.situ_linear_beta == 25.0f;
        const std::string mxfp8_dw13_hybrid_define =
            enable_k3_mxfp8_exact_epilogue_ring
            ? "#define DG_EXPERIMENTAL_K3_MXFP8_DW13_HYBRID 1\n"
            : "";
        const std::string exact_epilogue_ring_define =
            enable_k3_mxfp8_exact_epilogue_ring
            ? "#define DG_EXPERIMENTAL_K3_MXFP8_EXACT_EPILOGUE_RING 1\n"
            : "";
        const std::string exact_epilogue_ring_watchdog_define =
            !exact_epilogue_ring_define.empty() && args.trace_kernel
            ? "#define DG_EXPERIMENTAL_K3_MXFP8_EXACT_RING_WATCHDOG 1\n"
            : "";
        return fmt::format(R"(
{}{}{}{}{}{}{}{}
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_backward.cuh>

using namespace deep_gemm;

// K3 in-kernel on-demand residual-weight cache revision 250.

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &sm100_fp8_fp4_mega_moe_backward_wave_impl<
            {}, {},                 // hidden, intermediate hidden
            {},                     // experts
            {}, {}, {},             // M, N, K
            {}, {},                 // scale M, N
            {},                     // stages
            {},                     // SMs
            {},                     // ranks
            {},                     // compile W13 dgrad
            {},                     // BF16 mode
            {},                     // inline weight dequant
            {},                     // phase-ordered weight dequant
            {},                     // inline residual MXFP8 dgrad
            {},                     // residual MXFP8 dgrad
            {},                     // build residual MXFP8 weights
            {},                     // exact source x
            {},                     // prepared gate/up
            {},                     // activation
            {}, {},                 // SiTU beta, linear beta
            {},                     // fast math
            {},                     // route-weight mode
            {},                     // combine-order mode
            {},                     // inputs prepared
            {},                     // dispatch inputs prepared
            {},                     // direct remote grad-x
            {},                     // write grad-x pool
            {},                     // clear wgrad padding
            {},                     // inline wgrad
            {},                     // accumulate wgrad
            {},                     // compute route grad
            {},                     // trace
            {},                     // vectorized grad-x store
            {},                     // wide grad-x store
            {}                      // multi-range backward
        >);
}};
)",
            ready_wgrad_defines,
            mxfp8_three_term_wgrad_define,
            branch_major_bf16_wgrad_tail_define,
            branch_major_bf16_dynamic_tail_define,
            two_segment_bf16_progressive_define,
            mxfp8_dw13_hybrid_define,
            exact_epilogue_ring_define,
            exact_epilogue_ring_watchdog_define,
            args.hidden, args.intermediate_hidden,
            args.num_experts,
            args.block_m, args.block_n, args.block_k,
            args.sf_block_m, args.sf_block_n,
            args.num_stages,
            args.num_sms,
            args.num_ranks,
            args.compute_w13_dgrad ? "true" : "false",
            args.bf16_mode ? "true" : "false",
            args.inline_weight_dequant ? "true" : "false",
            args.phase_ordered_weight_dequant ? "true" : "false",
            args.inline_residual_mxfp8_dgrad ? "true" : "false",
            args.residual_mxfp8_dgrad ? "true" : "false",
            args.build_residual_mxfp8_weights ? "true" : "false",
            args.exact_source_x ? "true" : "false",
            args.gate_up_prepared ? "true" : "false",
            get_backward_activation_type_name(args.activation),
            to_string(args.situ_beta),
            to_string(args.situ_linear_beta),
            args.fast_math ? "true" : "false",
            get_backward_route_weight_mode_name(
                args.route_weight_mode),
            get_backward_combine_order_mode_name(
                args.combine_order_mode),
            args.inputs_prepared ? "true" : "false",
            args.dispatch_inputs_prepared ? "true" : "false",
            args.direct_remote_grad_x ? "true" : "false",
            args.write_grad_x_pool ? "true" : "false",
            args.clear_wgrad_padding ? "true" : "false",
            args.inline_wgrad ? "true" : "false",
            args.accumulate_wgrad ? "true" : "false",
            args.compute_route_grad ? "true" : "false",
            args.trace_kernel ? "true" : "false",
            args.vectorized_grad_x_store ? "true" : "false",
            args.wide_grad_x_store ? "true" : "false",
            args.multi_range_backward ? "true" : "false");
    }

    static void launch_impl(
        const KernelHandle& kernel,
        const LaunchConfigHandle& config,
        Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(
            kernel, config,
            args.expert_counts,
            args.backward_sym_buffer,
            args.backward_workspace,
            args.backward_grad_y,
            args.backward_x,
            args.backward_topk_weights,
            args.backward_grad_route,
            args.backward_grad_x_output,
            args.num_backward_tokens,
            args.token_src_metadata,
            args.num_topk,
            args.num_pool_rows,
            args.num_acts_rows,
            args.backward_ranges,
            args.acts_sf_stride,
            args.tensor_map_acts,
            args.tensor_map_acts_sf,
            args.tensor_map_weights,
            args.tensor_map_weights_sf,
            args.tensor_map_output,
            args.tensor_map_grad_ye,
            args.tensor_map_w2_dequant,
            args.tensor_map_w2_dgrad_weights,
            args.tensor_map_w2_dgrad_weights_sf,
            args.tensor_map_w2_weights,
            args.tensor_map_w2_scales,
            args.tensor_map_w13_dequant,
            args.tensor_map_w13_dgrad_weights,
            args.tensor_map_w13_dgrad_weights_sf,
            args.tensor_map_w13_weights,
            args.tensor_map_w13_scales,
            args.tensor_map_grad_gate_up,
            args.tensor_map_w2_wgrad_slot_a,
            args.tensor_map_w2_wgrad_slot_b,
            args.tensor_map_w2_wgrad_slot_d,
            args.tensor_map_w13_wgrad_slot_a,
            args.tensor_map_w13_wgrad_slot_b,
            args.tensor_map_w13_wgrad_slot_d,
            args.k3_mxfp8_wgrad_tensor_maps,
            args.acts_ptr,
            args.acts_sf_ptr,
            args.w2_weights,
            args.w2_scales,
            args.w2_dequant_scratch,
            args.w13_weights,
            args.w13_scales,
            args.w13_dequant_scratch,
            args.gate_up_output,
            args.grad_ye_output,
            args.grad_y_unweighted_output,
            args.route_weights,
            args.route_weights_fp32,
            args.grad_h_output,
            args.grad_gate_up_output,
            args.h_act_output,
            args.h_weighted_output,
            args.x_pool_output,
            args.grad_x_pool_output,
            args.down_unweighted_output,
            args.grad_route_output,
            args.grid_sync_counter,
            args.clear_empty_wgrad_expert_outputs,
            args.activation_limit,
            args.kernel_trace));
    }
};

static void sm100_fp8_fp4_mega_moe_backward_dgrad_swiglu(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& acts,
    const torch::Tensor& acts_sf,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l1_weights_sf,
    const torch::Tensor& grad_ye,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w2_scales,
    const torch::Tensor& w2_dequant_scratch,
    const torch::Tensor& w13_weights,
    const torch::Tensor& w13_scales,
    const torch::Tensor& w13_dequant_scratch,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const bool& compute_w13_dgrad,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const bool& clear_empty_wgrad_expert_outputs,
    const int& block_m,
    const std::vector<int64_t>& backward_sym_buffer_ptrs = {},
    const int& backward_rank = 0,
    const int& num_max_tokens_per_rank = 0,
    const int& num_topk = 0,
    const std::optional<torch::Tensor>& backward_grad_y = std::nullopt,
    const std::optional<torch::Tensor>& backward_topk_weights = std::nullopt,
    const std::optional<torch::Tensor>& backward_grad_route = std::nullopt,
    const std::optional<torch::Tensor>& token_src_metadata = std::nullopt,
    const std::string& route_weight_mode = "pre_down",
    const std::optional<torch::Tensor>& grad_y_unweighted_output =
        std::nullopt,
    const std::optional<torch::Tensor>& down_unweighted_output =
        std::nullopt,
    const std::optional<torch::Tensor>& grad_route_output =
        std::nullopt,
    const std::string& activation = "swiglu",
    const std::optional<float>& situ_beta_opt = std::nullopt,
    const std::optional<float>& situ_linear_beta_opt =
        std::nullopt,
    const bool& inline_weight_dequant = false,
    const bool& inline_residual_mxfp8_dgrad = false,
    const bool& build_residual_mxfp8_weights = false,
    const std::optional<torch::Tensor>& w2_dgrad_weights =
        std::nullopt,
    const std::optional<torch::Tensor>& w2_dgrad_weights_sf =
        std::nullopt,
    const std::optional<torch::Tensor>& w13_dgrad_weights =
        std::nullopt,
    const std::optional<torch::Tensor>& w13_dgrad_weights_sf =
        std::nullopt,
    const std::optional<torch::Tensor>& backward_x =
        std::nullopt,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt,
    const bool& inline_wgrad = false,
    const bool& accumulate_wgrad = false,
    const std::optional<torch::Tensor>& combined_grad_x_output =
        std::nullopt,
    const bool& gate_up_prepared = false,
    const std::optional<torch::Tensor>& w2_wgrad_output =
        std::nullopt,
    const std::optional<torch::Tensor>& w13_wgrad_output =
        std::nullopt,
    const std::vector<int64_t>& backward_range_sizes = {},
    // Explicit mode-owned opt-in. The host predicate below rejects every
    // shape, topology, activation, or output contract outside exact K3 EP8.
    const bool& mxfp8_three_term_wgrad = false) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    const int residual_mxfp8_operand_count =
        static_cast<int>(w2_dgrad_weights.has_value()) +
        static_cast<int>(w2_dgrad_weights_sf.has_value()) +
        static_cast<int>(w13_dgrad_weights.has_value()) +
        static_cast<int>(w13_dgrad_weights_sf.has_value());
    DG_HOST_ASSERT(
        residual_mxfp8_operand_count == 0 ||
        residual_mxfp8_operand_count == 4);
    const bool external_residual_mxfp8_weights =
        w2_dgrad_weights.has_value() &&
        w2_dgrad_weights_sf.has_value() &&
        w13_dgrad_weights.has_value() &&
        w13_dgrad_weights_sf.has_value();
    const bool residual_mxfp8_dgrad =
        inline_residual_mxfp8_dgrad ||
        external_residual_mxfp8_weights;
    const bool early_w2_wgrad =
        compute_w13_dgrad && inline_wgrad &&
        build_residual_mxfp8_weights &&
        route_weight_mode == "post_down";
    // This selector also binds the once-quantized activation descriptors.
    // External compact dgrad weights do not need the on-demand conversion
    // path, but they still require the prepared gate/up source and aliases.
    DG_HOST_ASSERT(
        !build_residual_mxfp8_weights ||
        (external_residual_mxfp8_weights && gate_up_prepared));
    DG_HOST_ASSERT(
        !residual_mxfp8_dgrad || !inline_weight_dequant);
    const int dgrad_block_k = residual_mxfp8_dgrad ? 128 : 64;
    constexpr int store_block_m = 16;
    constexpr int gran_k = 32;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;
    constexpr int wgrad_block_m = 128;
    constexpr int wgrad_block_n = 256;
    // Inline Wgrad runs as a 2-CTA cluster with multicast on A. Each CTA
    // therefore owns half of the B/N tile, and its TMA descriptor must encode
    // that 128-column load box rather than the cluster-wide 256-column tile.
    constexpr int wgrad_load_block_n = wgrad_block_n / 2;
    constexpr int wgrad_store_block_n = 64;
    constexpr int wgrad_num_stages = 4;
    const int wgrad_block_k =
        block_m % 64 == 0 ? 64 : block_m % 32 == 0 ? 32 : 16;
    const int wgrad_swizzle =
        wgrad_block_k * sizeof(cutlass::bfloat16_t);
    const int wgrad_smem_size =
        2 * wgrad_block_m * wgrad_store_block_n *
            sizeof(cutlass::bfloat16_t) +
        wgrad_num_stages * (wgrad_block_m + wgrad_block_n) *
            wgrad_block_k * sizeof(cutlass::bfloat16_t) +
        1024;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        check_grouped_ab_fp8_fp4(
            l1_weights, cute::UMMA::Major::K,
            device_runtime->get_arch_major());
    const int intermediate_hidden = intermediate_hidden_2 / 2;
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int physical_num_acts_rows = static_cast<int>(acts.size(0));
    const int physical_num_sf_pool_rows =
        static_cast<int>(acts_sf.size(0));
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_ranks = backward_sym_buffer_ptrs.empty()
        ? 1
        : static_cast<int>(backward_sym_buffer_ptrs.size());
    const int num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(
        backward_range_sizes.empty() ||
        backward_range_sizes.size() % 5u == 0u);
    const int num_backward_ranges = backward_range_sizes.empty()
        ? 1
        : static_cast<int>(backward_range_sizes.size() / 5u);
    const bool multi_range_backward = num_backward_ranges > 1;
    // Hybrid dW13 uses the FP8 activation ring only as a compact byte arena
    // for exact scales; its logical BF16 sources live in the full captured
    // pools. Preserve the range-total descriptor extents while allowing the
    // existing physical ring to be shorter when its byte capacity is enough.
    const int num_acts_rows =
        mxfp8_three_term_wgrad && multi_range_backward
        ? num_pool_rows : physical_num_acts_rows;
    int num_sf_pool_rows = physical_num_sf_pool_rows;
    if (mxfp8_three_term_wgrad && multi_range_backward) {
        uint64_t logical_sf_rows = 0u;
        for (size_t field = 4u;
             field < backward_range_sizes.size(); field += 5u) {
            logical_sf_rows += backward_range_sizes[field];
        }
        DG_HOST_ASSERT(
            logical_sf_rows > 0u &&
            logical_sf_rows <=
                static_cast<uint64_t>(std::numeric_limits<int>::max()));
        num_sf_pool_rows = static_cast<int>(logical_sf_rows);
    }
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;
    const float situ_beta = situ_beta_opt.value_or(1.0f);
    const float situ_linear_beta = situ_linear_beta_opt.value_or(
        std::numeric_limits<float>::infinity());
    const bool aliases_w2_dequant =
        down_unweighted_output.has_value() &&
        w2_dequant_scratch.data_ptr() ==
            down_unweighted_output->data_ptr();
    const bool aliases_w13_dequant =
        w13_dequant_scratch.data_ptr() ==
        gate_up_output.data_ptr();
    DG_HOST_ASSERT(aliases_w2_dequant == aliases_w13_dequant);
    const bool phase_ordered_weight_dequant =
        !inline_weight_dequant && aliases_w2_dequant;
    // This host-only mode bit is part of the generated-source cache key and
    // is intentionally independent of the dgrad weight-dequant schedule.
    // It therefore selects the same suffix for first-chunk, small accumulated
    // (inline dequant), and large phase-ordered accumulated launches.
    const bool k3_mxfp8_three_term_wgrad_eligible =
        hidden == 3584 && intermediate_hidden == 3072 &&
        num_experts == 112 && block_m == 192 &&
        num_sms == 148 && num_ranks == 8 && num_topk == 16 &&
        compute_w13_dgrad && direct_remote_grad_x &&
        clear_wgrad_padding && inline_wgrad &&
        !inline_residual_mxfp8_dgrad &&
        !external_residual_mxfp8_weights &&
        !build_residual_mxfp8_weights &&
        backward_x.has_value() && gate_up_prepared &&
        activation == "situ" && situ_beta == 4.0f &&
        situ_linear_beta == 25.0f &&
        route_weight_mode == "post_down" &&
        grad_route_output.has_value() &&
        combined_grad_x_output.has_value() &&
        grad_y_unweighted_output.has_value() &&
        down_unweighted_output.has_value() &&
        w2_wgrad_output.has_value() &&
        w13_wgrad_output.has_value();
    DG_HOST_ASSERT(
        !mxfp8_three_term_wgrad ||
        k3_mxfp8_three_term_wgrad_eligible);
    // Preserve the public mxfp8_mxfp4_megamoe mode while selecting the terminal
    // BF16 wgrad suffix internally for either one physical range or exactly two
    // immutable physical ranges. Three-range, accumulated, and aliased dequant
    // launches retain their current exact backends.
    const bool enable_k3_branch_major_bf16_wgrad_tail =
        mxfp8_three_term_wgrad &&
        k3_mxfp8_three_term_wgrad_eligible &&
        num_ranks == 8 &&
        (!multi_range_backward || num_backward_ranges == 2) &&
        !accumulate_wgrad &&
        !inline_weight_dequant && !phase_ordered_weight_dequant &&
        !residual_mxfp8_dgrad;
    const bool enable_k3_mxfp8_three_term_wgrad =
        mxfp8_three_term_wgrad &&
        k3_mxfp8_three_term_wgrad_eligible &&
        !enable_k3_branch_major_bf16_wgrad_tail;
    DG_HOST_ASSERT(
        !enable_k3_branch_major_bf16_wgrad_tail ||
        !enable_k3_mxfp8_three_term_wgrad);
    // The training mode owns the full exact-ring selection. There is no
    // process-global override: an eligible three-range invocation either
    // emits and caches the exact specialization or leaves it unreachable.
    const bool enable_k3_mxfp8_dw13_hybrid =
        enable_k3_mxfp8_three_term_wgrad && multi_range_backward &&
        !inline_weight_dequant && !phase_ordered_weight_dequant &&
        !residual_mxfp8_dgrad &&
        num_backward_ranges == static_cast<int>(kK3MaxBackwardRanges) &&
        !accumulate_wgrad;
    const bool enable_k3_mxfp8_exact_epilogue_ring =
        enable_k3_mxfp8_dw13_hybrid;

    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        num_backward_ranges >= 1 &&
        num_backward_ranges <=
            static_cast<int>(kK3MaxBackwardRanges));
    DG_HOST_ASSERT(!accumulate_wgrad || inline_wgrad);
    DG_HOST_ASSERT(
        w2_wgrad_output.has_value() ==
        w13_wgrad_output.has_value());
    DG_HOST_ASSERT(
        !accumulate_wgrad || w2_wgrad_output.has_value());
    DG_HOST_ASSERT(
        !inline_wgrad ||
        (hidden % wgrad_block_n == 0 &&
         intermediate_hidden % wgrad_block_n == 0 &&
         block_m % 16 == 0));
    if (multi_range_backward) {
        DG_HOST_ASSERT(num_sms == 148);
        DG_HOST_ASSERT(
            num_ranks == 8 && num_experts == 112 &&
            hidden == 3584 && intermediate_hidden == 3072 &&
            block_m == 192 && compute_w13_dgrad &&
            direct_remote_grad_x &&
            clear_wgrad_padding && inline_wgrad &&
            !accumulate_wgrad && backward_x.has_value() &&
            gate_up_prepared && activation == "situ" &&
            situ_beta == 4.0f && situ_linear_beta == 25.0f &&
            route_weight_mode == "post_down" &&
            !inline_weight_dequant &&
            !inline_residual_mxfp8_dgrad &&
            !build_residual_mxfp8_weights &&
            num_pool_rows == num_acts_rows &&
            num_pool_rows % block_m == 0);
        DG_HOST_ASSERT(
            gate_up_output.data_ptr() ==
                grad_gate_up_output.data_ptr());
        DG_HOST_ASSERT(down_unweighted_output.has_value());
        DG_HOST_ASSERT(
            h_act_output.data_ptr() ==
                grad_h_output.data_ptr());
        DG_HOST_ASSERT(
            h_weighted_output.data_ptr() ==
                h_act_output.data_ptr());
        DG_HOST_ASSERT(
            x_pool_output.data_ptr() ==
                down_unweighted_output->data_ptr());
        DG_HOST_ASSERT(
            down_unweighted_output->sizes() ==
                x_pool_output.sizes());
    }
    DG_HOST_ASSERT(
        activation == "swiglu" || activation == "geglu" ||
        activation == "situ");
    DG_HOST_ASSERT(activation != "situ" || situ_beta > 0.0f);
    DG_HOST_ASSERT(
        activation != "situ" || situ_linear_beta > 0.0f);
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(num_ranks == 1 ||
                   (backward_rank >= 0 && backward_rank < num_ranks));
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % block_k == 0);
    DG_HOST_ASSERT(hidden % dgrad_block_k == 0);
    DG_HOST_ASSERT(intermediate_hidden_2 % block_n == 0);
    DG_HOST_ASSERT(acts.dim() == 2 and acts.size(1) == hidden);
    DG_HOST_ASSERT(num_acts_rows > 0 && num_acts_rows <= num_pool_rows);
    DG_HOST_ASSERT(num_sf_pool_rows > 0);
    DG_HOST_ASSERT(grad_ye.dim() == 2);
    DG_HOST_ASSERT(grad_ye.size(0) == num_pool_rows);
    DG_HOST_ASSERT(grad_ye.size(1) == hidden);
    DG_HOST_ASSERT(grad_ye.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_ye.is_contiguous());
    DG_HOST_ASSERT(gate_up_output.dim() == 2);
    DG_HOST_ASSERT(
        gate_up_output.size(1) == intermediate_hidden_2);
    DG_HOST_ASSERT(
        gate_up_prepared
            ? (gate_up_output.size(0) >= num_acts_rows &&
               gate_up_output.size(0) <= num_pool_rows)
            : gate_up_output.size(0) == num_pool_rows);
    DG_HOST_ASSERT(
        grad_gate_up_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_h_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(grad_x_pool_output.sizes() ==
                   torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_h_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_gate_up_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(h_act_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(
        h_weighted_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_pool_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(gate_up_output.is_contiguous());
    DG_HOST_ASSERT(grad_h_output.is_contiguous());
    DG_HOST_ASSERT(grad_gate_up_output.is_contiguous());
    DG_HOST_ASSERT(h_act_output.is_contiguous());
    DG_HOST_ASSERT(h_weighted_output.is_contiguous());
    DG_HOST_ASSERT(x_pool_output.is_contiguous());
    DG_HOST_ASSERT(grad_x_pool_output.is_contiguous());
    DG_HOST_ASSERT(
        route_weights.scalar_type() == torch::kFloat ||
        route_weights.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    const auto check_bf16_hidden_pool =
        [&](const torch::Tensor& tensor) {
            DG_HOST_ASSERT(
                tensor.scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(tensor.is_contiguous());
            DG_HOST_ASSERT(
                tensor.sizes() ==
                torch::IntArrayRef(
                    {num_pool_rows, hidden}));
        };
    if (grad_y_unweighted_output.has_value())
        check_bf16_hidden_pool(
            *grad_y_unweighted_output);
    if (down_unweighted_output.has_value()) {
        DG_HOST_ASSERT(
            down_unweighted_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(down_unweighted_output->is_contiguous());
        DG_HOST_ASSERT(down_unweighted_output->dim() == 2);
        DG_HOST_ASSERT(
            down_unweighted_output->size(1) == hidden);
        DG_HOST_ASSERT(
            down_unweighted_output->size(0) > 0 &&
            down_unweighted_output->size(0) <= num_pool_rows);
    }
    if (grad_route_output.has_value()) {
        DG_HOST_ASSERT(
            grad_route_output->scalar_type() ==
            torch::kFloat);
        DG_HOST_ASSERT(grad_route_output->is_contiguous());
        DG_HOST_ASSERT(
            grad_route_output->numel() ==
            num_pool_rows);
    }
    if (route_weight_mode == "post_down") {
        DG_HOST_ASSERT(
            route_weights.scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(
            grad_y_unweighted_output.has_value());
        DG_HOST_ASSERT(
            down_unweighted_output.has_value());
        DG_HOST_ASSERT(grad_route_output.has_value());
    }
    DG_HOST_ASSERT(
        w2_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w2_weights.dim() == 3);
    DG_HOST_ASSERT(w2_weights.size(0) == num_experts);
    DG_HOST_ASSERT(w2_weights.size(1) == hidden);
    DG_HOST_ASSERT(w2_weights.size(2) == intermediate_hidden / 2);
    DG_HOST_ASSERT(w2_weights.is_contiguous());
    DG_HOST_ASSERT(w2_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w2_scales.dim() == 3);
    DG_HOST_ASSERT(w2_scales.size(0) == num_experts);
    DG_HOST_ASSERT(w2_scales.size(1) == hidden);
    DG_HOST_ASSERT(w2_scales.size(2) == intermediate_hidden / gran_k);
    DG_HOST_ASSERT(w2_scales.is_contiguous());
    DG_HOST_ASSERT(
        w2_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, hidden, intermediate_hidden}));
    DG_HOST_ASSERT(
        w2_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w2_dequant_scratch.is_contiguous());
    if (w2_wgrad_output.has_value()) {
        DG_HOST_ASSERT(
            w2_wgrad_output->sizes() ==
            w2_dequant_scratch.sizes());
        DG_HOST_ASSERT(
            w2_wgrad_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(w2_wgrad_output->is_contiguous());
    }
    DG_HOST_ASSERT(
        w13_weights.scalar_type() ==
        torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(w13_weights.dim() == 3);
    DG_HOST_ASSERT(w13_weights.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_weights.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_weights.size(2) == hidden / 2);
    DG_HOST_ASSERT(w13_weights.is_contiguous());
    DG_HOST_ASSERT(w13_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(w13_scales.dim() == 3);
    DG_HOST_ASSERT(w13_scales.size(0) == 2 * num_experts);
    DG_HOST_ASSERT(w13_scales.size(1) == intermediate_hidden);
    DG_HOST_ASSERT(w13_scales.size(2) == hidden / gran_k);
    DG_HOST_ASSERT(w13_scales.is_contiguous());
    DG_HOST_ASSERT(
        w13_dequant_scratch.sizes() ==
        torch::IntArrayRef(
            {num_experts, intermediate_hidden_2, hidden}));
    DG_HOST_ASSERT(
        w13_dequant_scratch.scalar_type() ==
        torch::kBFloat16);
    DG_HOST_ASSERT(w13_dequant_scratch.is_contiguous());
    if (w13_wgrad_output.has_value()) {
        DG_HOST_ASSERT(
            w13_wgrad_output->sizes() ==
            w13_dequant_scratch.sizes());
        DG_HOST_ASSERT(
            w13_wgrad_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(w13_wgrad_output->is_contiguous());
    }
    if (external_residual_mxfp8_weights) {
        DG_HOST_ASSERT(!inline_weight_dequant);
        DG_HOST_ASSERT(
            w2_dgrad_weights->scalar_type() ==
            torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(
            w2_dgrad_weights->sizes() ==
            torch::IntArrayRef(
                {num_experts, intermediate_hidden, hidden}));
        DG_HOST_ASSERT(w2_dgrad_weights->is_contiguous());
        check_sf_layout(
            *w2_dgrad_weights_sf, intermediate_hidden, hidden,
            1, gran_k, num_experts, true, false, torch::kInt);
        DG_HOST_ASSERT(
            w13_dgrad_weights->scalar_type() ==
            torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(
            w13_dgrad_weights->sizes() ==
            torch::IntArrayRef(
                {num_experts, hidden, intermediate_hidden_2}));
        DG_HOST_ASSERT(w13_dgrad_weights->is_contiguous());
        check_sf_layout(
            *w13_dgrad_weights_sf, hidden, intermediate_hidden_2,
            1, gran_k, num_experts, true, false, torch::kInt);
    }
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        expert_counts.numel() ==
        static_cast<int64_t>(num_backward_ranges) * num_experts);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_experts *
            ((hidden / dgrad_block_k) *
                 (intermediate_hidden / block_n) +
             (intermediate_hidden_2 / dgrad_block_k) *
                 (hidden / block_n)) +
            2);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    if (compute_w13_dgrad)
        DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    else
        DG_HOST_ASSERT(!direct_remote_grad_x);
    if (direct_remote_grad_x) {
        DG_HOST_ASSERT(compute_w13_dgrad);
        DG_HOST_ASSERT(num_ranks > 1);
    }
    if (combined_grad_x_output.has_value()) {
        DG_HOST_ASSERT(inline_wgrad && direct_remote_grad_x);
        DG_HOST_ASSERT(
            combined_grad_x_output->scalar_type() ==
            torch::kBFloat16);
        DG_HOST_ASSERT(combined_grad_x_output->is_contiguous());
        DG_HOST_ASSERT(combined_grad_x_output->dim() == 2);
        DG_HOST_ASSERT(combined_grad_x_output->size(1) == hidden);
        DG_HOST_ASSERT(
            combined_grad_x_output->size(0) <=
            num_max_tokens_per_rank);
    }
    DG_HOST_ASSERT(
        !inline_wgrad || !direct_remote_grad_x ||
        combined_grad_x_output.has_value());
    if (!backward_sym_buffer_ptrs.empty()) {
        DG_HOST_ASSERT(backward_grad_y.has_value());
        DG_HOST_ASSERT(backward_topk_weights.has_value());
        DG_HOST_ASSERT(token_src_metadata.has_value());
        DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
        DG_HOST_ASSERT(num_topk > 0);
        DG_HOST_ASSERT(
            backward_grad_y->scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(backward_grad_y->dim() == 2);
        DG_HOST_ASSERT(backward_grad_y->size(0) >= num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_grad_y->size(1) == hidden);
        DG_HOST_ASSERT(backward_grad_y->is_contiguous());
        if (backward_x.has_value()) {
            DG_HOST_ASSERT(
                backward_x->scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(backward_x->dim() == 2);
            DG_HOST_ASSERT(
                backward_x->size(0) >= num_max_tokens_per_rank);
            DG_HOST_ASSERT(backward_x->size(1) == hidden);
            DG_HOST_ASSERT(backward_x->is_contiguous());
        }
        DG_HOST_ASSERT(
            backward_topk_weights->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_topk_weights->dim() == 2);
        DG_HOST_ASSERT(backward_topk_weights->size(0) >=
                       num_max_tokens_per_rank);
        DG_HOST_ASSERT(backward_topk_weights->size(1) == num_topk);
        DG_HOST_ASSERT(backward_topk_weights->is_contiguous());
        if (grad_route_output.has_value()) {
            DG_HOST_ASSERT(backward_grad_route.has_value());
            DG_HOST_ASSERT(
                backward_grad_route->scalar_type() == torch::kFloat);
            DG_HOST_ASSERT(backward_grad_route->dim() == 2);
            DG_HOST_ASSERT(backward_grad_route->size(0) >=
                           num_max_tokens_per_rank);
            DG_HOST_ASSERT(backward_grad_route->size(1) == num_topk);
            DG_HOST_ASSERT(backward_grad_route->is_contiguous());
        }
        DG_HOST_ASSERT(
            token_src_metadata->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(token_src_metadata->dim() == 2);
        // Metadata is an immutable active-layout snapshot. Bucket-only output
        // rows are never visited because expert_counts guards every access.
        DG_HOST_ASSERT(token_src_metadata->size(0) >= num_acts_rows);
        DG_HOST_ASSERT(token_src_metadata->size(1) == 3);
        DG_HOST_ASSERT(token_src_metadata->is_contiguous());
    }

    check_sf_layout(
        l1_weights_sf, intermediate_hidden_2, hidden,
        1, gran_k, num_experts, true, false, torch::kInt);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        (residual_mxfp8_dgrad
             ? load_block_m * block_k +
                   sf_block_m * static_cast<int>(sizeof(uint32_t))
             : 0) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        2 * num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        2 * static_cast<int>(sizeof(uint64_t)) +
        (residual_mxfp8_dgrad
             ? dgrad_block_k * (load_block_n / 2) *
                       static_cast<int>(sizeof(int8_t)) +
                   dgrad_block_k * (load_block_n / gran_k) *
                       static_cast<int>(sizeof(float)) +
                   2 * static_cast<int>(sizeof(uint64_t))
             : 0) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(32, (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size = std::max({
        align(smem_fixed + num_stages * smem_per_stage, 1024),
        inline_wgrad ? wgrad_smem_size : 0,
        enable_k3_mxfp8_three_term_wgrad
            ? static_cast<int>(kK3MxFp8WgradStreamingSmemBytes)
            : 0});
    DG_HOST_ASSERT(num_stages >= 2);
    DG_HOST_ASSERT(smem_size <= smem_capacity);

    auto tensor_map_acts = make_tma_2d_desc(
        acts, hidden, num_acts_rows,
        block_k, load_block_m,
        static_cast<int>(acts.stride(-2)), 128);
    auto tensor_map_acts_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, acts_sf,
        num_sf_pool_rows, hidden,
        sf_block_m, gran_k, 1, 0);
    auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights, hidden, num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(l1_weights.stride(-2)), 128);
    auto tensor_map_l1_weights_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, l1_weights_sf,
        intermediate_hidden_2, hidden,
        block_n, gran_k, num_experts, 0);
    auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2,
        static_cast<int>(gate_up_output.size(0)),
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    if (build_residual_mxfp8_weights) {
        DG_HOST_ASSERT(num_acts_rows % block_m == 0);
        const int residual_sf_rows =
            num_acts_rows / block_m * sf_block_m;

        const auto w2_act_storage =
            grad_x_pool_output.view(torch::kUInt8).reshape({-1});
        const int64_t w2_act_values =
            static_cast<int64_t>(num_acts_rows) * hidden;
        DG_HOST_ASSERT(w2_act_storage.numel() >= 2 * w2_act_values);
        const auto w2_acts =
            w2_act_storage.narrow(0, 0, 2 * w2_act_values)
                .view(torch::kFloat8_e4m3fn)
                .view({2 * num_acts_rows, hidden});
        // W13 dequant/output storage is dead until W13 wgrad. Unlike
        // grad_gate_up, it cannot be overwritten by W2's SiTU epilogue while
        // later W2 tiles still consume their activation scales. Keep those
        // scales after the exact W13-weight alias so both operands can coexist
        // until W2 dgrad retires.
        const auto w13_output_storage =
            w13_dequant_scratch.view(torch::kUInt8).reshape({-1});
        const int64_t w13_weight_alias_values =
            static_cast<int64_t>(num_experts) * hidden *
            intermediate_hidden_2;
        const int64_t w13_weight_alias_bytes =
            w13_weight_alias_values + w13_weight_alias_values / gran_k;
        const int64_t w2_sf_bytes =
            static_cast<int64_t>(residual_sf_rows) * hidden / gran_k;
        DG_HOST_ASSERT(
            w13_output_storage.numel() >=
            w13_weight_alias_bytes + 2 * w2_sf_bytes);
        const auto w2_sfs =
            w13_output_storage.narrow(
                0, w13_weight_alias_bytes, 2 * w2_sf_bytes)
                .view(torch::kInt32);

        const int64_t w13_act_values =
            static_cast<int64_t>(num_acts_rows) *
            intermediate_hidden_2;
        const int64_t w13_sf_bytes =
            static_cast<int64_t>(residual_sf_rows) *
            intermediate_hidden_2 / gran_k;

        // Prepared gate/up eliminates replay, so its four descriptor slots
        // can carry both primary+residual activation pairs without growing
        // the CUDA kernel-parameter footprint.
        tensor_map_acts = make_tma_2d_desc(
            w2_acts, hidden, 2 * num_acts_rows,
            dgrad_block_k, load_block_m, hidden, 128);
        tensor_map_acts_sf = make_tma_sf_desc(
            cute::UMMA::Major::MN, w2_sfs,
            residual_sf_rows, hidden,
            sf_block_m, gran_k, 2, 0);
        if (early_w2_wgrad) {
            // W2 wgrad retires both BF16 operands before W13 dgrad begins.
            // Reuse those now-dead, row-scaled buffers for W13's primary and
            // residual MXFP8 activation planes.  This avoids both the fixed
            // dW2-sized cache ceiling and any additional allocation.
            const auto grad_ye_storage =
                grad_ye.view(torch::kUInt8).reshape({-1});
            const auto grad_h_storage =
                grad_h_output.view(torch::kUInt8).reshape({-1});
            DG_HOST_ASSERT(
                grad_ye_storage.numel() >=
                w13_act_values + 2 * w13_sf_bytes);
            DG_HOST_ASSERT(
                grad_h_storage.numel() >= w13_act_values);
            const auto w13_primary =
                grad_ye_storage.narrow(0, 0, w13_act_values)
                    .view(torch::kFloat8_e4m3fn)
                    .view({num_acts_rows, intermediate_hidden_2});
            const auto w13_residual =
                grad_h_storage.narrow(0, 0, w13_act_values)
                    .view(torch::kFloat8_e4m3fn)
                    .view({num_acts_rows, intermediate_hidden_2});
            const auto w13_sfs =
                grad_ye_storage
                    .narrow(
                        0, w13_act_values,
                        2 * w13_sf_bytes)
                    .view(torch::kInt32);
            tensor_map_l1_weights = make_tma_2d_desc(
                w13_primary, intermediate_hidden_2,
                num_acts_rows, dgrad_block_k, load_block_m,
                intermediate_hidden_2, 128);
            tensor_map_gate_up = make_tma_2d_desc(
                w13_residual, intermediate_hidden_2,
                num_acts_rows, dgrad_block_k, load_block_m,
                intermediate_hidden_2, 128);
            tensor_map_l1_weights_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, w13_sfs,
                residual_sf_rows, intermediate_hidden_2,
                sf_block_m, gran_k, 2, 0);
        } else {
            const auto w13_storage =
                w2_dequant_scratch.view(torch::kUInt8).reshape({-1});
            DG_HOST_ASSERT(
                w13_storage.numel() >=
                2 * w13_act_values + 2 * w13_sf_bytes);
            const auto w13_acts =
                w13_storage.narrow(0, 0, 2 * w13_act_values)
                    .view(torch::kFloat8_e4m3fn)
                    .view({2 * num_acts_rows, intermediate_hidden_2});
            const auto w13_sfs =
                w13_storage
                    .narrow(
                        0, 2 * w13_act_values,
                        2 * w13_sf_bytes)
                    .view(torch::kInt32);
            tensor_map_l1_weights = make_tma_2d_desc(
                w13_acts, intermediate_hidden_2,
                2 * num_acts_rows, dgrad_block_k, load_block_m,
                intermediate_hidden_2, 128);
            tensor_map_l1_weights_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, w13_sfs,
                residual_sf_rows, intermediate_hidden_2,
                sf_block_m, gran_k, 2, 0);
        }
    }
    const auto tensor_map_w2_dequant = make_tma_2d_desc(
        w2_dequant_scratch, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    DG_HOST_ASSERT(hidden % (gran_k * 4) == 0);
    const int64_t w2_cached_q_bytes =
        static_cast<int64_t>(num_experts) * intermediate_hidden * hidden;
    const int64_t w2_cached_sf_words =
        w2_cached_q_bytes / (gran_k * 4);
    const int64_t w2_cached_sf_bytes =
        w2_cached_sf_words * static_cast<int64_t>(sizeof(int32_t));
    const auto w2_cached_storage =
        w2_dequant_scratch.view(torch::kUInt8).reshape({-1});
    DG_HOST_ASSERT(
        w2_cached_storage.numel() >=
        w2_cached_q_bytes + w2_cached_sf_bytes);
    const auto w2_cached_q =
        w2_cached_storage
            .narrow(0, 0, w2_cached_q_bytes)
            .view(torch::kFloat8_e4m3fn)
            .view({num_experts, intermediate_hidden, hidden});
    const auto w2_cached_sf =
        w2_cached_storage
            .narrow(0, w2_cached_q_bytes, w2_cached_sf_bytes)
            .view(torch::kInt32)
            .view({
                num_experts, hidden / (gran_k * 4),
                intermediate_hidden});
    DG_HOST_ASSERT(w2_cached_q.stride(-2) == hidden);
    DG_HOST_ASSERT(w2_cached_sf.stride(-2) == intermediate_hidden);
    const auto tensor_map_w2_cached_q = make_tma_2d_desc(
        w2_cached_q, hidden,
        num_experts * intermediate_hidden,
        block_k, load_block_n, hidden, 128);
    const auto tensor_map_w2_cached_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, w2_cached_sf,
        intermediate_hidden, hidden,
        block_n, gran_k, num_experts, 0);
    const auto tensor_map_w2_dgrad_weights =
        external_residual_mxfp8_weights
        ? make_tma_2d_desc(
              *w2_dgrad_weights, hidden,
              num_experts * intermediate_hidden,
              block_k, load_block_n,
              static_cast<int>(w2_dgrad_weights->stride(-2)), 128)
        : tensor_map_w2_cached_q;
    const auto tensor_map_w2_dgrad_weights_sf =
        external_residual_mxfp8_weights
        ? make_tma_sf_desc(
              cute::UMMA::Major::MN, *w2_dgrad_weights_sf,
              intermediate_hidden, hidden,
              block_n, gran_k, num_experts, 0)
        : tensor_map_w2_cached_sf;
    const int native_weight_tma_n = load_block_n;
    const auto tensor_map_w2_weights = make_tma_2d_desc(
        w2_weights, intermediate_hidden / 2,
        num_experts * hidden,
        native_weight_tma_n / 2,
        residual_mxfp8_dgrad ? dgrad_block_k : 256,
        intermediate_hidden / 2, 0);
    const auto tensor_map_w2_scales = make_tma_2d_desc(
        w2_scales, intermediate_hidden / gran_k,
        num_experts * hidden,
        native_weight_tma_n / gran_k,
        residual_mxfp8_dgrad ? dgrad_block_k : 256,
        intermediate_hidden / gran_k, 0);
    const auto tensor_map_w13_dequant = make_tma_2d_desc(
        w13_dequant_scratch, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        hidden, 128);
    DG_HOST_ASSERT(intermediate_hidden_2 % (gran_k * 4) == 0);
    const int64_t w13_cached_q_bytes =
        static_cast<int64_t>(num_experts) * hidden * intermediate_hidden_2;
    const int64_t w13_cached_sf_words =
        w13_cached_q_bytes / (gran_k * 4);
    const int64_t w13_cached_sf_bytes =
        w13_cached_sf_words * static_cast<int64_t>(sizeof(int32_t));
    const auto w13_cached_storage =
        w13_dequant_scratch.view(torch::kUInt8).reshape({-1});
    DG_HOST_ASSERT(
        w13_cached_storage.numel() >=
        w13_cached_q_bytes + w13_cached_sf_bytes);
    const auto w13_cached_q =
        w13_cached_storage
            .narrow(0, 0, w13_cached_q_bytes)
            .view(torch::kFloat8_e4m3fn)
            .view({num_experts, hidden, intermediate_hidden_2});
    const auto w13_cached_sf =
        w13_cached_storage
            .narrow(0, w13_cached_q_bytes, w13_cached_sf_bytes)
            .view(torch::kInt32)
            .view({
                num_experts,
                intermediate_hidden_2 / (gran_k * 4), hidden});
    DG_HOST_ASSERT(w13_cached_q.stride(-2) == intermediate_hidden_2);
    DG_HOST_ASSERT(w13_cached_sf.stride(-2) == hidden);
    const auto tensor_map_w13_cached_q = make_tma_2d_desc(
        w13_cached_q, intermediate_hidden_2,
        num_experts * hidden,
        block_k, load_block_n, intermediate_hidden_2, 128);
    const auto tensor_map_w13_cached_sf = make_tma_sf_desc(
        cute::UMMA::Major::MN, w13_cached_sf,
        hidden, intermediate_hidden_2,
        block_n, gran_k, num_experts, 0);
    const auto tensor_map_w13_dgrad_weights =
        external_residual_mxfp8_weights
        ? make_tma_2d_desc(
              *w13_dgrad_weights, intermediate_hidden_2,
              num_experts * hidden,
              block_k, load_block_n,
              static_cast<int>(w13_dgrad_weights->stride(-2)), 128)
        : tensor_map_w13_cached_q;
    const auto tensor_map_w13_dgrad_weights_sf =
        external_residual_mxfp8_weights
        ? make_tma_sf_desc(
              cute::UMMA::Major::MN, *w13_dgrad_weights_sf,
              hidden, intermediate_hidden_2,
              block_n, gran_k, num_experts, 0)
        : tensor_map_w13_cached_sf;
    const auto tensor_map_w13_weights = make_tma_2d_desc(
        w13_weights, hidden / 2,
        num_experts * intermediate_hidden_2,
        native_weight_tma_n / 2,
        residual_mxfp8_dgrad ? dgrad_block_k : 256,
        hidden / 2, 0);
    const auto tensor_map_w13_scales = make_tma_2d_desc(
        w13_scales, hidden / gran_k,
        num_experts * intermediate_hidden_2,
        native_weight_tma_n / gran_k,
        residual_mxfp8_dgrad ? dgrad_block_k : 256,
        hidden / gran_k, 0);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_gate_up_output.stride(-2)), 128);
    const auto tensor_map_w2_wgrad_a = inline_wgrad
        ? make_tma_a_desc(
              cute::UMMA::Major::MN, grad_ye, hidden, num_pool_rows,
              wgrad_block_m, wgrad_block_k,
              static_cast<int>(grad_ye.stride(0)), 1, wgrad_swizzle)
        : tensor_map_grad_ye;
    const auto tensor_map_w2_wgrad_b = inline_wgrad
        ? make_tma_b_desc(
              cute::UMMA::Major::MN, h_weighted_output,
              intermediate_hidden, num_pool_rows,
              wgrad_load_block_n, wgrad_block_k,
              static_cast<int>(h_weighted_output.stride(0)), 1,
              wgrad_swizzle)
        : tensor_map_grad_ye;
    auto tensor_map_w2_wgrad_d = inline_wgrad
        ? make_tma_cd_desc(
              w2_wgrad_output.value_or(w2_dequant_scratch),
              hidden, intermediate_hidden,
              wgrad_block_m, wgrad_store_block_n,
              static_cast<int>(
                  w2_wgrad_output
                      .value_or(w2_dequant_scratch)
                      .stride(1)),
              num_experts, wgrad_swizzle)
        : tensor_map_w2_dequant;
    const auto tensor_map_w13_wgrad_a = inline_wgrad
        ? make_tma_a_desc(
              cute::UMMA::Major::MN, grad_gate_up_output,
              intermediate_hidden_2, num_pool_rows,
              wgrad_block_m, wgrad_block_k,
              static_cast<int>(grad_gate_up_output.stride(0)), 1,
              wgrad_swizzle)
        : tensor_map_grad_gate_up;
    const auto tensor_map_w13_wgrad_b = inline_wgrad
        ? make_tma_b_desc(
              cute::UMMA::Major::MN, x_pool_output, hidden, num_pool_rows,
              wgrad_load_block_n, wgrad_block_k,
              static_cast<int>(x_pool_output.stride(0)), 1, wgrad_swizzle)
        : tensor_map_grad_gate_up;
    auto tensor_map_w13_wgrad_d = inline_wgrad
        ? make_tma_cd_desc(
              w13_wgrad_output.value_or(w13_dequant_scratch),
              intermediate_hidden_2, hidden,
              wgrad_block_m, wgrad_store_block_n,
              static_cast<int>(
                  w13_wgrad_output
                      .value_or(w13_dequant_scratch)
                      .stride(1)),
              num_experts, wgrad_swizzle)
        : tensor_map_w13_dequant;
    // The exact 32x64 output descriptors below deliberately replace the local
    // MXFP8 variables, but the one-way hybrid still executes BF16 dW2 first.
    // Preserve its 128x256 D map (and the BF16 fallback dW13 map) explicitly;
    // an AuxSlot must never reinterpret an exact D map as legacy_map.
    const auto tensor_map_w2_wgrad_d_bf16 = tensor_map_w2_wgrad_d;
    const auto tensor_map_w13_wgrad_d_bf16 = tensor_map_w13_wgrad_d;
    K3MxFp8WgradTensorMapPack<CUtensorMap>
        k3_mxfp8_wgrad_tensor_maps{};
    static_assert(k3_mxfp8_wgrad_tensor_map_pack_abi<CUtensorMap>());

    if (enable_k3_mxfp8_three_term_wgrad) {
        DG_HOST_ASSERT(num_acts_rows % block_m == 0);
        DG_HOST_ASSERT(acts.scalar_type() == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(acts.is_contiguous());
        DG_HOST_ASSERT(
            multi_range_backward ||
            acts.numel() >=
                static_cast<int64_t>(num_acts_rows) * hidden);

        // Device aliases are sized by the active captured extent, not by the
        // larger rank-uniform pool bucket. Exact X aliases at most one of the
        // two saved H pools; choose and prove the other one here exactly as
        // the device selector does.
        const bool x_aliases_grad_y =
            x_pool_output.data_ptr() ==
            grad_y_unweighted_output->data_ptr();
        const torch::Tensor& safe_h_arena = x_aliases_grad_y
            ? *down_unweighted_output
            : *grad_y_unweighted_output;
        DG_HOST_ASSERT(safe_h_arena.dim() == 2);
        DG_HOST_ASSERT(safe_h_arena.size(0) >= num_acts_rows);
        DG_HOST_ASSERT(safe_h_arena.size(1) == hidden);
        DG_HOST_ASSERT(safe_h_arena.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(safe_h_arena.is_contiguous());

        const auto tensor_bytes = [](const torch::Tensor& tensor) {
            return static_cast<uint64_t>(tensor.numel()) *
                static_cast<uint64_t>(tensor.element_size());
        };
        const auto tensors_disjoint = [&] (
                const torch::Tensor& lhs,
                const torch::Tensor& rhs) {
            const uint64_t lhs_begin = reinterpret_cast<uint64_t>(
                lhs.data_ptr());
            const uint64_t rhs_begin = reinterpret_cast<uint64_t>(
                rhs.data_ptr());
            return lhs_begin + tensor_bytes(lhs) <= rhs_begin ||
                rhs_begin + tensor_bytes(rhs) <= lhs_begin;
        };
        const auto require_disjoint = [&] (
                const torch::Tensor& lhs,
                const torch::Tensor& rhs) {
            DG_HOST_ASSERT(tensors_disjoint(lhs, rhs));
        };

        require_disjoint(safe_h_arena, x_pool_output);
        require_disjoint(safe_h_arena, grad_ye);
        require_disjoint(safe_h_arena, h_weighted_output);
        require_disjoint(safe_h_arena, grad_gate_up_output);
        require_disjoint(safe_h_arena, acts);
        require_disjoint(grad_ye, h_weighted_output);
        require_disjoint(grad_ye, grad_gate_up_output);
        require_disjoint(grad_ye, x_pool_output);
        require_disjoint(grad_ye, acts);
        require_disjoint(h_weighted_output, grad_gate_up_output);
        require_disjoint(h_weighted_output, x_pool_output);
        require_disjoint(h_weighted_output, acts);
        require_disjoint(grad_gate_up_output, acts);
        require_disjoint(x_pool_output, acts);

        // D must be explicit: legacy fallback descriptors point at dequant
        // scratch, which may be an activation arena in accumulated launches.
        // Both dW tensors remain live accumulation state and therefore may
        // overlap none of the phase values, scales, sources, or each other.
        const auto& dw2_output = *w2_wgrad_output;
        const auto& dw13_output = *w13_wgrad_output;
        if (!accumulate_wgrad) {
            DG_HOST_ASSERT(!inline_weight_dequant);
            DG_HOST_ASSERT(!phase_ordered_weight_dequant);
            DG_HOST_ASSERT(
                dw2_output.data_ptr() ==
                w2_dequant_scratch.data_ptr());
            DG_HOST_ASSERT(
                dw13_output.data_ptr() ==
                w13_dequant_scratch.data_ptr());
            DG_HOST_ASSERT(clear_empty_wgrad_expert_outputs);
        }
        for (const auto* arena : {
                 &safe_h_arena, &grad_ye, &h_weighted_output,
                 &grad_gate_up_output, &x_pool_output, &acts}) {
            require_disjoint(dw2_output, *arena);
            require_disjoint(dw13_output, *arena);
        }
        require_disjoint(dw2_output, dw13_output);

        // Reuse the same constexpr layout calculator as the CUDA producer;
        // duplicated host arithmetic could silently bind valid TensorMaps to
        // different bytes than the device-side scale-storage records.
        const auto dw2_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(intermediate_hidden),
            static_cast<uint32_t>(num_acts_rows),
            static_cast<uint32_t>(block_m));
        const auto dw13_scale_layout = k3_mxfp8_wgrad_scale_arena_layout(
            static_cast<uint32_t>(intermediate_hidden_2),
            static_cast<uint32_t>(hidden),
            static_cast<uint32_t>(num_acts_rows),
            static_cast<uint32_t>(block_m));
        DG_HOST_ASSERT(
            dw2_scale_layout.packed_row_capacity ==
            dw13_scale_layout.packed_row_capacity);
        const int64_t packed_scale_rows =
            static_cast<int64_t>(dw2_scale_layout.packed_row_capacity);
        const int64_t acts_bytes = multi_range_backward
            ? acts.numel()
            : static_cast<int64_t>(num_acts_rows) * hidden;
        const int64_t dw13_scale_phase_offset =
            static_cast<int64_t>(
                k3_mxfp8_wgrad_next_scale_phase_offset(
                    dw2_scale_layout));
        DG_HOST_ASSERT(
            dw13_scale_phase_offset >=
                static_cast<int64_t>(
                    dw2_scale_layout.raw_bytes +
                    dw2_scale_layout.packed_bytes) &&
            dw13_scale_phase_offset % 128 == 0);
        DG_HOST_ASSERT(
            k3_mxfp8_wgrad_two_phase_scale_bytes(
                dw2_scale_layout, dw13_scale_layout) <=
            static_cast<uint64_t>(acts_bytes));

        constexpr int mxfp8_wgrad_store_block_m =
            static_cast<int>(kK3MxFp8WgradStoreBlockM);
        constexpr int mxfp8_wgrad_store_block_n = 64;
        tensor_map_w2_wgrad_d = make_tma_cd_desc(
            dw2_output, hidden, intermediate_hidden,
            mxfp8_wgrad_store_block_m, mxfp8_wgrad_store_block_n,
            static_cast<int>(dw2_output.stride(1)),
            num_experts, 128);
        tensor_map_w13_wgrad_d = make_tma_cd_desc(
            dw13_output, intermediate_hidden_2, hidden,
            mxfp8_wgrad_store_block_m, mxfp8_wgrad_store_block_n,
            static_cast<int>(dw13_output.stride(1)),
            num_experts, 128);

        // Bind final addresses on the host. dW2-B uses two I-wide planes in
        // each retired grad_ye row but retains grad_ye's physical 2H byte row
        // pitch. dW13-A-primary exposes the logical 2I feature extent through
        // that same 2H pitch. Consequently every expert's B destination stays
        // inside its own retired A rows while independent producer clusters
        // progress at different rates. dW13 scales start after the aligned
        // complete dW2 scale footprint. The exact suffix consumes this
        // grid-constant pack directly, without device TensorMap mutation,
        // descriptor state, or another allocation.
        const auto safe_h_bytes =
            safe_h_arena.view(torch::kUInt8).reshape({-1});
        const auto grad_ye_bytes =
            grad_ye.view(torch::kUInt8).reshape({-1});
        const auto h_weighted_bytes =
            h_weighted_output.view(torch::kUInt8).reshape({-1});
        const auto scale_arena_bytes =
            acts.view(torch::kUInt8).reshape({-1});
        const int64_t h_plane_bytes =
            static_cast<int64_t>(num_acts_rows) * hidden;
        const int64_t i_plane_bytes =
            static_cast<int64_t>(num_acts_rows) * intermediate_hidden;
        const int retired_grad_ye_row_bytes = 2 * hidden;
        DG_HOST_ASSERT(safe_h_bytes.numel() >= 2 * h_plane_bytes);
        DG_HOST_ASSERT(intermediate_hidden_2 <= retired_grad_ye_row_bytes);
        DG_HOST_ASSERT(
            grad_ye_bytes.numel() >=
                static_cast<int64_t>(num_acts_rows) *
                    retired_grad_ye_row_bytes);
        DG_HOST_ASSERT(h_weighted_bytes.numel() >= 2 * i_plane_bytes);
        DG_HOST_ASSERT(scale_arena_bytes.numel() >= acts_bytes);

        const auto make_value_map = [&] (
                const torch::Tensor& storage, int64_t byte_offset,
                int feature, int row_stride) {
            DG_HOST_ASSERT(row_stride >= feature);
            const int64_t value_bytes = num_acts_rows == 0
                ? 0
                : static_cast<int64_t>(num_acts_rows - 1) * row_stride +
                    feature;
            DG_HOST_ASSERT(
                byte_offset >= 0 &&
                byte_offset + value_bytes <= storage.numel());
            const auto values = storage
                .narrow(0, byte_offset, value_bytes)
                .view(torch::kFloat8_e4m3fn);
            // The exact body computes one 256x128 logical task as two
            // 128x128 output panels. Each CTA consumes 64 A rows, so every
            // exact value descriptor uses a 64-byte FP8 atom. B retains its
            // 128-row tile as two adjacent atoms, and dW13-B may continue to
            // alias the dW2-A descriptors without another ABI slot.
            return make_tma_2d_desc(
                values, feature, num_acts_rows,
                64, 128, row_stride, 64);
        };
        const auto make_scale_map = [&] (
                int64_t byte_offset, int feature, int tile_inner) {
            // The physical feature dimension is already in UTCCP-native
            // 128-word tiles.  It stays a linear [packed_row, feature]
            // TensorMap: producers apply the intra-tile permutation, and the
            // exact body therefore TMA-loads the native words without a
            // software transpose or any extra storage.
            DG_HOST_ASSERT(feature % 128 == 0);
            DG_HOST_ASSERT(tile_inner % 128 == 0);
            const int64_t scale_bytes =
                packed_scale_rows * static_cast<int64_t>(feature) *
                static_cast<int64_t>(sizeof(uint32_t));
            DG_HOST_ASSERT(byte_offset % alignof(uint32_t) == 0);
            DG_HOST_ASSERT(
                byte_offset >= 0 &&
                byte_offset + scale_bytes <= scale_arena_bytes.numel());
            const auto scales = scale_arena_bytes
                .narrow(0, byte_offset, scale_bytes)
                .view(torch::kInt32)
                .view({packed_scale_rows, feature});
            return make_tma_2d_desc(
                scales, feature, static_cast<int>(packed_scale_rows),
                tile_inner, 1, feature, 0);
        };
        const auto make_stream_source_map = [&] (
                const torch::Tensor& source, int feature,
                int feature_tile) {
            DG_HOST_ASSERT(source.scalar_type() == torch::kBFloat16);
            DG_HOST_ASSERT(source.is_contiguous());
            DG_HOST_ASSERT(source.dim() == 2);
            DG_HOST_ASSERT(source.size(0) >= num_acts_rows);
            DG_HOST_ASSERT(source.size(1) == feature);
            return make_tma_2d_desc(
                source, feature, num_acts_rows,
                feature_tile,
                kK3MxFp8DW13QuantRowsPerGroup,
                feature, 0);
        };
        const auto make_stream_value_map = [&] (
                const torch::Tensor& storage, int64_t byte_offset,
                int feature, int row_stride, int feature_tile) {
            DG_HOST_ASSERT(row_stride >= feature);
            const int64_t value_bytes = num_acts_rows == 0
                ? 0
                : static_cast<int64_t>(num_acts_rows - 1) * row_stride +
                    feature;
            DG_HOST_ASSERT(
                byte_offset >= 0 &&
                byte_offset + value_bytes <= storage.numel());
            const auto values = storage
                .narrow(0, byte_offset, value_bytes)
                .view(torch::kFloat8_e4m3fn);
            return make_tma_2d_desc(
                values, feature, num_acts_rows,
                feature_tile,
                kK3MxFp8DW13QuantRowsPerGroup,
                row_stride, 0);
        };
        const auto make_ring_value_map = [&] (
                uint64_t address, int feature, int row_stride) {
            const int64_t usable_rows =
                static_cast<int64_t>(num_max_tokens_per_rank) -
                kK3MxFp8RingReservedRows;
            const int64_t ring_rows = usable_rows <= 0
                ? 0 : (usable_rows / block_m) * block_m;
            DG_HOST_ASSERT(
                num_max_tokens_per_rank > kK3MxFp8RingReservedRows &&
                address != 0u && ring_rows >= block_m &&
                row_stride >= feature);
            const int64_t storage_bytes = ring_rows * row_stride;
            const auto storage = torch::from_blob(
                reinterpret_cast<void*>(address), {storage_bytes},
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
                    .device(backward_grad_y->device()));
            const auto values = storage.view(torch::kFloat8_e4m3fn);
            // The rolling consumer gathers one physical group-32 fragment at
            // a time.  This descriptor box is the actual global transaction
            // shape; the later tma::copy template controls only destination
            // staging and cannot crop a 128-row descriptor box to 32 rows.
            return make_tma_2d_desc(
                values, feature, static_cast<int>(ring_rows),
                64, 32, row_stride, 64);
        };

        auto& exact_maps = k3_mxfp8_wgrad_tensor_maps;
        exact_maps.maps[kK3MxFp8DW2ValueAPrimaryMap] =
            make_value_map(safe_h_bytes, 0, hidden, hidden);
        exact_maps.maps[kK3MxFp8DW2ValueAResidualMap] =
            make_value_map(
                safe_h_bytes, h_plane_bytes, hidden, hidden);
        exact_maps.maps[kK3MxFp8DW2ValueBPrimaryMap] =
            make_value_map(
                grad_ye_bytes, 0, intermediate_hidden,
                retired_grad_ye_row_bytes);
        exact_maps.maps[kK3MxFp8DW2ValueBResidualMap] =
            make_value_map(
                grad_ye_bytes, intermediate_hidden, intermediate_hidden,
                retired_grad_ye_row_bytes);
        exact_maps.maps[kK3MxFp8DW2ScaleAPrimaryMap] =
            make_scale_map(
                dw2_scale_layout.raw_bytes +
                    dw2_scale_layout.packed_a_primary,
                hidden, 256);
        exact_maps.maps[kK3MxFp8DW2ScaleAResidualMap] =
            make_scale_map(
                dw2_scale_layout.raw_bytes +
                    dw2_scale_layout.packed_a_residual,
                hidden, 256);
        exact_maps.maps[kK3MxFp8DW2ScaleBPrimaryMap] =
            make_scale_map(
                dw2_scale_layout.raw_bytes +
                    dw2_scale_layout.packed_b_primary,
                intermediate_hidden, 128);
        exact_maps.maps[kK3MxFp8DW2ScaleBResidualMap] =
            make_scale_map(
                dw2_scale_layout.raw_bytes +
                    dw2_scale_layout.packed_b_residual,
                intermediate_hidden, 128);

        exact_maps.maps[kK3MxFp8DW13ValueAPrimaryMap] =
            make_value_map(
                grad_ye_bytes, 0, intermediate_hidden_2,
                retired_grad_ye_row_bytes);
        exact_maps.maps[kK3MxFp8DW13ValueAResidualMap] =
            make_value_map(
                h_weighted_bytes, 0, intermediate_hidden_2,
                intermediate_hidden_2);
        exact_maps.maps[kK3MxFp8DW13ValueBPrimaryMap] =
            exact_maps.maps[kK3MxFp8DW2ValueAPrimaryMap];
        exact_maps.maps[kK3MxFp8DW13ValueBResidualMap] =
            exact_maps.maps[kK3MxFp8DW2ValueAResidualMap];
        exact_maps.maps[kK3MxFp8DW13ScaleAPrimaryMap] =
            make_scale_map(
                dw13_scale_phase_offset +
                    dw13_scale_layout.raw_bytes +
                    dw13_scale_layout.packed_a_primary,
                intermediate_hidden_2, 256);
        exact_maps.maps[kK3MxFp8DW13ScaleAResidualMap] =
            make_scale_map(
                dw13_scale_phase_offset +
                    dw13_scale_layout.raw_bytes +
                    dw13_scale_layout.packed_a_residual,
                intermediate_hidden_2, 256);
        exact_maps.maps[kK3MxFp8DW13ScaleBPrimaryMap] =
            make_scale_map(
                dw13_scale_phase_offset +
                    dw13_scale_layout.raw_bytes +
                    dw13_scale_layout.packed_b_primary,
                hidden, 128);
        exact_maps.maps[kK3MxFp8DW13ScaleBResidualMap] =
            make_scale_map(
                dw13_scale_phase_offset +
                    dw13_scale_layout.raw_bytes +
                    dw13_scale_layout.packed_b_residual,
                hidden, 128);
        exact_maps.maps[kK3MxFp8DW13ProducerSourceAMap] =
            make_stream_source_map(
                grad_gate_up_output, intermediate_hidden_2,
                kK3MxFp8DW13QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW13ProducerSourceBMap] =
            make_stream_source_map(
                x_pool_output, hidden,
                kK3MxFp8DW13QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW13ProducerValueAPrimaryMap] =
            make_stream_value_map(
                grad_ye_bytes, 0, intermediate_hidden_2,
                retired_grad_ye_row_bytes,
                kK3MxFp8DW13QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW13ProducerValueAResidualMap] =
            make_stream_value_map(
                h_weighted_bytes, 0, intermediate_hidden_2,
                intermediate_hidden_2,
                kK3MxFp8DW13QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW13ProducerValueBPrimaryMap] =
            make_stream_value_map(
                safe_h_bytes, 0, hidden, hidden,
                kK3MxFp8DW13QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW13ProducerValueBResidualMap] =
            make_stream_value_map(
                safe_h_bytes, h_plane_bytes, hidden, hidden,
                kK3MxFp8DW13QuantFeatureTile);

        // The W13-dgrad overlap engine has only one borrowed warpgroup and a
        // 32x64 scratch tile.  Address-specialized producer maps let its
        // elected lane issue one source load and two value stores per exact
        // group instead of one one-dimensional transaction per row.
        exact_maps.maps[kK3MxFp8DW2ProducerSourceAMap] =
            make_stream_source_map(
                grad_ye, hidden, kK3MxFp8DW2QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW2ProducerSourceBMap] =
            make_stream_source_map(
                h_weighted_output, intermediate_hidden,
                kK3MxFp8DW2QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW2ProducerValueAPrimaryMap] =
            make_stream_value_map(
                safe_h_bytes, 0, hidden, hidden,
                kK3MxFp8DW2QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW2ProducerValueAResidualMap] =
            make_stream_value_map(
                safe_h_bytes, h_plane_bytes, hidden, hidden,
                kK3MxFp8DW2QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW2ProducerValueBPrimaryMap] =
            make_stream_value_map(
                grad_ye_bytes, 0, intermediate_hidden,
                retired_grad_ye_row_bytes,
                kK3MxFp8DW2QuantFeatureTile);
        exact_maps.maps[kK3MxFp8DW2ProducerValueBResidualMap] =
            make_stream_value_map(
                grad_ye_bytes, intermediate_hidden, intermediate_hidden,
                retired_grad_ye_row_bytes,
                kK3MxFp8DW2QuantFeatureTile);
        if (enable_k3_mxfp8_exact_epilogue_ring) {
            const uint64_t combine_begin =
                reinterpret_cast<uint64_t>(
                    backward_grad_y->data_ptr());
            const uint64_t combine_plane_bytes =
                static_cast<uint64_t>(num_max_tokens_per_rank) *
                retired_grad_ye_row_bytes;
            const uint64_t ring_row_base_bytes =
                static_cast<uint64_t>(kK3MxFp8RingReservedRows) *
                retired_grad_ye_row_bytes;
            // Ring mode never executes the streaming dW13-A value stores.
            // Rebind those two immutable pack slots to the physical ring
            // row pitch instead of extending the parent with a 30-map pack.
            exact_maps.maps[kK3MxFp8DW13RingValueAPrimaryMap] =
                make_ring_value_map(
                    combine_begin +
                        kK3MxFp8EpilogueScratchPrimaryPlane *
                            combine_plane_bytes +
                        ring_row_base_bytes,
                    intermediate_hidden_2,
                    retired_grad_ye_row_bytes);
            exact_maps.maps[kK3MxFp8DW13RingValueAResidualMap] =
                make_ring_value_map(
                    combine_begin +
                        kK3MxFp8EpilogueScratchResidualPlane *
                            combine_plane_bytes +
                        ring_row_base_bytes,
                    intermediate_hidden_2,
                    retired_grad_ye_row_bytes);
            exact_maps.maps[kK3MxFp8DW13RingValueBPrimaryMap] =
                make_ring_value_map(
                    combine_begin +
                        kK3MxFp8DW13BScratchPrimaryPlane *
                            combine_plane_bytes +
                        ring_row_base_bytes,
                    hidden, retired_grad_ye_row_bytes);
            exact_maps.maps[kK3MxFp8DW13RingValueBResidualMap] =
                make_ring_value_map(
                    combine_begin +
                        kK3MxFp8DW13BScratchResidualPlane *
                            combine_plane_bytes +
                        ring_row_base_bytes,
                    hidden, retired_grad_ye_row_bytes);
        }
    }

    DG_HOST_ASSERT(num_sms % 2 == 0);
    constexpr int num_trace_sites = 22;
    constexpr int num_trace_values = 5;
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(
            kernel_trace->scalar_type() == torch::kInt64 &&
            kernel_trace->is_contiguous() &&
            kernel_trace->dim() == 3 &&
            kernel_trace->size(0) == num_trace_sites &&
            kernel_trace->size(1) == num_sms &&
            kernel_trace->size(2) == num_trace_values);
    }
    layout::SymBuffer<> backward_sym_buffer{};
    void* backward_workspace_base = nullptr;
    if (!backward_sym_buffer_ptrs.empty()) {
        backward_sym_buffer =
            layout::SymBuffer<>(
                backward_sym_buffer_ptrs, backward_rank);
        backward_workspace_base =
            reinterpret_cast<void*>(
                backward_sym_buffer_ptrs[backward_rank]);
    }
    const auto backward_workspace = layout::Workspace(
        backward_workspace_base, num_ranks,
        num_experts * num_ranks,
        std::max(num_max_tokens_per_rank, 1),
        std::max(num_topk, 1),
        layout::kMinCandidateBlockM);
    const uint32_t num_backward_tokens =
        combined_grad_x_output.has_value()
        ? static_cast<uint32_t>(combined_grad_x_output->size(0))
        : 0u;
    const auto backward_ranges = make_k3_backward_range_set(
        backward_range_sizes,
        num_backward_tokens,
        std::max(
            static_cast<uint32_t>(
                std::max(num_max_tokens_per_rank, 0)),
            num_backward_tokens),
        static_cast<uint32_t>(num_pool_rows),
        static_cast<uint32_t>(num_acts_rows),
        static_cast<uint32_t>(num_sf_pool_rows),
        static_cast<uint32_t>(num_experts));
    DG_HOST_ASSERT((
        !multi_range_backward ||
        backward_ranges.is_full_arena_compatible<112, 192>()));
    // Exactly two physical ranges are compacted into one logical K axis and
    // consumed by the phase-tagged exact dW2/dW13 engine. This must match the
    // generated-source selector above: the exact AuxSlots carry two D maps,
    // the immutable range set, and the suffix arguments.
    const bool enable_k3_mxfp8_two_range_exact =
        enable_k3_mxfp8_three_term_wgrad && multi_range_backward &&
        !inline_weight_dequant && !phase_ordered_weight_dequant &&
        !residual_mxfp8_dgrad && backward_ranges.num_ranges == 2u &&
        num_topk == 16 && clear_wgrad_padding && !accumulate_wgrad &&
        situ_beta == 4.0f && situ_linear_beta == 25.0f;
    const bool enable_k3_two_segment_bf16_progressive =
        enable_k3_mxfp8_three_term_wgrad && multi_range_backward &&
        !enable_k3_mxfp8_two_range_exact &&
        !inline_weight_dequant && !phase_ordered_weight_dequant &&
        !residual_mxfp8_dgrad && backward_ranges.num_ranges == 2u &&
        !accumulate_wgrad;
    if (enable_k3_mxfp8_exact_epilogue_ring) {
        // The explicit-mode ring borrows fixed-top-k planes two and three;
        // planes zero and one remain the live grad-y and exact-X sources.
        // Prove the whole mapping before launching the selected specialization.
        const auto ring_proof =
            k3_mxfp8_epilogue_panel_ring_host_proof<
                3584u, 6144u, 192u>(
                    static_cast<uint32_t>(
                        num_max_tokens_per_rank),
                    static_cast<uint32_t>(num_topk));
        const auto b_ring_proof =
            k3_mxfp8_epilogue_panel_ring_host_proof<
                3584u, 3584u, 192u,
                kK3MxFp8DW13BScratchPrimaryPlane,
                kK3MxFp8DW13BScratchResidualPlane>(
                    static_cast<uint32_t>(
                        num_max_tokens_per_rank),
                    static_cast<uint32_t>(num_topk));
        DG_HOST_ASSERT(
            ring_proof.capacity_ok && ring_proof.planes_disjoint &&
            ring_proof.metadata_fits && ring_proof.ring_depth != 0u &&
            b_ring_proof.capacity_ok &&
            b_ring_proof.planes_disjoint &&
            b_ring_proof.metadata_fits &&
            b_ring_proof.ring_depth == ring_proof.ring_depth &&
            b_ring_proof.ring_row_base == ring_proof.ring_row_base);
        // The producer and grouped consumer now overlap through generation
        // tickets.  Total compact K may therefore exceed physical ring depth;
        // slot-open waits provide rolling backpressure before each wrap.
        DG_HOST_ASSERT(
            backward_ranges.total_pool_rows != 0u &&
            backward_ranges.total_pool_rows % 192u == 0u);
        DG_HOST_ASSERT(
            backward_grad_y.has_value() && backward_x.has_value() &&
            backward_grad_y->size(0) >= num_max_tokens_per_rank &&
            backward_x->size(0) >= num_max_tokens_per_rank);
        const uint64_t combine_begin = reinterpret_cast<uint64_t>(
            backward_grad_y->data_ptr());
        const uint64_t exact_x_begin = reinterpret_cast<uint64_t>(
            backward_x->data_ptr());
        DG_HOST_ASSERT(
            exact_x_begin == combine_begin + ring_proof.plane_bytes);
        const uint64_t primary_begin =
            combine_begin + ring_proof.primary_plane_offset;
        const uint64_t residual_begin =
            combine_begin + ring_proof.residual_plane_offset;
        const uint64_t b_primary_begin =
            combine_begin + b_ring_proof.primary_plane_offset;
        const uint64_t b_residual_begin =
            combine_begin + b_ring_proof.residual_plane_offset;
        DG_HOST_ASSERT(
            combine_begin + 2u * ring_proof.plane_bytes <=
                primary_begin &&
            primary_begin + ring_proof.ring_bytes <= residual_begin &&
            residual_begin + ring_proof.ring_bytes <= b_primary_begin &&
            b_primary_begin + b_ring_proof.ring_bytes <=
                b_residual_begin &&
            b_residual_begin + b_ring_proof.ring_bytes <=
                combine_begin + ring_proof.combine_bytes);
        DG_HOST_ASSERT(
            !backward_sym_buffer_ptrs.empty() &&
            backward_rank >= 0 && backward_rank < num_ranks);
        const uint64_t local_base = static_cast<uint64_t>(
            backward_sym_buffer_ptrs[backward_rank]);
        DG_HOST_ASSERT(
            combine_begin >= local_base &&
            local_base % 128u == 0u &&
            combine_begin % 128u == 0u &&
            ring_proof.primary_plane_offset % 128u == 0u &&
            ring_proof.residual_plane_offset % 128u == 0u &&
            b_ring_proof.primary_plane_offset % 128u == 0u &&
            b_ring_proof.residual_plane_offset % 128u == 0u);
        const uint64_t combine_offset = combine_begin - local_base;
        for (const int64_t signed_peer_base :
             backward_sym_buffer_ptrs) {
            DG_HOST_ASSERT(signed_peer_base != 0);
            const uint64_t peer_base = static_cast<uint64_t>(
                signed_peer_base);
            const uint64_t mapped_combine =
                k3_mxfp8_epilogue_ring_mapped_address(
                    peer_base, combine_offset, 0u);
            const uint64_t mapped_primary =
                k3_mxfp8_epilogue_ring_mapped_address(
                    peer_base, combine_offset,
                    ring_proof.primary_plane_offset);
            const uint64_t mapped_residual =
                k3_mxfp8_epilogue_ring_mapped_address(
                    peer_base, combine_offset,
                    ring_proof.residual_plane_offset);
            const uint64_t mapped_b_primary =
                k3_mxfp8_epilogue_ring_mapped_address(
                    peer_base, combine_offset,
                    b_ring_proof.primary_plane_offset);
            const uint64_t mapped_b_residual =
                k3_mxfp8_epilogue_ring_mapped_address(
                    peer_base, combine_offset,
                    b_ring_proof.residual_plane_offset);
            // Same combine offset and plane offsets on every peer are the
            // symmetric mapping proof.  Pin every TMA base's 128-byte ABI on
            // every peer as well; ordered endpoints also reject wrap.
            DG_HOST_ASSERT(
                mapped_combine == peer_base + combine_offset &&
                peer_base % 128u == 0u &&
                mapped_combine % 128u == 0u &&
                mapped_primary % 128u == 0u &&
                mapped_residual % 128u == 0u &&
                mapped_b_primary % 128u == 0u &&
                mapped_b_residual % 128u == 0u &&
                mapped_primary >= mapped_combine &&
                mapped_primary + ring_proof.ring_bytes <=
                    mapped_residual &&
                mapped_residual + ring_proof.ring_bytes <=
                    mapped_b_primary &&
                mapped_b_primary + b_ring_proof.ring_bytes <=
                    mapped_b_residual &&
                mapped_b_residual + b_ring_proof.ring_bytes >=
                    mapped_b_residual);
        }
    }
    DG_HOST_ASSERT(
        !enable_k3_two_segment_bf16_progressive ||
        !enable_k3_mxfp8_dw13_hybrid);

    using WgradAuxSlot = K3MxFp8WgradAuxSlot<
        CUtensorMap, cutlass::bfloat16_t, cutlass::float_e4m3_t>;
    static_assert(k3_mxfp8_wgrad_aux_slot_abi<
        CUtensorMap, cutlass::bfloat16_t, cutlass::float_e4m3_t>());
    const auto make_legacy_wgrad_slot = [](const CUtensorMap& map) {
        WgradAuxSlot slot{};
        slot.legacy_map = map;
        return slot;
    };
    auto tensor_map_w2_wgrad_slot_a =
        make_legacy_wgrad_slot(tensor_map_w2_wgrad_a);
    auto tensor_map_w2_wgrad_slot_b =
        make_legacy_wgrad_slot(tensor_map_w2_wgrad_b);
    auto tensor_map_w2_wgrad_slot_d =
        make_legacy_wgrad_slot(tensor_map_w2_wgrad_d_bf16);
    auto tensor_map_w13_wgrad_slot_a =
        make_legacy_wgrad_slot(tensor_map_w13_wgrad_a);
    auto tensor_map_w13_wgrad_slot_b =
        make_legacy_wgrad_slot(tensor_map_w13_wgrad_b);
    auto tensor_map_w13_wgrad_slot_d =
        make_legacy_wgrad_slot(tensor_map_w13_wgrad_d_bf16);

    if (enable_k3_mxfp8_dw13_hybrid) {
        // Preserve slots zero through two as BF16 dW2 A/B/D maps.  The
        // hybrid does not execute BF16 dW13, so its legacy D slot can safely
        // carry the exact dW13 output descriptor.
        tensor_map_w13_wgrad_slot_d.exact_output_map =
            tensor_map_w13_wgrad_d;
    } else if (enable_k3_mxfp8_three_term_wgrad &&
               (!multi_range_backward ||
                enable_k3_mxfp8_two_range_exact)) {
        // The one-range and exact two-range unified suffixes need two D maps
        // plus their immutable range set in slots zero through two. A
        // multi-range launch that misses this exact selector retains the
        // separately validated BF16/hybrid slot contract.
        tensor_map_w2_wgrad_slot_a.exact_output_map =
            tensor_map_w2_wgrad_d;
        tensor_map_w2_wgrad_slot_b.exact_output_map =
            tensor_map_w13_wgrad_d;
        tensor_map_w2_wgrad_slot_d.exact_ranges = backward_ranges;
        // Slot three is exactly one TensorMap wide: thirteen pointers followed
        // by six uint32 scalars.
        const uint32_t first_range_tokens =
            backward_ranges.ranges[0].num_tokens;
        const uint32_t second_range_begin =
            backward_ranges.num_ranges > 1u
            ? backward_ranges.token_begin(1u)
            : first_range_tokens;
        tensor_map_w13_wgrad_slot_a.exact_args = {
            .expert_counts = expert_counts.data_ptr<int>(),
            .grad_ye_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_ye.data_ptr<at::BFloat16>()),
            .h_weighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    h_weighted_output.data_ptr<at::BFloat16>()),
            .grad_gate_up_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_gate_up_output.data_ptr<at::BFloat16>()),
            .x_pool_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    x_pool_output.data_ptr<at::BFloat16>()),
            .grad_y_unweighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_y_unweighted_output
                        .value()
                        .data_ptr<at::BFloat16>()),
            .down_unweighted_output =
                reinterpret_cast<const cutlass::bfloat16_t*>(
                    down_unweighted_output
                        .value()
                        .data_ptr<at::BFloat16>()),
            .scale_arena_source =
                reinterpret_cast<const cutlass::float_e4m3_t*>(
                    acts.data_ptr()),
            .state = reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
            .w2_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    w2_dequant_scratch.data_ptr<at::BFloat16>()),
            .w13_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    w13_dequant_scratch.data_ptr<at::BFloat16>()),
            .backward_grad_x_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    combined_grad_x_output
                        .value()
                        .data_ptr<at::BFloat16>()),
            .backward_grad_y =
                reinterpret_cast<const cutlass::bfloat16_t*>(
                    backward_grad_y
                        .value()
                        .data_ptr<at::BFloat16>()),
            .k_capacity = static_cast<uint32_t>(num_acts_rows),
            .num_backward_tokens = num_backward_tokens,
            .first_range_tokens = first_range_tokens,
            .second_range_begin = second_range_begin,
            .num_topk = static_cast<uint32_t>(num_topk),
            .clear_empty_outputs =
                clear_empty_wgrad_expert_outputs ? 1u : 0u,
        };
    }
    const SM100FP8FP4MegaMoEBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = num_acts_rows,
        .num_sf_pool_rows = num_sf_pool_rows,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .inline_weight_dequant = inline_weight_dequant,
        .phase_ordered_weight_dequant =
            phase_ordered_weight_dequant,
        .branch_major_bf16_wgrad_tail =
            enable_k3_branch_major_bf16_wgrad_tail,
        .mxfp8_three_term_wgrad =
            enable_k3_mxfp8_three_term_wgrad,
        .inline_residual_mxfp8_dgrad =
            inline_residual_mxfp8_dgrad,
        .residual_mxfp8_dgrad = residual_mxfp8_dgrad,
        .build_residual_mxfp8_weights =
            build_residual_mxfp8_weights,
        .exact_source_x = backward_x.has_value(),
        .gate_up_prepared = gate_up_prepared,
        .activation = activation,
        .situ_beta = situ_beta,
        .situ_linear_beta = situ_linear_beta,
        .route_weight_mode = route_weight_mode,
        .combine_order_mode = "fixed_topk",
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y = num_ranks > 1
            ? reinterpret_cast<const cutlass::bfloat16_t*>(
                  backward_grad_y->data_ptr<at::BFloat16>())
            : nullptr,
        .backward_x = backward_x.has_value()
            ? reinterpret_cast<const cutlass::bfloat16_t*>(
                  backward_x->data_ptr<at::BFloat16>())
            : nullptr,
        .backward_topk_weights = num_ranks > 1
            ? backward_topk_weights->data_ptr<float>()
            : nullptr,
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .backward_grad_x_output =
            combined_grad_x_output.has_value()
            ? reinterpret_cast<cutlass::bfloat16_t*>(
                  combined_grad_x_output->data_ptr<at::BFloat16>())
            : nullptr,
        .num_backward_tokens = num_backward_tokens,
        .backward_ranges = backward_ranges,
        .token_src_metadata = !backward_sym_buffer_ptrs.empty()
            ? reinterpret_cast<const layout::TokenSrcMetadata*>(
                  token_src_metadata->data_ptr<int>())
            : nullptr,
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride =
            static_cast<uint32_t>(acts_sf.stride(1)),
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts_sf,
        .tensor_map_weights = tensor_map_l1_weights,
        .tensor_map_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2_dequant,
        .tensor_map_w2_dgrad_weights =
            tensor_map_w2_dgrad_weights,
        .tensor_map_w2_dgrad_weights_sf =
            tensor_map_w2_dgrad_weights_sf,
        .tensor_map_w2_weights = tensor_map_w2_weights,
        .tensor_map_w2_scales = tensor_map_w2_scales,
        .tensor_map_w13_dequant = tensor_map_w13_dequant,
        .tensor_map_w13_dgrad_weights =
            tensor_map_w13_dgrad_weights,
        .tensor_map_w13_dgrad_weights_sf =
            tensor_map_w13_dgrad_weights_sf,
        .tensor_map_w13_weights = tensor_map_w13_weights,
        .tensor_map_w13_scales = tensor_map_w13_scales,
        .tensor_map_grad_gate_up = tensor_map_grad_gate_up,
        .tensor_map_w2_wgrad_slot_a = tensor_map_w2_wgrad_slot_a,
        .tensor_map_w2_wgrad_slot_b = tensor_map_w2_wgrad_slot_b,
        .tensor_map_w2_wgrad_slot_d = tensor_map_w2_wgrad_slot_d,
        .tensor_map_w13_wgrad_slot_a = tensor_map_w13_wgrad_slot_a,
        .tensor_map_w13_wgrad_slot_b = tensor_map_w13_wgrad_slot_b,
        .tensor_map_w13_wgrad_slot_d = tensor_map_w13_wgrad_slot_d,
        .k3_mxfp8_wgrad_tensor_maps =
            k3_mxfp8_wgrad_tensor_maps,
        .acts_ptr =
            reinterpret_cast<const cutlass::float_e4m3_t*>(
                acts.data_ptr()),
        .acts_sf_ptr =
            reinterpret_cast<const uint32_t*>(
                acts_sf.data_ptr<int>()),
        .w2_weights =
            reinterpret_cast<const int8_t*>(
                w2_weights.data_ptr()),
        .w2_scales = w2_scales.data_ptr<float>(),
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_dequant_scratch.data_ptr<at::BFloat16>()),
        .w13_weights =
            reinterpret_cast<const int8_t*>(
                w13_weights.data_ptr()),
        .w13_scales = w13_scales.data_ptr<float>(),
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_dequant_scratch.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .value_or(grad_ye)
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            route_weights.scalar_type() == torch::kBFloat16
            ? reinterpret_cast<cutlass::bfloat16_t*>(
                  route_weights.data_ptr<at::BFloat16>())
            : nullptr,
        .route_weights_fp32 =
            route_weights.scalar_type() == torch::kFloat
            ? route_weights.data_ptr<float>()
            : nullptr,
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            down_unweighted_output.has_value()
            ? reinterpret_cast<
                  const cutlass::bfloat16_t*>(
                  down_unweighted_output
                      ->data_ptr<at::BFloat16>())
            : nullptr,
        .grad_route_output =
            grad_route_output.has_value()
            ? grad_route_output->data_ptr<float>()
            : nullptr,
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .activation_limit = activation_limit,
        .compute_w13_dgrad = compute_w13_dgrad,
        .direct_remote_grad_x = direct_remote_grad_x,
        .write_grad_x_pool = write_grad_x_pool,
        .clear_wgrad_padding = clear_wgrad_padding,
        .clear_empty_wgrad_expert_outputs =
            clear_empty_wgrad_expert_outputs,
        .inline_wgrad = inline_wgrad,
        .accumulate_wgrad = accumulate_wgrad,
        .compute_route_grad =
            grad_route_output.has_value(),
        .trace_kernel = kernel_trace.has_value(),
        .vectorized_grad_x_store = multi_range_backward,
        .multi_range_backward = multi_range_backward,
        .kernel_trace =
            kernel_trace.has_value()
            ? reinterpret_cast<uint64_t*>(
                  kernel_trace->data_ptr<int64_t>())
            : nullptr,
        // Backward consumes predecessor outputs and allocates TMEM at entry;
        // keep PDL disabled until the kernel performs dependency sync first.
        .launch_args = LaunchArgs(
            num_sms, 1024, smem_size, 2, false),
    };
    const auto code =
        SM100FP8FP4MegaMoEBackwardWaveRuntime::generate(args);
    const auto kernel_name = fmt::format(
        "sm100_fp8_fp4_mega_moe_backward_dgrad_{}_{}_r{}_inline{}_phase{}_residual{}_build{}_x{}_gate{}_wgrad{}_accum{}_d4{}_mxfp8wgrad{}_exact2{}_ring{}_trace{}_ranges{}",
        activation, route_weight_mode,
        grad_route_output.has_value(), inline_weight_dequant,
        phase_ordered_weight_dequant, residual_mxfp8_dgrad,
        build_residual_mxfp8_weights, backward_x.has_value(),
        gate_up_prepared, inline_wgrad, accumulate_wgrad,
        enable_k3_branch_major_bf16_wgrad_tail ? "t" : "f",
        enable_k3_mxfp8_three_term_wgrad,
        enable_k3_mxfp8_two_range_exact,
        enable_k3_mxfp8_exact_epilogue_ring,
        kernel_trace.has_value(), args.multi_range_backward);
    // Compiler::build materializes `kernel.<name>.<32-hex-digest>`.  Keep the
    // complete directory component below POSIX NAME_MAX on every filesystem.
    constexpr size_t kJitCacheComponentOverhead = 7u + 1u + 32u;
    constexpr size_t kFilesystemNameMax = 255u;
    DG_HOST_ASSERT(
        kernel_name.size() + kJitCacheComponentOverhead <=
        kFilesystemNameMax);
    const auto runtime = compiler->build(kernel_name, code);
    SM100FP8FP4MegaMoEBackwardWaveRuntime::launch(runtime, args);
}

static void sm100_mega_moe_backward_combine_grad_x(
    const torch::Tensor& grad_x_output,
    const torch::Tensor& combine_buffer,
    const std::optional<torch::Tensor>& topk_ids,
    const int& num_max_tokens,
    const int& num_topk,
    const int& num_ranks,
    const int& num_local_experts,
    const std::string& combine_order_mode) {
    const auto [num_tokens, hidden] =
        get_shape<2>(grad_x_output);
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(num_local_experts >= 1);
    DG_HOST_ASSERT(num_max_tokens >= num_tokens);
    DG_HOST_ASSERT(num_topk >= 1 && num_topk <= 32);
    DG_HOST_ASSERT(hidden % 16 == 0);
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(
        grad_x_output.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grad_x_output.is_contiguous());
    DG_HOST_ASSERT(
        combine_buffer.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(combine_buffer.is_contiguous());
    DG_HOST_ASSERT(combine_buffer.dim() == 2);
    DG_HOST_ASSERT(
        combine_buffer.size(0) >=
        static_cast<int64_t>(num_topk) * num_max_tokens);
    DG_HOST_ASSERT(combine_buffer.size(1) == hidden);
    if (combine_order_mode != "fixed_topk") {
        DG_HOST_ASSERT(topk_ids.has_value());
        DG_HOST_ASSERT(topk_ids->is_cuda());
        DG_HOST_ASSERT(
            topk_ids->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(topk_ids->is_contiguous());
        DG_HOST_ASSERT(topk_ids->dim() == 2);
        DG_HOST_ASSERT(topk_ids->size(0) >= num_tokens);
        DG_HOST_ASSERT(topk_ids->size(1) == num_topk);
    }

    const int num_sms = device_runtime->get_num_sms();
    const SM100MegaMoEBackwardCombineRuntime::Args args = {
        .num_ranks = num_ranks,
        .num_local_experts = num_local_experts,
        .combine_order_mode = combine_order_mode,
        .grad_x_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_output.data_ptr<at::BFloat16>()),
        .combine_buffer =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                combine_buffer.data_ptr<at::BFloat16>()),
        .topk_ids =
            topk_ids.has_value()
            ? topk_ids->data_ptr<int64_t>()
            : nullptr,
        .num_tokens = static_cast<uint32_t>(num_tokens),
        .num_max_tokens =
            static_cast<uint32_t>(num_max_tokens),
        .num_topk = static_cast<uint32_t>(num_topk),
        .hidden = static_cast<uint32_t>(hidden),
        .launch_args = LaunchArgs(num_sms, 256),
    };
    const auto code =
        SM100MegaMoEBackwardCombineRuntime::generate(args);
    const auto runtime = compiler->build(fmt::format(
        "sm100_mega_moe_backward_combine_grad_x_r{}_e{}_{}",
        num_ranks, num_local_experts, combine_order_mode),
        code);
    SM100MegaMoEBackwardCombineRuntime::launch(runtime, args);
}

static void sm100_bf16_mega_moe_backward_post_down_prelude(
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& grad_y_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& route_weights_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& down_unweighted_output,
    const torch::Tensor& expert_counts,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_topk,
    const int& block_m,
    const std::string& combine_order_mode,
    const bool& do_reverse_dispatch,
    const bool& compute_route_dot,
    const bool& write_weighted,
    const bool& synchronize_ranks,
    const bool& synchronize_after_dispatch,
    const bool& barrier_only,
    const bool& x_prepared,
    const int& route_prelude_threads) {
    const auto [num_pool_rows, hidden] =
        get_shape<2>(grad_y_unweighted_output);
    const int num_experts =
        static_cast<int>(expert_counts.numel());
    const int num_ranks =
        static_cast<int>(backward_sym_buffer_ptrs.size());
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(
        combine_order_mode == "fixed_topk" ||
        combine_order_mode == "deepep" ||
        combine_order_mode == "deepep_v1");
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        backward_rank >= 0 && backward_rank < num_ranks);
    DG_HOST_ASSERT(hidden % 256 == 0);
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(num_topk > 0);
    DG_HOST_ASSERT(
        route_prelude_threads == 128 ||
        route_prelude_threads == 256);
    if (route_prelude_threads == 128) {
        DG_HOST_ASSERT(hidden == 2048);
        DG_HOST_ASSERT(compute_route_dot);
        DG_HOST_ASSERT(combine_order_mode != "fixed_topk");
    }
    const auto check_bf16_pool = [=](
        const torch::Tensor& tensor) {
        DG_HOST_ASSERT(
            tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
        DG_HOST_ASSERT(
            tensor.sizes() ==
            torch::IntArrayRef({num_pool_rows, hidden}));
    };
    check_bf16_pool(grad_y_unweighted_output);
    check_bf16_pool(x_pool_output);
    if (write_weighted) {
        check_bf16_pool(grad_y_weighted_output);
    }
    if (compute_route_dot) {
        check_bf16_pool(down_unweighted_output);
    }
    DG_HOST_ASSERT(
        backward_grad_y.scalar_type() ==
            torch::kBFloat16 &&
        backward_grad_y.is_contiguous() &&
        backward_grad_y.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_x.scalar_type() == torch::kBFloat16 &&
        backward_x.is_contiguous() &&
        backward_x.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_topk_weights.scalar_type() ==
            torch::kFloat &&
        backward_topk_weights.is_contiguous() &&
        backward_topk_weights.size(1) == num_topk);
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->scalar_type() == torch::kFloat &&
            backward_grad_route->is_contiguous() &&
            backward_grad_route->size(0) >=
                backward_grad_y.size(0) &&
            backward_grad_route->size(1) == num_topk);
    }
    DG_HOST_ASSERT(
        expert_counts.scalar_type() == torch::kInt &&
        expert_counts.is_contiguous());
    DG_HOST_ASSERT(
        token_src_metadata.scalar_type() == torch::kInt &&
        token_src_metadata.is_contiguous() &&
        token_src_metadata.size(0) >= num_pool_rows &&
        token_src_metadata.size(1) == 3);
    DG_HOST_ASSERT(
        route_weights_output.scalar_type() ==
            torch::kFloat &&
        route_weights_output.is_contiguous() &&
        route_weights_output.numel() == num_pool_rows);
    if (compute_route_dot) {
        DG_HOST_ASSERT(
            grad_route_output.scalar_type() ==
                torch::kFloat &&
            grad_route_output.is_contiguous() &&
            grad_route_output.numel() == num_pool_rows);
    }

    const auto exact_alias = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        return lhs.data_ptr() == rhs.data_ptr() &&
               lhs.numel() == rhs.numel();
    };
    const auto overlaps = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        const auto lhs_begin =
            reinterpret_cast<uintptr_t>(lhs.data_ptr());
        const auto rhs_begin =
            reinterpret_cast<uintptr_t>(rhs.data_ptr());
        const auto lhs_end =
            lhs_begin + lhs.numel() * lhs.element_size();
        const auto rhs_end =
            rhs_begin + rhs.numel() * rhs.element_size();
        return std::max(lhs_begin, rhs_begin) <
               std::min(lhs_end, rhs_end);
    };
    DG_HOST_ASSERT(!overlaps(
        x_pool_output, grad_y_unweighted_output));
    if (compute_route_dot || write_weighted) {
        // The weighted destination may replace the retained down pool only
        // because every row is read and overwritten by the same route group.
        if (compute_route_dot && write_weighted) {
            DG_HOST_ASSERT(
                exact_alias(
                    grad_y_weighted_output,
                    down_unweighted_output) ||
                !overlaps(
                    grad_y_weighted_output,
                    down_unweighted_output));
        }
        if (compute_route_dot) {
            DG_HOST_ASSERT(!overlaps(
                grad_y_unweighted_output,
                down_unweighted_output));
            DG_HOST_ASSERT(!overlaps(
                x_pool_output, down_unweighted_output));
        }
        if (write_weighted) {
            DG_HOST_ASSERT(!overlaps(
                grad_y_unweighted_output,
                grad_y_weighted_output));
            DG_HOST_ASSERT(!overlaps(
                x_pool_output, grad_y_weighted_output));
        }
    }

    const int num_sms = device_runtime->get_num_sms();
    const auto backward_sym_buffer = layout::SymBuffer<>(
        backward_sym_buffer_ptrs, backward_rank);
    const auto backward_workspace = layout::Workspace(
        reinterpret_cast<void*>(
            backward_sym_buffer_ptrs[backward_rank]),
        num_ranks,
        num_experts * num_ranks,
        static_cast<uint32_t>(backward_grad_y.size(0)),
        static_cast<uint32_t>(num_topk),
        0);
    SM100BF16MegaMoEBackwardPostDownPreludeRuntime::Args
        args = {
            .hidden = hidden,
            .num_experts = num_experts,
            .block_m = block_m,
            .num_sms = num_sms,
            .num_ranks = num_ranks,
            .combine_order_mode = combine_order_mode,
            .do_reverse_dispatch = do_reverse_dispatch,
            .compute_route_dot = compute_route_dot,
            .write_weighted = write_weighted,
            .synchronize_ranks = synchronize_ranks,
            .synchronize_after_dispatch =
                synchronize_after_dispatch,
            .barrier_only = barrier_only,
            .x_prepared = x_prepared,
            .route_prelude_threads =
                route_prelude_threads,
            .expert_counts = expert_counts.data_ptr<int>(),
            .backward_workspace = backward_workspace,
            .backward_sym_buffer = backward_sym_buffer,
            .backward_grad_y =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    backward_grad_y
                        .data_ptr<at::BFloat16>()),
            .backward_x =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    backward_x.data_ptr<at::BFloat16>()),
            .backward_topk_weights =
                backward_topk_weights.data_ptr<float>(),
            .backward_grad_route =
                backward_grad_route.has_value()
                ? backward_grad_route->data_ptr<float>()
                : nullptr,
            .token_src_metadata =
                reinterpret_cast<
                    const layout::TokenSrcMetadata*>(
                    token_src_metadata.data_ptr<int>()),
            .num_topk =
                static_cast<uint32_t>(num_topk),
            .num_pool_rows =
                static_cast<uint32_t>(num_pool_rows),
            .grad_y_unweighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_y_unweighted_output
                        .data_ptr<at::BFloat16>()),
            .grad_y_weighted_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    grad_y_weighted_output
                        .data_ptr<at::BFloat16>()),
            .x_pool_output =
                reinterpret_cast<cutlass::bfloat16_t*>(
                    x_pool_output
                        .data_ptr<at::BFloat16>()),
            .route_weights_output =
                route_weights_output.data_ptr<float>(),
            .down_unweighted =
                reinterpret_cast<
                    const cutlass::bfloat16_t*>(
                    down_unweighted_output
                        .data_ptr<at::BFloat16>()),
            .grad_route_output =
                grad_route_output.data_ptr<float>(),
            .launch_args =
                LaunchArgs(num_sms, 1024, 4096),
        };
    const auto code =
        SM100BF16MegaMoEBackwardPostDownPreludeRuntime::
            generate(args);
    const auto runtime = compiler->build(fmt::format(
        "sm100_bf16_mega_moe_backward_prelude_r{}_d{}_w{}_s{}_c{}_b{}_x{}_t{}",
        do_reverse_dispatch, compute_route_dot,
        write_weighted, synchronize_ranks,
        synchronize_after_dispatch, barrier_only,
        x_prepared, route_prelude_threads), code);
    SM100BF16MegaMoEBackwardPostDownPreludeRuntime::launch(
        runtime, args);
}

static void sm100_bf16_mega_moe_backward_dgrad(
    const torch::Tensor& gate_up_output,
    const torch::Tensor& grad_h_output,
    const torch::Tensor& grad_gate_up_output,
    const torch::Tensor& h_act_output,
    const torch::Tensor& h_weighted_output,
    const torch::Tensor& x_pool_output,
    const torch::Tensor& grad_x_pool_output,
    const torch::Tensor& grad_route_output,
    const torch::Tensor& grad_ye,
    const torch::Tensor& grad_y_unweighted_output,
    const torch::Tensor& route_weights,
    const torch::Tensor& w2_weights,
    const torch::Tensor& w13_weights,
    const torch::Tensor& expert_counts,
    const torch::Tensor& grid_sync_counter,
    const float& activation_limit,
    const std::string& activation,
    const bool& fast_math,
    const std::string& route_weight_mode,
    const std::string& combine_order_mode,
    const torch::Tensor& down_unweighted_output,
    const int& block_m,
    const bool& direct_remote_grad_x,
    const bool& write_grad_x_pool,
    const bool& clear_wgrad_padding,
    const torch::Tensor& backward_grad_y,
    const torch::Tensor& backward_x,
    const torch::Tensor& backward_topk_weights,
    const std::optional<torch::Tensor>& backward_grad_route,
    const torch::Tensor& token_src_metadata,
    const std::vector<int64_t>& backward_sym_buffer_ptrs,
    const int& backward_rank,
    const int& num_max_tokens_per_rank,
    const int& num_topk,
    const std::string& memory_mode,
    const std::optional<torch::Tensor>& kernel_trace =
        std::nullopt) {
    constexpr int block_n = 128;
    constexpr int block_k = 128;
    constexpr int dgrad_block_k = 64;
    constexpr int store_block_m = 16;
    constexpr int smem_capacity = 232448;
    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;

    const auto [num_experts, intermediate_hidden_2, hidden] =
        get_shape<3>(w13_weights);
    const auto [num_experts_w2, hidden_w2, intermediate_hidden] =
        get_shape<3>(w2_weights);
    const int num_pool_rows = static_cast<int>(grad_ye.size(0));
    const int num_ranks =
        static_cast<int>(backward_sym_buffer_ptrs.size());
    const int sf_block_m = align(block_m, 128);
    const int sf_block_n = block_n;
    const int load_block_m = block_m / 2;
    const int load_block_n = block_n;
    const int num_dispatch_warps = num_ranks > 1 ? 4 : 0;

    DG_HOST_ASSERT(device_runtime->get_arch_major() == 10);
    DG_HOST_ASSERT(activation == "swiglu" || activation == "geglu");
    DG_HOST_ASSERT(
        route_weight_mode == "pre_down" ||
        route_weight_mode == "post_down");
    DG_HOST_ASSERT(
        memory_mode == "legacy" ||
        memory_mode == "phase_ordered");
    DG_HOST_ASSERT(num_ranks >= 1);
    DG_HOST_ASSERT(
        backward_rank >= 0 && backward_rank < num_ranks);
    DG_HOST_ASSERT(num_experts == num_experts_w2);
    DG_HOST_ASSERT(hidden == hidden_w2);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(block_m % 16 == 0);
    DG_HOST_ASSERT(hidden % 256 == 0);
    DG_HOST_ASSERT(intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(num_pool_rows > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_topk > 0);

    const auto check_bf16_contiguous = [](const torch::Tensor& tensor) {
        DG_HOST_ASSERT(tensor.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(tensor.is_contiguous());
    };
    check_bf16_contiguous(gate_up_output);
    check_bf16_contiguous(grad_h_output);
    check_bf16_contiguous(grad_gate_up_output);
    check_bf16_contiguous(h_act_output);
    check_bf16_contiguous(h_weighted_output);
    check_bf16_contiguous(x_pool_output);
    check_bf16_contiguous(grad_x_pool_output);
    check_bf16_contiguous(grad_ye);
    check_bf16_contiguous(grad_y_unweighted_output);
    check_bf16_contiguous(down_unweighted_output);
    check_bf16_contiguous(w2_weights);
    check_bf16_contiguous(w13_weights);
    check_bf16_contiguous(backward_grad_y);
    check_bf16_contiguous(backward_x);
    DG_HOST_ASSERT(backward_topk_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(backward_topk_weights.is_contiguous());
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(backward_grad_route->is_contiguous());
    }
    DG_HOST_ASSERT(route_weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(route_weights.is_contiguous());
    DG_HOST_ASSERT(grad_route_output.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(grad_route_output.is_contiguous());
    DG_HOST_ASSERT(expert_counts.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(expert_counts.is_contiguous());
    DG_HOST_ASSERT(expert_counts.numel() == num_experts);
    DG_HOST_ASSERT(grid_sync_counter.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(grid_sync_counter.is_contiguous());
    DG_HOST_ASSERT(token_src_metadata.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(token_src_metadata.is_contiguous());

    DG_HOST_ASSERT(
        gate_up_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden_2}));
    DG_HOST_ASSERT(grad_gate_up_output.sizes() ==
                   gate_up_output.sizes());
    DG_HOST_ASSERT(
        grad_h_output.sizes() ==
        torch::IntArrayRef(
            {num_pool_rows, intermediate_hidden}));
    DG_HOST_ASSERT(h_act_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        h_weighted_output.sizes() == grad_h_output.sizes());
    DG_HOST_ASSERT(
        x_pool_output.sizes() ==
        torch::IntArrayRef({num_pool_rows, hidden}));
    DG_HOST_ASSERT(
        write_grad_x_pool
            ? grad_x_pool_output.sizes() == x_pool_output.sizes()
            : grad_x_pool_output.sizes() ==
                  torch::IntArrayRef({0, hidden}));
    DG_HOST_ASSERT(grad_ye.sizes() == x_pool_output.sizes());
    DG_HOST_ASSERT(
        grad_y_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(
        down_unweighted_output.sizes() ==
        x_pool_output.sizes());
    DG_HOST_ASSERT(route_weights.numel() == num_pool_rows);
    DG_HOST_ASSERT(grad_route_output.numel() == num_pool_rows);
    DG_HOST_ASSERT(
        backward_grad_y.size(0) >= num_max_tokens_per_rank &&
        backward_grad_y.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_x.size(0) >= num_max_tokens_per_rank &&
        backward_x.size(1) == hidden);
    DG_HOST_ASSERT(
        backward_topk_weights.size(0) >=
            num_max_tokens_per_rank &&
        backward_topk_weights.size(1) == num_topk);
    if (backward_grad_route.has_value()) {
        DG_HOST_ASSERT(
            backward_grad_route->size(0) >=
                num_max_tokens_per_rank &&
            backward_grad_route->size(1) == num_topk);
    }
    DG_HOST_ASSERT(
        token_src_metadata.size(0) >= num_pool_rows &&
        token_src_metadata.size(1) == 3);
    DG_HOST_ASSERT(write_grad_x_pool || direct_remote_grad_x);
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(kernel_trace->is_cuda());
        DG_HOST_ASSERT(
            kernel_trace->scalar_type() == torch::kInt64);
        DG_HOST_ASSERT(kernel_trace->is_contiguous());
        DG_HOST_ASSERT(kernel_trace->device() == grad_ye.device());
    }

    const auto exact_alias = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        return lhs.data_ptr() == rhs.data_ptr() &&
               lhs.numel() == rhs.numel();
    };
    const auto overlaps = [](
        const torch::Tensor& lhs,
        const torch::Tensor& rhs) {
        if (lhs.numel() == 0 || rhs.numel() == 0)
            return false;
        const auto lhs_begin = reinterpret_cast<uintptr_t>(
            lhs.data_ptr());
        const auto rhs_begin = reinterpret_cast<uintptr_t>(
            rhs.data_ptr());
        const auto lhs_end =
            lhs_begin + lhs.numel() * lhs.element_size();
        const auto rhs_end =
            rhs_begin + rhs.numel() * rhs.element_size();
        return std::max(lhs_begin, rhs_begin) <
               std::min(lhs_end, rhs_end);
    };
    const std::array<std::reference_wrapper<const torch::Tensor>, 10>
        alias_tensors = {
            gate_up_output, grad_gate_up_output,
            grad_h_output, h_act_output, h_weighted_output,
            x_pool_output, grad_x_pool_output, grad_ye,
            grad_y_unweighted_output, down_unweighted_output};
    const auto allowed_overlap = [&](const std::size_t lhs, const std::size_t rhs) {
        const auto pair_is = [=](
            const std::size_t first, const std::size_t second) {
            return (lhs == first && rhs == second) ||
                   (lhs == second && rhs == first);
        };
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 8))
            return exact_alias(
                grad_ye, grad_y_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(7, 9))
            return exact_alias(
                grad_ye, down_unweighted_output);
        if (route_weight_mode == "pre_down" &&
            pair_is(8, 9))
            return exact_alias(
                grad_y_unweighted_output,
                down_unweighted_output);
        if (route_weight_mode == "post_down" &&
            pair_is(3, 4))
            return exact_alias(
                h_act_output, h_weighted_output);
        if (memory_mode != "phase_ordered")
            return false;
        if (pair_is(0, 1))
            return exact_alias(
                gate_up_output, grad_gate_up_output);
        if (route_weight_mode == "post_down") {
            if (pair_is(7, 9))
                return exact_alias(
                    grad_ye, down_unweighted_output);
            if (pair_is(2, 8))
                return
                    grad_h_output.data_ptr() ==
                        grad_y_unweighted_output.data_ptr() &&
                    grad_h_output.numel() <=
                        grad_y_unweighted_output.numel();
            if (pair_is(2, 3) || pair_is(2, 4))
                return exact_alias(
                    grad_h_output,
                    pair_is(2, 3)
                        ? h_act_output
                        : h_weighted_output);
            if (pair_is(3, 8) || pair_is(4, 8))
                return
                    grad_y_unweighted_output.data_ptr() ==
                    (pair_is(3, 8)
                         ? h_act_output.data_ptr()
                         : h_weighted_output.data_ptr());
        } else if (pair_is(3, 4)) {
            return exact_alias(
                h_act_output, h_weighted_output);
        }
        return false;
    };
    for (std::size_t lhs = 0; lhs < alias_tensors.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1;
             rhs < alias_tensors.size(); ++rhs) {
            DG_HOST_ASSERT(
                !overlaps(
                    alias_tensors[lhs].get(),
                    alias_tensors[rhs].get()) ||
                allowed_overlap(lhs, rhs));
        }
    }
    if (memory_mode == "phase_ordered") {
        DG_HOST_ASSERT(exact_alias(
            gate_up_output, grad_gate_up_output));
        if (route_weight_mode == "post_down") {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, down_unweighted_output));
            DG_HOST_ASSERT(
                grad_h_output.data_ptr() ==
                    grad_y_unweighted_output.data_ptr() &&
                grad_h_output.numel() <=
                    grad_y_unweighted_output.numel());
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_act_output));
            DG_HOST_ASSERT(exact_alias(
                grad_h_output, h_weighted_output));
        } else {
            DG_HOST_ASSERT(exact_alias(
                grad_ye, grad_y_unweighted_output));
            DG_HOST_ASSERT(exact_alias(
                h_act_output, h_weighted_output));
        }
    }

    const int num_w2_states =
        num_experts * (hidden / dgrad_block_k) *
        (intermediate_hidden / block_n);
    const int num_w13_states =
        num_experts *
        (intermediate_hidden_2 / dgrad_block_k) *
        (hidden / block_n);
    DG_HOST_ASSERT(
        grid_sync_counter.numel() >=
        num_w2_states + num_w13_states + 2);

    const int smem_cd =
        store_block_m * block_n *
        static_cast<int>(sizeof(cutlass::bfloat16_t)) *
        num_tma_store_stages;
    const int smem_per_stage =
        load_block_m * block_k +
        load_block_n * block_k +
        sf_block_m * static_cast<int>(sizeof(uint32_t)) +
        sf_block_n * static_cast<int>(sizeof(uint32_t)) +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        num_dispatch_warps * hidden *
            static_cast<int>(sizeof(cutlass::bfloat16_t)) +
        smem_cd +
        2 * num_epilogue_stages *
            static_cast<int>(sizeof(uint64_t)) +
        num_dispatch_warps *
            static_cast<int>(sizeof(uint64_t)) +
        2 * static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages =
        std::min(
            32,
            (smem_capacity - smem_fixed) / smem_per_stage);
    const int smem_size =
        align(
            smem_fixed + num_stages * smem_per_stage,
            1024);
    DG_HOST_ASSERT(num_stages >= 2);

    // Recompute descriptors are compile-time dead in BF16 mode. Keep valid
    // descriptors in every slot so descriptor prefetch remains well-defined.
    const auto tensor_map_acts = make_tma_2d_desc(
        x_pool_output, hidden, num_pool_rows,
        block_k, load_block_m,
        static_cast<int>(x_pool_output.stride(-2)), 128);
    const auto tensor_map_w13 = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        block_k, load_block_n,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_w13_dgrad = make_tma_2d_desc(
        w13_weights, hidden,
        num_experts * intermediate_hidden_2,
        load_block_n, dgrad_block_k,
        static_cast<int>(w13_weights.stride(-2)), 128);
    const auto tensor_map_gate_up = make_tma_2d_desc(
        gate_up_output, intermediate_hidden_2, num_pool_rows,
        block_n, store_block_m,
        static_cast<int>(gate_up_output.stride(-2)), 128);
    const auto tensor_map_grad_ye = make_tma_2d_desc(
        grad_ye, hidden, num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(grad_ye.stride(-2)), 128);
    const auto tensor_map_w2 = make_tma_2d_desc(
        w2_weights, intermediate_hidden,
        num_experts * hidden,
        load_block_n, dgrad_block_k,
        intermediate_hidden, 128);
    const auto tensor_map_grad_gate_up = make_tma_2d_desc(
        grad_gate_up_output, intermediate_hidden_2,
        num_pool_rows,
        dgrad_block_k, load_block_m,
        static_cast<int>(
            grad_gate_up_output.stride(-2)), 128);

    const int num_sms = device_runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms % 2 == 0);
    constexpr int num_trace_sites = 22;
    constexpr int num_trace_values = 5;
    if (kernel_trace.has_value()) {
        DG_HOST_ASSERT(
            kernel_trace->dim() == 3 &&
            kernel_trace->size(0) == num_trace_sites &&
            kernel_trace->size(1) == num_sms &&
            kernel_trace->size(2) == num_trace_values);
    }
    const auto backward_sym_buffer =
        layout::SymBuffer<>(
            backward_sym_buffer_ptrs, backward_rank);
    const auto backward_workspace = layout::Workspace(
        reinterpret_cast<void*>(
            backward_sym_buffer_ptrs[backward_rank]),
        num_ranks, num_experts * num_ranks,
        num_max_tokens_per_rank, num_topk,
        layout::kMinCandidateBlockM);

    const uint32_t num_backward_tokens =
        static_cast<uint32_t>(backward_grad_y.size(0));
    const auto backward_ranges = K3BackwardRangeSet::single_range(
        num_backward_tokens,
        std::max(
            static_cast<uint32_t>(
                std::max(num_max_tokens_per_rank, 0)),
            num_backward_tokens),
        static_cast<uint32_t>(num_pool_rows),
        0u,
        1u);

    const SM100FP8FP4MegaMoEBackwardWaveRuntime::Args args = {
        .hidden = hidden,
        .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_pool_rows = num_pool_rows,
        .num_acts_rows = 0,
        .num_sf_pool_rows = 1,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .sf_block_m = sf_block_m,
        .sf_block_n = sf_block_n,
        .num_stages = num_stages,
        .num_sms = num_sms,
        .num_ranks = num_ranks,
        .bf16_mode = true,
        .activation = activation,
        .fast_math = fast_math,
        .route_weight_mode = route_weight_mode,
        .combine_order_mode = combine_order_mode,
        .expert_counts = expert_counts.data_ptr<int>(),
        .backward_sym_buffer = backward_sym_buffer,
        .backward_workspace = backward_workspace,
        .backward_grad_y =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_grad_y.data_ptr<at::BFloat16>()),
        .backward_x =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                backward_x.data_ptr<at::BFloat16>()),
        .backward_topk_weights =
            backward_topk_weights.data_ptr<float>(),
        .backward_grad_route =
            backward_grad_route.has_value()
            ? backward_grad_route->data_ptr<float>()
            : nullptr,
        .num_backward_tokens = num_backward_tokens,
        .backward_ranges = backward_ranges,
        .token_src_metadata =
            reinterpret_cast<
                const layout::TokenSrcMetadata*>(
                token_src_metadata.data_ptr<int>()),
        .num_topk = static_cast<uint32_t>(num_topk),
        .acts_sf_stride = 0,
        .tensor_map_acts = tensor_map_acts,
        .tensor_map_acts_sf = tensor_map_acts,
        .tensor_map_weights = tensor_map_w13,
        .tensor_map_weights_sf = tensor_map_w13,
        .tensor_map_output = tensor_map_gate_up,
        .tensor_map_grad_ye = tensor_map_grad_ye,
        .tensor_map_w2_dequant = tensor_map_w2,
        .tensor_map_w2_weights = tensor_map_w2,
        .tensor_map_w2_scales = tensor_map_w2,
        .tensor_map_w13_dequant = tensor_map_w13_dgrad,
        .tensor_map_w13_weights = tensor_map_w13,
        .tensor_map_w13_scales = tensor_map_w13,
        .tensor_map_grad_gate_up =
            tensor_map_grad_gate_up,
        .acts_ptr = nullptr,
        .acts_sf_ptr = nullptr,
        .w2_weights = nullptr,
        .w2_scales = nullptr,
        .w2_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w2_weights.data_ptr<at::BFloat16>()),
        .w13_weights = nullptr,
        .w13_scales = nullptr,
        .w13_dequant_scratch =
            reinterpret_cast<cutlass::bfloat16_t*>(
                w13_weights.data_ptr<at::BFloat16>()),
        .gate_up_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                gate_up_output.data_ptr<at::BFloat16>()),
        .grad_ye_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_ye.data_ptr<at::BFloat16>()),
        .grad_y_unweighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_y_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .route_weights =
            nullptr,
        .route_weights_fp32 = route_weights.data_ptr<float>(),
        .grad_h_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_h_output.data_ptr<at::BFloat16>()),
        .grad_gate_up_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_gate_up_output.data_ptr<at::BFloat16>()),
        .h_act_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_act_output.data_ptr<at::BFloat16>()),
        .h_weighted_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                h_weighted_output.data_ptr<at::BFloat16>()),
        .x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                x_pool_output.data_ptr<at::BFloat16>()),
        .grad_x_pool_output =
            reinterpret_cast<cutlass::bfloat16_t*>(
                grad_x_pool_output.data_ptr<at::BFloat16>()),
        .down_unweighted_output =
            reinterpret_cast<const cutlass::bfloat16_t*>(
                down_unweighted_output
                    .data_ptr<at::BFloat16>()),
        .grad_route_output =
            grad_route_output.data_ptr<float>(),
        .grid_sync_counter =
            reinterpret_cast<uint32_t*>(
                grid_sync_counter.data_ptr<int>()),
        .activation_limit = activation_limit,
        .compute_w13_dgrad = true,
        .direct_remote_grad_x = direct_remote_grad_x,
        .write_grad_x_pool = write_grad_x_pool,
        .clear_wgrad_padding = clear_wgrad_padding,
        .clear_empty_wgrad_expert_outputs = true,
        .compute_route_grad = true,
        .trace_kernel = kernel_trace.has_value(),
        .vectorized_grad_x_store = get_env<int>(
            "DG_BF16_MEGA_MOE_VECTORIZED_GRAD_X_STORE",
            1) == 1,
        .wide_grad_x_store = get_env<int>(
            "DG_BF16_MEGA_MOE_WIDE_GRAD_X_STORE",
            0) == 1,
        .kernel_trace =
            kernel_trace.has_value()
            ? reinterpret_cast<uint64_t*>(
                  kernel_trace->data_ptr<int64_t>())
            : nullptr,
        .inputs_prepared =
            memory_mode == "phase_ordered" &&
            route_weight_mode == "post_down",
        .dispatch_inputs_prepared =
            memory_mode == "phase_ordered",
        .launch_args =
            LaunchArgs(num_sms, 1024, smem_size, 2),
    };
    const auto code =
        SM100FP8FP4MegaMoEBackwardWaveRuntime::generate(args);
    const auto runtime = compiler->build(
        fmt::format(
            "sm100_bf16_mega_moe_backward_dgrad_trace{}_vec{}_wide{}_ranges{}",
            kernel_trace.has_value(),
            args.vectorized_grad_x_store,
            args.wide_grad_x_store,
            args.multi_range_backward),
        code);
    SM100FP8FP4MegaMoEBackwardWaveRuntime::launch(
        runtime, args);
}

}  // namespace deep_gemm
