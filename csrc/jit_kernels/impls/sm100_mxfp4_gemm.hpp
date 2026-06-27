#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

// Standalone packed MXFP4 x MXFP4 (E2M1 data, UE8M0 SF gran-32, 2-CTA) GEMM.
// De-risk vehicle: explicit (compile-time) template params, no `get_best_config`.
class SM100MXFP4GemmRuntime final: public LaunchRuntime<SM100MXFP4GemmRuntime> {
public:
    struct Args {
        int m, n, k;
        int block_m, block_n, block_k;
        int num_stages;
        int num_non_epilogue_threads, num_epilogue_threads;
        int num_sms;
        bool is_nvfp4;
        float ab_global_scale;  // NVFP4 output dequant scale (gs_a * gs_b); 1.0 for MXFP4

        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_mxfp4_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_mxfp4_gemm_impl<
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}
    >);
}};
)",
        args.m, args.n, args.k,
        args.block_m, args.block_n, args.block_k,
        args.num_stages,
        args.num_non_epilogue_threads, args.num_epilogue_threads,
        args.num_sms,
        args.is_nvfp4 ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            static_cast<uint32_t>(args.m), static_cast<uint32_t>(args.n), static_cast<uint32_t>(args.k),
            args.tensor_map_a, args.tensor_map_sfa,
            args.tensor_map_b, args.tensor_map_sfb,
            args.tensor_map_cd,
            args.ab_global_scale));
    }
};

static void sm100_mxfp4_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                             const torch::Tensor& b, const torch::Tensor& sfb,
                             const torch::Tensor& d,
                             const int& m, const int& n, const int& k,
                             const MmaKind& mma_kind = MmaKind::MXFP4,
                             const float& a_global_scale = 1.0f,
                             const float& b_global_scale = 1.0f) {
    // Fixed de-risk configuration
    constexpr int block_m = 128, block_n = 128, block_k = 128;
    constexpr int num_stages = 4;
    constexpr int num_non_epilogue_threads = 128, num_epilogue_threads = 128;
    const bool is_nvfp4 = (mma_kind == MmaKind::NVFP4);
    const int gran_k = get_sf_gran_k(mma_kind);          // 32 (mxfp4) / 16 (nvfp4)
    const int num_sf_k_per_load = block_k / (gran_k * 4); // int32s per token per K-block (1 / 2)

    // Packed FP4 tensors are stored as int8 (`kPackedFP4`), 2 elements per byte
    DG_HOST_ASSERT(a.scalar_type() == kPackedFP4 and b.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and sfb.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(m % block_m == 0 and n % block_n == 0 and k % block_k == 0);
    // 2-CTA (cluster_n = 2) requires an even number of N blocks
    DG_HOST_ASSERT((n / block_n) % 2 == 0 and "MXFP4 de-risk GEMM requires N divisible by 256 (2-CTA)");

    // Even SM count for 2-CTA clusters
    int num_sms = device_runtime->get_num_sms();
    num_sms -= num_sms % 2;

    const int load_block_m = block_m / 2;  // acts split on M across the cluster (cluster_n = 2)
    const int load_block_n = block_n;
    constexpr int store_block_m = 16, store_block_n = 128, swizzle_cd = 128;
    const int swizzle_ab = block_k / 2;    // packed FP4 K-major swizzle in bytes (64)

    // Packed-FP4 A/B TMA descriptors (K-major, `fp4_unpacked_smem = false` -> 16U4_ALIGN8B)
    // 2-CTA `cta_group::2`: per CUTLASS, each CTA loads its OWN per-CTA box (`load_block_m`) at
    // its own M coord (offset by rank); the 2-SM atom routes all tx to the leader's barrier.
    const auto tensor_map_a = make_tma_2d_desc(a, k, m, block_k, load_block_m,
                                               static_cast<int>(a.stride(0)),
                                               swizzle_ab, 0, false, false);
    const auto tensor_map_b = make_tma_2d_desc(b, k, n, block_k, load_block_n,
                                               static_cast<int>(b.stride(0)),
                                               swizzle_ab, 0, false, false);
    // SF descriptors (MN-major, no swizzle): UE8M0 gran-32 (mxfp4) / E4M3 gran-16 (nvfp4).
    // `smem_outer_dim = num_sf_k_per_load` loads all K-uint32s of a K-block (2 for gran-16).
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k, block_m, gran_k, 1, 0, 0, false, num_sf_k_per_load);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k, block_n, gran_k, 1, 0, 0, false, num_sf_k_per_load);
    // BF16 output (N-major)
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n, store_block_m, store_block_n,
                                                static_cast<int>(d.stride(-2)), 1, swizzle_cd);

    // Shared memory size (must mirror the kernel's `SharedStorage` layout)
    const int sf_block_m = align(block_m, 128), sf_block_n = align(block_n, 128);
    constexpr int num_epilogue_stages = 2, num_tma_store_stages = 2;
    const int smem_cd = store_block_m * store_block_n * static_cast<int>(sizeof(int16_t)) * num_tma_store_stages;
    const int smem_a = load_block_m * (block_k / 2);
    const int smem_b = load_block_n * (block_k / 2);
    const int smem_sfa = sf_block_m * num_sf_k_per_load * static_cast<int>(sizeof(int));
    const int smem_sfb = sf_block_n * num_sf_k_per_load * static_cast<int>(sizeof(int));
    const int smem_barriers = (num_stages * 3 + num_epilogue_stages * 2) * 8;
    const int smem_size = smem_cd
                        + num_stages * (smem_a + smem_b)
                        + num_stages * (smem_sfa + smem_sfb)
                        + smem_barriers + 4;
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    const SM100MXFP4GemmRuntime::Args args = {
        .m = m, .n = n, .k = k,
        .block_m = block_m, .block_n = block_n, .block_k = block_k,
        .num_stages = num_stages,
        .num_non_epilogue_threads = num_non_epilogue_threads,
        .num_epilogue_threads = num_epilogue_threads,
        .num_sms = num_sms,
        .is_nvfp4 = is_nvfp4,
        .ab_global_scale = a_global_scale * b_global_scale,
        .tensor_map_a = tensor_map_a,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_cd = tensor_map_cd,
        .launch_args = LaunchArgs(num_sms,
                                  num_non_epilogue_threads + num_epilogue_threads,
                                  smem_size, /*cluster_dim=*/2)
    };
    const auto code = SM100MXFP4GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_mxfp4_gemm", code);
    SM100MXFP4GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
