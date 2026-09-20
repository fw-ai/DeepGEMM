#pragma once

#include "../../runtime/runtime.hpp"

namespace deep_gemm {

// Side kernels share DeepJIT's cache, current-stream launch and launch hooks.
// This helper only constructs options; it owns no compiler or kernel cache.
static deep_jit::cuda::LaunchOptions side_lora_launch_options(
    const int num_ctas, const int num_threads,
    const int smem_bytes, const int cluster_size) {
    return {
        .num_smem_bytes = smem_bytes,
        .grid_dim = dim3(num_ctas, 1, 1),
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(cluster_size, 1, 1),
    };
}

}  // namespace deep_gemm
