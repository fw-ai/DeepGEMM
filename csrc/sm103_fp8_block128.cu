#include "apis/sm103_fp8_block128.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_grouped_mm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/packed_stride.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

// CUTLASS 4.2 exposes an SM103 epilogue-builder alias but omits the equivalent
// alias for its software (FP32) blockwise-scale mainloop.  The underlying UMMA
// implementation is ISA-compatible; this bridge lets the public kernel use an
// SM103 ArchTag while retaining CUTLASS's software-scale collective.  The
// translation unit itself contains only sm_103a code and every host entry point
// checks compute capability 10.3 exactly.
namespace cutlass::gemm {
struct KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm103 final
    : KernelSchedule2Sm, KernelScheduleSm100PtrArrayBlockwise {};
}  // namespace cutlass::gemm

namespace cutlass::gemm::collective {
template <
    class ElementA,
    class GmemLayoutA,
    int AlignmentA,
    class ElementB,
    class GmemLayoutB,
    int AlignmentB,
    class ElementAccumulator,
    class TileShape_MNK,
    class ClusterShape_MNK,
    class StageCountType,
    class KernelScheduleType>
struct CollectiveBuilder<
    arch::Sm103,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_base_of_v<KernelScheduleSm100PtrArrayBlockwise,
                                         KernelScheduleType>>>
    : CollectiveBuilder<
          arch::Sm100,
          arch::OpClassTensorOp,
          ElementA,
          GmemLayoutA,
          AlignmentA,
          ElementB,
          GmemLayoutB,
          AlignmentB,
          ElementAccumulator,
          TileShape_MNK,
          ClusterShape_MNK,
          StageCountType,
          KernelScheduleType> {};
}  // namespace cutlass::gemm::collective

namespace deep_gemm::sm103_fp8_block128 {
namespace {

constexpr int kRequiredMajor = 10;
constexpr int kRequiredMinor = 3;
constexpr int kBlockK = 128;
constexpr float kE4M3Max = 448.0f;

constexpr int64_t align_rows(const int64_t rows) {
    return (rows + kBlockK - 1) / kBlockK * kBlockK;
}

#define DG_CHECK_CUDA(tensor) \
    TORCH_CHECK((tensor).is_cuda(), #tensor " must be a CUDA tensor")
#define DG_CHECK_CONTIGUOUS(tensor) \
    TORCH_CHECK((tensor).is_contiguous(), #tensor " must be contiguous")

void check_sm103_device(const torch::Tensor& tensor) {
    DG_CHECK_CUDA(tensor);
    c10::cuda::CUDAGuard guard(tensor.device());
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
    TORCH_CHECK(
        properties.major == kRequiredMajor && properties.minor == kRequiredMinor,
        "FP8-block128 MegaMoE is SM103-only; CUDA runtime reported compute capability ",
        properties.major, ".", properties.minor, " for device ", tensor.get_device(),
        ". No fallback is available."
    );
}

void check_bf16_matrix(const torch::Tensor& tensor, const char* name) {
    check_sm103_device(tensor);
    DG_CHECK_CONTIGUOUS(tensor);
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
    TORCH_CHECK(tensor.size(1) % kBlockK == 0, name, " K dimension must be divisible by 128");
}

void check_fp8_matrix_and_scales(
    const torch::Tensor& tensor,
    const torch::Tensor& scales,
    const char* name
) {
    check_sm103_device(tensor);
    DG_CHECK_CONTIGUOUS(tensor);
    DG_CHECK_CUDA(scales);
    DG_CHECK_CONTIGUOUS(scales);
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat8_e4m3fn, name, " must be float8_e4m3fn");
    TORCH_CHECK(tensor.size(1) % kBlockK == 0, name, " K dimension must be divisible by 128");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, name, " scales must be float32");
    TORCH_CHECK(scales.dim() == 2, name, " scales must be rank 2");
    TORCH_CHECK(scales.size(0) == tensor.size(0), name, " scales row count mismatch");
    TORCH_CHECK(scales.size(1) == tensor.size(1) / kBlockK, name, " scales block count mismatch");
    TORCH_CHECK(scales.device() == tensor.device(), name, " and scales must be on the same device");
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ __forceinline__ float block_max_128(float value, float* warp_values) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 4 ? warp_values[threadIdx.x] : 0.0f;
    if (warp == 0) {
        value = warp_max(value);
    }
    if (threadIdx.x == 0) {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

__global__ void sm103_quantize_bf16_e4m3_group128_kernel(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* scales,
    int64_t rows,
    int64_t columns
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t num_blocks_k = columns / kBlockK;
    const int64_t work_idx = blockIdx.x;
    const int64_t row = work_idx / num_blocks_k;
    const int64_t block_k = work_idx - row * num_blocks_k;
    if (row >= rows) {
        return;
    }

    const int64_t column = block_k * kBlockK + threadIdx.x;
    const int64_t offset = row * columns + column;
    const float value = __bfloat162float(input[offset]);
    __shared__ float warp_values[4];
    const float amax = block_max_128(fabsf(value), warp_values);
    const float scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (threadIdx.x == 0) {
        scales[row * num_blocks_k + block_k] = scale;
    }
    output[offset] = __nv_fp8_e4m3(value / scale);
#endif
}

__global__ void sm103_dequantize_e4m3_group128_kernel(
    const __nv_fp8_e4m3* input,
    const float* scales,
    __nv_bfloat16* output,
    int64_t rows,
    int64_t columns
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = rows * columns;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t row = linear_idx / columns;
    const int64_t column = linear_idx - row * columns;
    const float scale = scales[row * (columns / kBlockK) + column / kBlockK];
    output[linear_idx] = __float2bfloat16_rn(static_cast<float>(input[linear_idx]) * scale);
#endif
}

__global__ void sm103_swiglu_quantize_group128_kernel(
    const __nv_bfloat16* preactivation,
    __nv_fp8_e4m3* output,
    float* scales,
    int64_t rows,
    int64_t hidden
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t num_blocks_k = hidden / kBlockK;
    const int64_t work_idx = blockIdx.x;
    const int64_t row = work_idx / num_blocks_k;
    const int64_t block_k = work_idx - row * num_blocks_k;
    if (row >= rows) {
        return;
    }

    const int64_t column = block_k * kBlockK + threadIdx.x;
    const int64_t pre_row_offset = row * hidden * 2;
    const float up = __bfloat162float(preactivation[pre_row_offset + column]);
    const float gate = __bfloat162float(preactivation[pre_row_offset + hidden + column]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float value = up * gate * sigmoid_gate;
    __shared__ float warp_values[4];
    const float amax = block_max_128(fabsf(value), warp_values);
    const float scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (threadIdx.x == 0) {
        scales[row * num_blocks_k + block_k] = scale;
    }
    output[row * hidden + column] = __nv_fp8_e4m3(value / scale);
#endif
}

__global__ void sm103_swiglu_backward_kernel(
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* preactivation,
    __nv_bfloat16* grad_preactivation,
    int64_t rows,
    int64_t hidden
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = rows * hidden;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t row = linear_idx / hidden;
    const int64_t column = linear_idx - row * hidden;
    const int64_t pre_row_offset = row * hidden * 2;
    const float up = __bfloat162float(preactivation[pre_row_offset + column]);
    const float gate = __bfloat162float(preactivation[pre_row_offset + hidden + column]);
    const float grad = __bfloat162float(grad_output[linear_idx]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float silu_gate = gate * sigmoid_gate;
    const float silu_grad = sigmoid_gate * (1.0f + gate * (1.0f - sigmoid_gate));
    grad_preactivation[pre_row_offset + column] = __float2bfloat16_rn(grad * silu_gate);
    grad_preactivation[pre_row_offset + hidden + column] = __float2bfloat16_rn(grad * up * silu_grad);
#endif
}

// The forward W13 ABI is [up; gate], while FireTitan's canonical gradient
// ownership is the interleaved [gate, up] expert layout.  Emit the activation
// gradient in canonical order so both dgrad and wgrad can consume the resident
// canonical W13 buffers directly without allocating a reordered weight copy.
__global__ void sm103_swiglu_backward_canonical_kernel(
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* preactivation,
    __nv_bfloat16* grad_preactivation,
    int64_t rows,
    int64_t hidden
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = rows * hidden;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t row = linear_idx / hidden;
    const int64_t column = linear_idx - row * hidden;
    const int64_t pre_row_offset = row * hidden * 2;
    const float up = __bfloat162float(preactivation[pre_row_offset + column]);
    const float gate = __bfloat162float(preactivation[pre_row_offset + hidden + column]);
    const float grad = __bfloat162float(grad_output[linear_idx]);
    const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    const float silu_gate = gate * sigmoid_gate;
    const float silu_grad = sigmoid_gate * (1.0f + gate * (1.0f - sigmoid_gate));
    // Canonical W13 order: gate, then up.
    grad_preactivation[pre_row_offset + column] = __float2bfloat16_rn(grad * up * silu_grad);
    grad_preactivation[pre_row_offset + hidden + column] = __float2bfloat16_rn(grad * silu_gate);
#endif
}

__global__ void sm103_route_scale_quantize_group128_kernel(
    const __nv_bfloat16* grad_output,
    const float* route_scores,
    const int64_t* route_order,
    __nv_fp8_e4m3* output,
    float* scales,
    int64_t num_routes,
    int64_t hidden,
    int64_t topk
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t num_blocks_k = hidden / kBlockK;
    const int64_t work_idx = blockIdx.x;
    const int64_t route = work_idx / num_blocks_k;
    const int64_t block_k = work_idx - route * num_blocks_k;
    if (route >= num_routes) {
        return;
    }

    const int64_t source_route = route_order[route];
    const int64_t token = source_route / topk;
    const float score = route_scores[source_route];
    const int64_t column = block_k * kBlockK + threadIdx.x;
    const float value = __bfloat162float(grad_output[token * hidden + column]) * score;
    __shared__ float warp_values[4];
    const float amax = block_max_128(fabsf(value), warp_values);
    const float scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (threadIdx.x == 0) {
        scales[route * num_blocks_k + block_k] = scale;
    }
    output[route * hidden + column] = __nv_fp8_e4m3(value / scale);
#endif
}

__global__ void sm103_post_down_combine_kernel(
    const __nv_bfloat16* route_output,
    const float* route_scores,
    __nv_bfloat16* output,
    int64_t num_tokens,
    int64_t hidden,
    int64_t topk
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = num_tokens * hidden;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t token = linear_idx / hidden;
    const int64_t column = linear_idx - token * hidden;
    float sum = 0.0f;
    #pragma unroll 1
    for (int64_t route = 0; route < topk; ++route) {
        const int64_t route_idx = token * topk + route;
        sum += __bfloat162float(route_output[route_idx * hidden + column]) * route_scores[route_idx];
    }
    output[linear_idx] = __float2bfloat16_rn(sum);
#endif
}

__global__ void sm103_route_sum_kernel(
    const __nv_bfloat16* route_grad,
    __nv_bfloat16* output,
    int64_t num_tokens,
    int64_t hidden,
    int64_t topk
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = num_tokens * hidden;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t token = linear_idx / hidden;
    const int64_t column = linear_idx - token * hidden;
    float sum = 0.0f;
    #pragma unroll 1
    for (int64_t route = 0; route < topk; ++route) {
        sum += __bfloat162float(route_grad[(token * topk + route) * hidden + column]);
    }
    output[linear_idx] = __float2bfloat16_rn(sum);
#endif
}

__global__ void sm103_post_down_score_grad_kernel(
    const __nv_bfloat16* route_output,
    const __nv_bfloat16* grad_output,
    float* grad_scores,
    int64_t num_routes,
    int64_t hidden,
    int64_t topk
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t route = blockIdx.x;
    if (route >= num_routes) {
        return;
    }
    const int64_t token = route / topk;
    float partial = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        partial += __bfloat162float(route_output[route * hidden + column]) *
                   __bfloat162float(grad_output[token * hidden + column]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0) {
            grad_scores[route] = total;
        }
    }
#endif
}

__global__ void sm103_expanded_post_down_scale_kernel(
    const __nv_bfloat16* route_output,
    const float* route_scores,
    __nv_bfloat16* scaled_output,
    int64_t rows,
    int64_t hidden
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numel = rows * hidden;
    if (linear_idx >= numel) {
        return;
    }
    const int64_t row = linear_idx / hidden;
    const float value = __bfloat162float(route_output[linear_idx]) * route_scores[row];
    scaled_output[linear_idx] = __float2bfloat16_rn(value);
#endif
}

template <typename RouteIndex>
__global__ void sm103_expand_compact_routes_kernel(
    const __nv_bfloat16* compact,
    const RouteIndex* routes,
    __nv_bfloat16* expanded,
    int64_t compact_rows,
    int64_t topk,
    int64_t hidden,
    int64_t expanded_rows,
    int64_t route_stride_0,
    int64_t route_stride_1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t route_linear = blockIdx.x;
    if (route_linear >= compact_rows * topk) {
        return;
    }
    const int64_t compact_row = route_linear / topk;
    const int64_t slot = route_linear - compact_row * topk;
    const int64_t expanded_row = static_cast<int64_t>(
        routes[compact_row * route_stride_0 + slot * route_stride_1]);
    if (expanded_row < 0 || expanded_row >= expanded_rows) {
        return;
    }
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        expanded[expanded_row * hidden + column] = compact[compact_row * hidden + column];
    }
#endif
}

template <typename RouteIndex>
__global__ void sm103_collapse_expanded_routes_kernel(
    const __nv_bfloat16* expanded,
    const RouteIndex* routes,
    __nv_bfloat16* compact,
    int64_t compact_rows,
    int64_t topk,
    int64_t hidden,
    int64_t expanded_rows,
    int64_t route_stride_0,
    int64_t route_stride_1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t compact_row = blockIdx.x;
    if (compact_row >= compact_rows) {
        return;
    }
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 1
        for (int64_t slot = 0; slot < topk; ++slot) {
            const int64_t expanded_row = static_cast<int64_t>(
                routes[compact_row * route_stride_0 + slot * route_stride_1]);
            if (expanded_row >= 0 && expanded_row < expanded_rows) {
                sum += __bfloat162float(expanded[expanded_row * hidden + column]);
            }
        }
        compact[compact_row * hidden + column] = __float2bfloat16_rn(sum);
    }
#endif
}

template <typename RouteIndex>
__global__ void sm103_expanded_post_down_score_grad_kernel(
    const __nv_bfloat16* route_output,
    const __nv_bfloat16* grad_output,
    const RouteIndex* routes,
    float* grad_scores,
    int64_t compact_rows,
    int64_t topk,
    int64_t hidden,
    int64_t expanded_rows,
    int64_t route_stride_0,
    int64_t route_stride_1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t route_linear = blockIdx.x;
    if (route_linear >= compact_rows * topk) {
        return;
    }
    const int64_t compact_row = route_linear / topk;
    const int64_t slot = route_linear - compact_row * topk;
    const int64_t expanded_row = static_cast<int64_t>(
        routes[compact_row * route_stride_0 + slot * route_stride_1]);
    if (expanded_row < 0 || expanded_row >= expanded_rows) {
        if (threadIdx.x == 0) {
            grad_scores[route_linear] = 0.0f;
        }
        return;
    }
    float partial = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        partial += __bfloat162float(route_output[expanded_row * hidden + column]) *
                   __bfloat162float(grad_output[expanded_row * hidden + column]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = partial;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0) {
            grad_scores[route_linear] = total;
        }
    }
#endif
}

__global__ void sm103_expanded_route_scale_quantize_group128_kernel(
    const __nv_bfloat16* grad_output,
    const float* route_scores,
    __nv_fp8_e4m3* output,
    float* scales,
    int64_t rows,
    int64_t hidden
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t num_blocks_k = hidden / kBlockK;
    const int64_t work_idx = blockIdx.x;
    const int64_t row = work_idx / num_blocks_k;
    const int64_t block_k = work_idx - row * num_blocks_k;
    if (row >= rows) {
        return;
    }
    const int64_t column = block_k * kBlockK + threadIdx.x;
    const float value = __bfloat162float(grad_output[row * hidden + column]) * route_scores[row];
    __shared__ float warp_values[4];
    const float amax = block_max_128(fabsf(value), warp_values);
    const float scale = amax == 0.0f ? 1.0f : amax / kE4M3Max;
    if (threadIdx.x == 0) {
        scales[row * num_blocks_k + block_k] = scale;
    }
    output[row * hidden + column] = __nv_fp8_e4m3(value / scale);
#endif
}

__global__ void sm103_zero_expanded_wgrad_padding_kernel(
    __nv_bfloat16* left,
    __nv_bfloat16* right,
    const int32_t* group_counts,
    const int32_t* padded_offsets,
    int64_t groups,
    int64_t left_columns,
    int64_t right_columns
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t expert = blockIdx.x;
    if (expert >= groups) {
        return;
    }
    const int64_t start = expert == 0 ? 0 : padded_offsets[expert - 1];
    const int64_t valid_end = start + group_counts[expert];
    const int64_t padded_end = padded_offsets[expert];
    const int64_t padding_rows = padded_end - valid_end;
    const int64_t left_elements = padding_rows * left_columns;
    const int64_t total_elements = left_elements + padding_rows * right_columns;
    for (int64_t index = threadIdx.x; index < total_elements; index += blockDim.x) {
        if (index < left_elements) {
            const int64_t row = valid_end + index / left_columns;
            const int64_t column = index - (row - valid_end) * left_columns;
            left[row * left_columns + column] = __float2bfloat16(0.0f);
        } else {
            const int64_t right_index = index - left_elements;
            const int64_t row = valid_end + right_index / right_columns;
            const int64_t column = right_index - (row - valid_end) * right_columns;
            right[row * right_columns + column] = __float2bfloat16(0.0f);
        }
    }
#endif
}

__global__ void sm103_prepare_expanded_wgrad_metadata_kernel(
    const int32_t* psum,
    int32_t* padded_offsets,
    int32_t* group_counts,
    int64_t groups,
    int64_t storage_rows
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t expert = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (expert >= groups) {
        return;
    }
    const int64_t previous_end = expert == 0 ? 0 : psum[expert - 1];
    const int64_t start = (previous_end + kBlockK - 1) / kBlockK * kBlockK;
    const int64_t end = psum[expert];
    const int64_t padded_end = (end + kBlockK - 1) / kBlockK * kBlockK;
    if (end < start || padded_end > storage_rows ||
        (expert == groups - 1 && padded_end != storage_rows)) {
        asm volatile("trap;");
        return;
    }
    padded_offsets[expert] = static_cast<int32_t>(padded_end);
    group_counts[expert] = static_cast<int32_t>(end - start);
#endif
}

__global__ void sm103_zero_empty_wgrad_groups_kernel(
    __nv_bfloat16* output,
    const int32_t* group_counts,
    int64_t groups,
    int64_t elements_per_group
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    const int64_t expert = blockIdx.x;
    if (expert >= groups || group_counts[expert] != 0) {
        return;
    }
    for (int64_t index = threadIdx.x; index < elements_per_group; index += blockDim.x) {
        output[expert * elements_per_group + index] = __float2bfloat16(0.0f);
    }
#endif
}

template <bool kWeightIsKByN>
struct SM103GroupedBlockwiseGemm {
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = std::conditional_t<
        kWeightIsKByN,
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor>;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 16;
    static constexpr int AlignmentB = 16;
    static constexpr int AlignmentC = 8;
    static constexpr int AlignmentD = 8;
    using MmaTileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

    // A scales are native row-major [M, K/128].  For the ordinary NT path,
    // B scales are [N/128, K/128]; for the transposed-weight path the original
    // [K/128, N/128] storage is consumed without a copy.
    using ScaleConfig = cutlass::detail::Sm1xxBlockwiseScaleConfig<
        1,
        kBlockK,
        kBlockK,
        cute::UMMA::Major::K,
        kWeightIsKByN ? cute::UMMA::Major::MN : cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm103,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC*,
        AlignmentC,
        ElementD,
        LayoutD*,
        AlignmentD,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm103,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        cute::tuple<LayoutA*, LayoutSFA*>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutB*, LayoutSFB*>,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm103>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
};

struct MetadataBlob {
    std::vector<uint8_t> bytes;

    template <typename T>
    size_t append(const std::vector<T>& values) {
        constexpr size_t kMinimumAlignment = 16;
        const size_t alignment = std::max(kMinimumAlignment, alignof(T));
        const size_t offset = (bytes.size() + alignment - 1) / alignment * alignment;
        const size_t value_bytes = values.size() * sizeof(T);
        bytes.resize(offset + value_bytes);
        if (value_bytes != 0) {
            std::memcpy(bytes.data() + offset, values.data(), value_bytes);
        }
        return offset;
    }

    torch::Tensor copy_to_device(
        const torch::TensorOptions& options,
        cudaStream_t stream
    ) const {
        auto storage = torch::empty(
            {std::max<int64_t>(static_cast<int64_t>(bytes.size()), 1)},
            options.dtype(torch::kUInt8));
        if (!bytes.empty()) {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                storage.data_ptr(), bytes.data(), bytes.size(),
                cudaMemcpyHostToDevice, stream));
        }
        return storage;
    }
};

void check_grouped_bf16_wgrad_inputs(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const int64_t groups
) {
    check_bf16_matrix(left, "left");
    check_bf16_matrix(right, "right");
    TORCH_CHECK(left.device() == right.device(), "left and right must share a device");
    TORCH_CHECK(left.size(0) == right.size(0), "left and right row counts differ");
    TORCH_CHECK(groups > 0 && groups < 1024,
                "SM103 grouped BF16 wgrad requires between 1 and 1023 groups");
}

torch::Tensor launch_grouped_bf16_wgrad(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& metadata_device,
    const bool zero_expanded_padding
) {
    const int64_t groups = metadata_device.numel() / 2;
    if (left.size(0) == 0) {
        return torch::zeros(
            {groups, left.size(1), right.size(1)}, left.options());
    }

    c10::cuda::CUDAGuard guard(left.device());
    const auto stream = at::cuda::getCurrentCUDAStream(left.get_device());
    auto padded_offsets = metadata_device.narrow(0, 0, groups);
    const auto* counts_device = metadata_device.data_ptr<int32_t>() + groups;
    if (zero_expanded_padding) {
        constexpr int threads = 256;
        sm103_zero_expanded_wgrad_padding_kernel<<<groups, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(left.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(right.data_ptr()),
            counts_device,
            padded_offsets.data_ptr<int32_t>(),
            groups, left.size(1), right.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Exact SM103 validation above makes PyTorch's BF16 grouped tensor-core
    // implementation the only reachable dispatch.  There is no architecture
    // fallback in this companion entry point.
    auto output = at::_grouped_mm(
        left.transpose(0, 1), right, padded_offsets, std::nullopt, std::nullopt);
    constexpr int threads = 256;
    sm103_zero_empty_wgrad_groups_kernel<<<groups, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        counts_device,
        groups, left.size(1) * right.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor grouped_bf16_wgrad(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const std::vector<int64_t>& padded_group_counts
) {
    check_grouped_bf16_wgrad_inputs(left, right, padded_group_counts.size());

    std::vector<int32_t> metadata(padded_group_counts.size() * 2);
    int64_t storage_rows = 0;
    for (size_t expert = 0; expert < padded_group_counts.size(); ++expert) {
        const int64_t count = padded_group_counts[expert];
        TORCH_CHECK(count >= 0 && count <= std::numeric_limits<int32_t>::max(),
                    "group counts must fit non-negative int32");
        TORCH_CHECK(count == 0 || count % kBlockK == 0,
                    "BF16 wgrad group storage must be padded to 128 rows");
        storage_rows += count;
        TORCH_CHECK(storage_rows <= std::numeric_limits<int32_t>::max(),
                    "BF16 wgrad storage rows must fit int32");
        metadata[expert] = static_cast<int32_t>(storage_rows);
        metadata[padded_group_counts.size() + expert] = static_cast<int32_t>(count);
    }
    TORCH_CHECK(storage_rows == left.size(0),
                "group counts do not match BF16 wgrad storage rows");

    c10::cuda::CUDAGuard guard(left.device());
    const auto stream = at::cuda::getCurrentCUDAStream(left.get_device());
    auto metadata_device = torch::empty(
        {static_cast<int64_t>(metadata.size())},
        left.options().dtype(torch::kInt32));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        metadata_device.data_ptr(), metadata.data(), metadata.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    return launch_grouped_bf16_wgrad(left, right, metadata_device, false);
}

torch::Tensor grouped_bf16_wgrad_expanded(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& psum
) {
    check_sm103_device(psum);
    DG_CHECK_CONTIGUOUS(psum);
    TORCH_CHECK(psum.dim() == 1 && psum.scalar_type() == torch::kInt32,
                "expanded PSUM must be contiguous rank-1 int32");
    TORCH_CHECK(psum.device() == left.device(), "expanded PSUM device mismatch");
    check_grouped_bf16_wgrad_inputs(left, right, psum.numel());
    TORCH_CHECK(left.size(0) <= std::numeric_limits<int32_t>::max(),
                "BF16 wgrad storage rows must fit int32");

    auto metadata_device = torch::empty(
        {psum.numel() * 2}, left.options().dtype(torch::kInt32));
    auto padded_offsets = metadata_device.narrow(0, 0, psum.numel());
    auto group_counts = metadata_device.narrow(0, psum.numel(), psum.numel());
    c10::cuda::CUDAGuard guard(left.device());
    const auto stream = at::cuda::getCurrentCUDAStream(left.get_device());
    constexpr int threads = 256;
    const int64_t blocks = (psum.numel() + threads - 1) / threads;
    sm103_prepare_expanded_wgrad_metadata_kernel<<<blocks, threads, 0, stream>>>(
        psum.data_ptr<int32_t>(),
        padded_offsets.data_ptr<int32_t>(),
        group_counts.data_ptr<int32_t>(),
        psum.numel(), left.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return launch_grouped_bf16_wgrad(left, right, metadata_device, true);
}

template <bool kWeightIsKByN>
torch::Tensor grouped_fp8_block128_gemm_impl(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::vector<int64_t>& group_counts,
    const bool expanded_layout
) {
    using Config = SM103GroupedBlockwiseGemm<kWeightIsKByN>;
    using Problem = typename Config::ProblemShape::UnderlyingProblemShape;
    using ElementA = typename Config::ElementA;
    using ElementB = typename Config::ElementB;
    using ElementC = typename Config::ElementC;
    using ElementD = typename Config::ElementD;

    check_fp8_matrix_and_scales(activations, activation_scales, "activations");
    check_sm103_device(weights);
    DG_CHECK_CONTIGUOUS(weights);
    DG_CHECK_CUDA(weight_scales);
    DG_CHECK_CONTIGUOUS(weight_scales);
    TORCH_CHECK(weights.scalar_type() == torch::kFloat8_e4m3fn, "weights must be float8_e4m3fn");
    TORCH_CHECK(weights.dim() == 3, "weights must be rank 3");
    TORCH_CHECK(weight_scales.scalar_type() == torch::kFloat32 && weight_scales.dim() == 3,
                "weight_scales must be contiguous rank-3 float32");
    TORCH_CHECK(weights.device() == activations.device() && weight_scales.device() == activations.device(),
                "activations, weights, and scales must share a device");
    TORCH_CHECK(static_cast<int64_t>(group_counts.size()) == weights.size(0),
                "group_counts must contain one entry per local expert");

    const int64_t groups = weights.size(0);
    const int64_t k = kWeightIsKByN ? weights.size(1) : weights.size(2);
    const int64_t n = kWeightIsKByN ? weights.size(2) : weights.size(1);
    TORCH_CHECK(k > 0 && n > 0 && k % kBlockK == 0 && n % kBlockK == 0,
                "GEMM N and K dimensions must be positive multiples of 128");
    TORCH_CHECK(activations.size(1) == k, "activation K dimension does not match weights");
    if constexpr (kWeightIsKByN) {
        TORCH_CHECK(weight_scales.sizes() == torch::IntArrayRef({groups, k / kBlockK, n / kBlockK}),
                    "transposed weight scales must have shape [G, K/128, N/128]");
    } else {
        TORCH_CHECK(weight_scales.sizes() == torch::IntArrayRef({groups, n / kBlockK, k / kBlockK}),
                    "weight scales must have shape [G, N/128, K/128]");
    }

    int64_t storage_rows = 0;
    int64_t active_groups = 0;
    for (const int64_t count : group_counts) {
        TORCH_CHECK(count >= 0, "group counts must be non-negative");
        TORCH_CHECK(expanded_layout || count == 0 || count % 4 == 0,
                    "packed active group counts must be padded to a multiple of four");
        storage_rows += expanded_layout ? align_rows(count) : count;
        active_groups += count != 0;
    }
    TORCH_CHECK(storage_rows == activations.size(0),
                "group counts do not match the activation storage rows");

    auto output = torch::empty({storage_rows, n}, activations.options().dtype(torch::kBFloat16));
    if (active_groups == 0) {
        return output;
    }

    c10::cuda::CUDAGuard guard(activations.device());
    const auto stream = at::cuda::getCurrentCUDAStream(activations.get_device());

    std::vector<Problem> problems;
    std::vector<const ElementA*> ptr_a;
    std::vector<const ElementB*> ptr_b;
    std::vector<const ElementC*> ptr_c;
    std::vector<ElementD*> ptr_d;
    std::vector<const float*> ptr_sfa;
    std::vector<const float*> ptr_sfb;
    std::vector<typename Config::StrideA> stride_a;
    std::vector<typename Config::StrideB> stride_b;
    std::vector<typename Config::StrideC> stride_c;
    std::vector<typename Config::StrideD> stride_d;
    std::vector<typename Config::LayoutSFA> layout_sfa;
    std::vector<typename Config::LayoutSFB> layout_sfb;
    problems.reserve(active_groups);
    ptr_a.reserve(active_groups);
    ptr_b.reserve(active_groups);
    ptr_c.reserve(active_groups);
    ptr_d.reserve(active_groups);
    ptr_sfa.reserve(active_groups);
    ptr_sfb.reserve(active_groups);
    stride_a.reserve(active_groups);
    stride_b.reserve(active_groups);
    stride_c.reserve(active_groups);
    stride_d.reserve(active_groups);
    layout_sfa.reserve(active_groups);
    layout_sfb.reserve(active_groups);

    auto* activation_ptr = reinterpret_cast<const ElementA*>(activations.data_ptr());
    auto* activation_scale_ptr = activation_scales.data_ptr<float>();
    auto* weight_ptr = reinterpret_cast<const ElementB*>(weights.data_ptr());
    auto* weight_scale_ptr = weight_scales.data_ptr<float>();
    auto* output_ptr = reinterpret_cast<ElementD*>(output.data_ptr());
    int64_t row_offset = 0;
    const int64_t weight_elements = k * n;
    const int64_t weight_scale_elements = (k / kBlockK) * (n / kBlockK);
    for (int64_t expert = 0; expert < groups; ++expert) {
        const int64_t m = group_counts[expert];
        if (m == 0) {
            continue;
        }
        problems.emplace_back(cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)));
        ptr_a.push_back(activation_ptr + row_offset * k);
        ptr_b.push_back(weight_ptr + expert * weight_elements);
        ptr_c.push_back(reinterpret_cast<const ElementC*>(output_ptr + row_offset * n));
        ptr_d.push_back(output_ptr + row_offset * n);
        ptr_sfa.push_back(activation_scale_ptr + row_offset * (k / kBlockK));
        ptr_sfb.push_back(weight_scale_ptr + expert * weight_scale_elements);
        stride_a.push_back(cutlass::make_cute_packed_stride(
            typename Config::StrideA{}, cute::make_shape(static_cast<int>(m), static_cast<int>(k), 1)));
        stride_b.push_back(cutlass::make_cute_packed_stride(
            typename Config::StrideB{}, cute::make_shape(static_cast<int>(n), static_cast<int>(k), 1)));
        stride_c.push_back(cutlass::make_cute_packed_stride(
            typename Config::StrideC{}, cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1)));
        stride_d.push_back(cutlass::make_cute_packed_stride(
            typename Config::StrideD{}, cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1)));
        layout_sfa.push_back(Config::ScaleConfig::tile_atom_to_shape_SFA(
            cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1)));
        layout_sfb.push_back(Config::ScaleConfig::tile_atom_to_shape_SFB(
            cute::make_shape(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1)));
        row_offset += expanded_layout ? align_rows(m) : m;
    }

    const auto metadata_options = activations.options().dtype(torch::kUInt8);
    MetadataBlob metadata_blob;
    const auto problems_offset = metadata_blob.append(problems);
    const auto ptr_a_offset = metadata_blob.append(ptr_a);
    const auto ptr_b_offset = metadata_blob.append(ptr_b);
    const auto ptr_c_offset = metadata_blob.append(ptr_c);
    const auto ptr_d_offset = metadata_blob.append(ptr_d);
    const auto ptr_sfa_offset = metadata_blob.append(ptr_sfa);
    const auto ptr_sfb_offset = metadata_blob.append(ptr_sfb);
    const auto stride_a_offset = metadata_blob.append(stride_a);
    const auto stride_b_offset = metadata_blob.append(stride_b);
    const auto stride_c_offset = metadata_blob.append(stride_c);
    const auto stride_d_offset = metadata_blob.append(stride_d);
    const auto layout_sfa_offset = metadata_blob.append(layout_sfa);
    const auto layout_sfb_offset = metadata_blob.append(layout_sfb);
    auto metadata_device = metadata_blob.copy_to_device(metadata_options, stream);
    auto* metadata_base = metadata_device.data_ptr<uint8_t>();

    cutlass::KernelHardwareInfo hardware_info;
    hardware_info.device_id = activations.get_device();
    hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hardware_info.device_id);

    typename Config::Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {static_cast<int>(active_groups),
         reinterpret_cast<Problem*>(metadata_base + problems_offset),
         problems.data()},
        {reinterpret_cast<const ElementA**>(metadata_base + ptr_a_offset),
         reinterpret_cast<typename Config::StrideA*>(metadata_base + stride_a_offset),
         reinterpret_cast<const ElementB**>(metadata_base + ptr_b_offset),
         reinterpret_cast<typename Config::StrideB*>(metadata_base + stride_b_offset),
         reinterpret_cast<const float**>(metadata_base + ptr_sfa_offset),
         reinterpret_cast<typename Config::LayoutSFA*>(metadata_base + layout_sfa_offset),
         reinterpret_cast<const float**>(metadata_base + ptr_sfb_offset),
         reinterpret_cast<typename Config::LayoutSFB*>(metadata_base + layout_sfb_offset)},
        {{},
         reinterpret_cast<const ElementC**>(metadata_base + ptr_c_offset),
         reinterpret_cast<typename Config::StrideC*>(metadata_base + stride_c_offset),
         reinterpret_cast<ElementD**>(metadata_base + ptr_d_offset),
         reinterpret_cast<typename Config::StrideD*>(metadata_base + stride_d_offset)},
        hardware_info};
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    typename Config::Gemm gemm;
    const auto implement_status = gemm.can_implement(arguments);
    TORCH_CHECK(implement_status == cutlass::Status::kSuccess,
                "SM103 grouped FP8-block128 GEMM cannot implement the requested problem: ",
                cutlassGetStatusString(implement_status));
    const int64_t workspace_bytes = static_cast<int64_t>(gemm.get_workspace_size(arguments));
    auto workspace = torch::empty(
        {std::max<int64_t>(workspace_bytes, 1)}, metadata_options);
    const auto initialize_status = gemm.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(initialize_status == cutlass::Status::kSuccess,
                "SM103 grouped FP8-block128 GEMM initialization failed: ",
                cutlassGetStatusString(initialize_status));
    const auto run_status = gemm.run(stream);
    TORCH_CHECK(run_status == cutlass::Status::kSuccess,
                "SM103 grouped FP8-block128 GEMM launch failed: ",
                cutlassGetStatusString(run_status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool kWeightIsKByN>
torch::Tensor grouped_fp8_block128_w13_gemm_nt_canonical_impl(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& canonical_weights,
    const torch::Tensor& canonical_weight_scales,
    const std::vector<int64_t>& group_counts,
    const bool expanded_layout
) {
    // FireTitan stores [gate_0, up_0, gate_1, up_1, ...].  Submit two
    // independent N=H problems per active expert, pointing the first at the up
    // weight and output columns [0,H), and the second at the gate weight and
    // output columns [H,2H).  This preserves the fused [up; gate] ABI without
    // materializing a multi-gigabyte reordered weight tensor.
    static_assert(!kWeightIsKByN, "canonical W13 forward requires NT weights");
    using Config = SM103GroupedBlockwiseGemm<kWeightIsKByN>;
    using Problem = typename Config::ProblemShape::UnderlyingProblemShape;
    using ElementA = typename Config::ElementA;
    using ElementB = typename Config::ElementB;
    using ElementC = typename Config::ElementC;
    using ElementD = typename Config::ElementD;

    check_fp8_matrix_and_scales(activations, activation_scales, "activations");
    check_sm103_device(canonical_weights);
    DG_CHECK_CONTIGUOUS(canonical_weights);
    DG_CHECK_CUDA(canonical_weight_scales);
    DG_CHECK_CONTIGUOUS(canonical_weight_scales);
    TORCH_CHECK(canonical_weights.scalar_type() == torch::kFloat8_e4m3fn,
                "canonical W13 weights must be float8_e4m3fn");
    TORCH_CHECK(canonical_weights.dim() == 3 && canonical_weights.size(0) % 2 == 0,
                "canonical W13 weights must have shape [2E, H, D]");
    TORCH_CHECK(canonical_weight_scales.scalar_type() == torch::kFloat32 &&
                    canonical_weight_scales.dim() == 3,
                "canonical W13 scales must be contiguous rank-3 float32");
    TORCH_CHECK(canonical_weights.device() == activations.device() &&
                    canonical_weight_scales.device() == activations.device(),
                "activations, canonical W13 weights, and scales must share a device");

    const int64_t groups = canonical_weights.size(0) / 2;
    const int64_t hidden = canonical_weights.size(1);
    const int64_t k = canonical_weights.size(2);
    const int64_t doubled_hidden = hidden * 2;
    TORCH_CHECK(static_cast<int64_t>(group_counts.size()) == groups,
                "group_counts must contain one entry per local expert");
    TORCH_CHECK(hidden > 0 && k > 0 && hidden % kBlockK == 0 && k % kBlockK == 0,
                "canonical W13 H and D dimensions must be positive multiples of 128");
    TORCH_CHECK(activations.size(1) == k,
                "activation K dimension does not match canonical W13 weights");
    TORCH_CHECK(canonical_weight_scales.sizes() ==
                    torch::IntArrayRef({groups * 2, hidden / kBlockK, k / kBlockK}),
                "canonical W13 scales must have shape [2E, H/128, D/128]");

    int64_t storage_rows = 0;
    int64_t active_groups = 0;
    for (const int64_t count : group_counts) {
        TORCH_CHECK(count >= 0, "group counts must be non-negative");
        TORCH_CHECK(expanded_layout || count == 0 || count % 4 == 0,
                    "packed active group counts must be padded to a multiple of four");
        storage_rows += expanded_layout ? align_rows(count) : count;
        active_groups += count != 0;
    }
    TORCH_CHECK(storage_rows == activations.size(0),
                "group counts do not match the activation storage rows");

    auto output = torch::empty(
        {storage_rows, doubled_hidden}, activations.options().dtype(torch::kBFloat16));
    if (active_groups == 0) {
        return output;
    }

    c10::cuda::CUDAGuard guard(activations.device());
    const auto stream = at::cuda::getCurrentCUDAStream(activations.get_device());
    const int64_t active_problems = active_groups * 2;

    std::vector<Problem> problems;
    std::vector<const ElementA*> ptr_a;
    std::vector<const ElementB*> ptr_b;
    std::vector<const ElementC*> ptr_c;
    std::vector<ElementD*> ptr_d;
    std::vector<const float*> ptr_sfa;
    std::vector<const float*> ptr_sfb;
    std::vector<typename Config::StrideA> stride_a;
    std::vector<typename Config::StrideB> stride_b;
    std::vector<typename Config::StrideC> stride_c;
    std::vector<typename Config::StrideD> stride_d;
    std::vector<typename Config::LayoutSFA> layout_sfa;
    std::vector<typename Config::LayoutSFB> layout_sfb;
    problems.reserve(active_problems);
    ptr_a.reserve(active_problems);
    ptr_b.reserve(active_problems);
    ptr_c.reserve(active_problems);
    ptr_d.reserve(active_problems);
    ptr_sfa.reserve(active_problems);
    ptr_sfb.reserve(active_problems);
    stride_a.reserve(active_problems);
    stride_b.reserve(active_problems);
    stride_c.reserve(active_problems);
    stride_d.reserve(active_problems);
    layout_sfa.reserve(active_problems);
    layout_sfb.reserve(active_problems);

    auto* activation_ptr = reinterpret_cast<const ElementA*>(activations.data_ptr());
    auto* activation_scale_ptr = activation_scales.data_ptr<float>();
    auto* weight_ptr = reinterpret_cast<const ElementB*>(canonical_weights.data_ptr());
    auto* weight_scale_ptr = canonical_weight_scales.data_ptr<float>();
    auto* output_ptr = reinterpret_cast<ElementD*>(output.data_ptr());
    const int64_t weight_elements = hidden * k;
    const int64_t weight_scale_elements = (hidden / kBlockK) * (k / kBlockK);
    int64_t row_offset = 0;
    constexpr int64_t canonical_pair_for_output_half[2] = {1, 0};  // up, gate
    for (int64_t expert = 0; expert < groups; ++expert) {
        const int64_t m = group_counts[expert];
        if (m == 0) {
            continue;
        }
        for (int64_t output_half = 0; output_half < 2; ++output_half) {
            const int64_t canonical_pair = canonical_pair_for_output_half[output_half];
            const int64_t canonical_index = expert * 2 + canonical_pair;
            problems.emplace_back(cute::make_shape(
                static_cast<int>(m), static_cast<int>(hidden), static_cast<int>(k)));
            ptr_a.push_back(activation_ptr + row_offset * k);
            ptr_b.push_back(weight_ptr + canonical_index * weight_elements);
            auto* output_half_ptr =
                output_ptr + row_offset * doubled_hidden + output_half * hidden;
            ptr_c.push_back(reinterpret_cast<const ElementC*>(output_half_ptr));
            ptr_d.push_back(output_half_ptr);
            ptr_sfa.push_back(activation_scale_ptr + row_offset * (k / kBlockK));
            ptr_sfb.push_back(weight_scale_ptr + canonical_index * weight_scale_elements);
            stride_a.push_back(cutlass::make_cute_packed_stride(
                typename Config::StrideA{},
                cute::make_shape(static_cast<int>(m), static_cast<int>(k), 1)));
            stride_b.push_back(cutlass::make_cute_packed_stride(
                typename Config::StrideB{},
                cute::make_shape(static_cast<int>(hidden), static_cast<int>(k), 1)));
            // Use the full 2H row extent for C/D while each problem writes only
            // one H-wide half.  The two problems therefore never overlap.
            stride_c.push_back(cutlass::make_cute_packed_stride(
                typename Config::StrideC{},
                cute::make_shape(static_cast<int>(m), static_cast<int>(doubled_hidden), 1)));
            stride_d.push_back(cutlass::make_cute_packed_stride(
                typename Config::StrideD{},
                cute::make_shape(static_cast<int>(m), static_cast<int>(doubled_hidden), 1)));
            layout_sfa.push_back(Config::ScaleConfig::tile_atom_to_shape_SFA(
                cute::make_shape(static_cast<int>(m), static_cast<int>(hidden), static_cast<int>(k), 1)));
            layout_sfb.push_back(Config::ScaleConfig::tile_atom_to_shape_SFB(
                cute::make_shape(static_cast<int>(m), static_cast<int>(hidden), static_cast<int>(k), 1)));
        }
        row_offset += expanded_layout ? align_rows(m) : m;
    }

    const auto metadata_options = activations.options().dtype(torch::kUInt8);
    MetadataBlob metadata_blob;
    const auto problems_offset = metadata_blob.append(problems);
    const auto ptr_a_offset = metadata_blob.append(ptr_a);
    const auto ptr_b_offset = metadata_blob.append(ptr_b);
    const auto ptr_c_offset = metadata_blob.append(ptr_c);
    const auto ptr_d_offset = metadata_blob.append(ptr_d);
    const auto ptr_sfa_offset = metadata_blob.append(ptr_sfa);
    const auto ptr_sfb_offset = metadata_blob.append(ptr_sfb);
    const auto stride_a_offset = metadata_blob.append(stride_a);
    const auto stride_b_offset = metadata_blob.append(stride_b);
    const auto stride_c_offset = metadata_blob.append(stride_c);
    const auto stride_d_offset = metadata_blob.append(stride_d);
    const auto layout_sfa_offset = metadata_blob.append(layout_sfa);
    const auto layout_sfb_offset = metadata_blob.append(layout_sfb);
    auto metadata_device = metadata_blob.copy_to_device(metadata_options, stream);
    auto* metadata_base = metadata_device.data_ptr<uint8_t>();

    cutlass::KernelHardwareInfo hardware_info;
    hardware_info.device_id = activations.get_device();
    hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hardware_info.device_id);

    typename Config::Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {static_cast<int>(active_problems),
         reinterpret_cast<Problem*>(metadata_base + problems_offset),
         problems.data()},
        {reinterpret_cast<const ElementA**>(metadata_base + ptr_a_offset),
         reinterpret_cast<typename Config::StrideA*>(metadata_base + stride_a_offset),
         reinterpret_cast<const ElementB**>(metadata_base + ptr_b_offset),
         reinterpret_cast<typename Config::StrideB*>(metadata_base + stride_b_offset),
         reinterpret_cast<const float**>(metadata_base + ptr_sfa_offset),
         reinterpret_cast<typename Config::LayoutSFA*>(metadata_base + layout_sfa_offset),
         reinterpret_cast<const float**>(metadata_base + ptr_sfb_offset),
         reinterpret_cast<typename Config::LayoutSFB*>(metadata_base + layout_sfb_offset)},
        {{},
         reinterpret_cast<const ElementC**>(metadata_base + ptr_c_offset),
         reinterpret_cast<typename Config::StrideC*>(metadata_base + stride_c_offset),
         reinterpret_cast<ElementD**>(metadata_base + ptr_d_offset),
         reinterpret_cast<typename Config::StrideD*>(metadata_base + stride_d_offset)},
        hardware_info};
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    typename Config::Gemm gemm;
    const auto implement_status = gemm.can_implement(arguments);
    TORCH_CHECK(implement_status == cutlass::Status::kSuccess,
                "SM103 canonical W13 grouped GEMM cannot implement the requested problem: ",
                cutlassGetStatusString(implement_status));
    const int64_t workspace_bytes = static_cast<int64_t>(gemm.get_workspace_size(arguments));
    auto workspace = torch::empty(
        {std::max<int64_t>(workspace_bytes, 1)}, metadata_options);
    const auto initialize_status = gemm.initialize(arguments, workspace.data_ptr(), stream);
    TORCH_CHECK(initialize_status == cutlass::Status::kSuccess,
                "SM103 canonical W13 grouped GEMM initialization failed: ",
                cutlassGetStatusString(initialize_status));
    const auto run_status = gemm.run(stream);
    TORCH_CHECK(run_status == cutlass::Status::kSuccess,
                "SM103 canonical W13 grouped GEMM launch failed: ",
                cutlassGetStatusString(run_status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor grouped_fp8_block128_gemm_nt(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_gemm_impl<false>(
        activations, activation_scales, weights, weight_scales, group_counts, false);
}

torch::Tensor grouped_fp8_block128_gemm_nn(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_gemm_impl<true>(
        activations, activation_scales, weights, weight_scales, group_counts, false);
}

torch::Tensor grouped_fp8_block128_gemm_nt_expanded(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_gemm_impl<false>(
        activations, activation_scales, weights, weight_scales, group_counts, true);
}

torch::Tensor grouped_fp8_block128_gemm_nn_expanded(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_gemm_impl<true>(
        activations, activation_scales, weights, weight_scales, group_counts, true);
}

torch::Tensor grouped_fp8_block128_w13_gemm_nt_canonical(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& canonical_weights,
    const torch::Tensor& canonical_weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_w13_gemm_nt_canonical_impl<false>(
        activations,
        activation_scales,
        canonical_weights,
        canonical_weight_scales,
        group_counts,
        false);
}

torch::Tensor grouped_fp8_block128_w13_gemm_nt_canonical_expanded(
    const torch::Tensor& activations,
    const torch::Tensor& activation_scales,
    const torch::Tensor& canonical_weights,
    const torch::Tensor& canonical_weight_scales,
    const std::vector<int64_t>& group_counts
) {
    return grouped_fp8_block128_w13_gemm_nt_canonical_impl<false>(
        activations,
        activation_scales,
        canonical_weights,
        canonical_weight_scales,
        group_counts,
        true);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_bf16(const torch::Tensor& input) {
    check_bf16_matrix(input, "input");
    c10::cuda::CUDAGuard guard(input.device());
    const auto rows = input.size(0);
    const auto columns = input.size(1);
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty({rows, columns / kBlockK}, input.options().dtype(torch::kFloat32));
    if (rows != 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        sm103_quantize_bf16_e4m3_group128_kernel<<<rows * (columns / kBlockK), kBlockK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
            scales.data_ptr<float>(), rows, columns
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, scales};
}

torch::Tensor dequantize_fp8(
    const torch::Tensor& input,
    const torch::Tensor& scales
) {
    check_fp8_matrix_and_scales(input, scales, "input");
    c10::cuda::CUDAGuard guard(input.device());
    const auto rows = input.size(0);
    const auto columns = input.size(1);
    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBFloat16));
    if (input.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (input.numel() + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        sm103_dequantize_e4m3_group128_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(input.data_ptr()),
            scales.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), rows, columns
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> swiglu_quantize(const torch::Tensor& preactivation) {
    check_bf16_matrix(preactivation, "preactivation");
    TORCH_CHECK(preactivation.size(1) % (2 * kBlockK) == 0, "preactivation width must be 2 * H with H divisible by 128");
    c10::cuda::CUDAGuard guard(preactivation.device());
    const auto rows = preactivation.size(0);
    const auto hidden = preactivation.size(1) / 2;
    auto output = torch::empty({rows, hidden}, preactivation.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty({rows, hidden / kBlockK}, preactivation.options().dtype(torch::kFloat32));
    if (rows != 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(preactivation.get_device());
        sm103_swiglu_quantize_group128_kernel<<<rows * (hidden / kBlockK), kBlockK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(preactivation.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
            scales.data_ptr<float>(), rows, hidden
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, scales};
}

torch::Tensor swiglu_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& preactivation
) {
    check_bf16_matrix(grad_output, "grad_output");
    check_sm103_device(preactivation);
    DG_CHECK_CONTIGUOUS(preactivation);
    TORCH_CHECK(preactivation.scalar_type() == torch::kBFloat16, "preactivation must be bfloat16");
    TORCH_CHECK(preactivation.dim() == 2, "preactivation must be rank 2");
    TORCH_CHECK(preactivation.size(0) == grad_output.size(0), "row count mismatch");
    TORCH_CHECK(preactivation.size(1) == grad_output.size(1) * 2, "preactivation width mismatch");
    TORCH_CHECK(preactivation.device() == grad_output.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(grad_output.device());
    auto grad_preactivation = torch::empty_like(preactivation);
    if (grad_output.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (grad_output.numel() + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
        sm103_swiglu_backward_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(preactivation.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_preactivation.data_ptr()),
            grad_output.size(0), grad_output.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_preactivation;
}

torch::Tensor swiglu_backward_canonical(
    const torch::Tensor& grad_output,
    const torch::Tensor& preactivation
) {
    check_bf16_matrix(grad_output, "grad_output");
    check_sm103_device(preactivation);
    DG_CHECK_CONTIGUOUS(preactivation);
    TORCH_CHECK(preactivation.scalar_type() == torch::kBFloat16,
                "preactivation must be bfloat16");
    TORCH_CHECK(preactivation.dim() == 2, "preactivation must be rank 2");
    TORCH_CHECK(preactivation.size(0) == grad_output.size(0), "row count mismatch");
    TORCH_CHECK(preactivation.size(1) == grad_output.size(1) * 2,
                "preactivation width mismatch");
    TORCH_CHECK(preactivation.device() == grad_output.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(grad_output.device());
    auto grad_preactivation = torch::empty_like(preactivation);
    if (grad_output.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (grad_output.numel() + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
        sm103_swiglu_backward_canonical_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(preactivation.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(grad_preactivation.data_ptr()),
            grad_output.size(0), grad_output.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return grad_preactivation;
}

std::tuple<torch::Tensor, torch::Tensor> route_scale_quantize(
    const torch::Tensor& grad_output,
    const torch::Tensor& route_scores,
    const torch::Tensor& route_order
) {
    check_bf16_matrix(grad_output, "grad_output");
    check_sm103_device(route_scores);
    DG_CHECK_CONTIGUOUS(route_scores);
    DG_CHECK_CUDA(route_order);
    DG_CHECK_CONTIGUOUS(route_order);
    TORCH_CHECK(route_scores.dim() == 2 && route_scores.scalar_type() == torch::kFloat32,
                "route_scores must be contiguous rank-2 float32");
    TORCH_CHECK(route_order.dim() == 1 && route_order.scalar_type() == torch::kInt64,
                "route_order must be contiguous rank-1 int64");
    TORCH_CHECK(route_scores.size(0) == grad_output.size(0), "token count mismatch");
    TORCH_CHECK(route_scores.device() == grad_output.device() && route_order.device() == grad_output.device(),
                "all tensors must share a device");
    c10::cuda::CUDAGuard guard(grad_output.device());
    const auto routes = route_order.numel();
    const auto hidden = grad_output.size(1);
    auto output = torch::empty({routes, hidden}, grad_output.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty({routes, hidden / kBlockK}, grad_output.options().dtype(torch::kFloat32));
    if (routes != 0) {
        const auto stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
        sm103_route_scale_quantize_group128_kernel<<<routes * (hidden / kBlockK), kBlockK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            route_scores.data_ptr<float>(), route_order.data_ptr<int64_t>(),
            reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()), scales.data_ptr<float>(),
            routes, hidden, route_scores.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, scales};
}

torch::Tensor post_down_combine(
    const torch::Tensor& route_output,
    const torch::Tensor& route_scores
) {
    check_bf16_matrix(route_output, "route_output");
    check_sm103_device(route_scores);
    DG_CHECK_CONTIGUOUS(route_scores);
    TORCH_CHECK(route_scores.dim() == 2 && route_scores.scalar_type() == torch::kFloat32,
                "route_scores must be contiguous rank-2 float32");
    TORCH_CHECK(route_output.size(0) == route_scores.numel(), "route count mismatch");
    TORCH_CHECK(route_scores.device() == route_output.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(route_output.device());
    auto output = torch::empty(
        {route_scores.size(0), route_output.size(1)},
        route_output.options().dtype(torch::kBFloat16)
    );
    if (output.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (output.numel() + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream(route_output.get_device());
        sm103_post_down_combine_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(route_output.data_ptr()),
            route_scores.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            route_scores.size(0), route_output.size(1), route_scores.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor route_sum(
    const torch::Tensor& route_grad,
    int64_t num_tokens,
    int64_t topk
) {
    check_bf16_matrix(route_grad, "route_grad");
    TORCH_CHECK(num_tokens >= 0 && topk > 0, "invalid token/top-k dimensions");
    TORCH_CHECK(route_grad.size(0) == num_tokens * topk, "route count mismatch");
    c10::cuda::CUDAGuard guard(route_grad.device());
    auto output = torch::empty({num_tokens, route_grad.size(1)}, route_grad.options());
    if (output.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (output.numel() + threads - 1) / threads;
        const auto stream = at::cuda::getCurrentCUDAStream(route_grad.get_device());
        sm103_route_sum_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(route_grad.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            num_tokens, route_grad.size(1), topk
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor post_down_score_grad(
    const torch::Tensor& route_output,
    const torch::Tensor& grad_output,
    int64_t topk
) {
    check_bf16_matrix(route_output, "route_output");
    check_bf16_matrix(grad_output, "grad_output");
    TORCH_CHECK(topk > 0, "topk must be positive");
    TORCH_CHECK(route_output.size(0) == grad_output.size(0) * topk, "route count mismatch");
    TORCH_CHECK(route_output.size(1) == grad_output.size(1), "hidden dimension mismatch");
    TORCH_CHECK(route_output.device() == grad_output.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(route_output.device());
    auto output = torch::empty({grad_output.size(0), topk}, grad_output.options().dtype(torch::kFloat32));
    if (route_output.size(0) != 0) {
        constexpr int threads = 256;
        const auto stream = at::cuda::getCurrentCUDAStream(route_output.get_device());
        sm103_post_down_score_grad_kernel<<<route_output.size(0), threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(route_output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            output.data_ptr<float>(), route_output.size(0), route_output.size(1), topk
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

void check_expanded_routes(const torch::Tensor& routes, const torch::Device& device) {
    check_sm103_device(routes);
    TORCH_CHECK(routes.dim() == 2, "expanded routes must be rank 2");
    TORCH_CHECK(routes.scalar_type() == torch::kInt32 || routes.scalar_type() == torch::kInt64,
                "expanded routes must be int32 or int64");
    TORCH_CHECK(routes.device() == device, "expanded routes must share the payload device");
    TORCH_CHECK(routes.stride(1) > 0 && routes.stride(0) > 0,
                "expanded routes must have positive strides");
}

torch::Tensor expanded_post_down_scale(
    const torch::Tensor& route_output,
    const torch::Tensor& route_scores
) {
    check_bf16_matrix(route_output, "route_output");
    check_sm103_device(route_scores);
    DG_CHECK_CONTIGUOUS(route_scores);
    TORCH_CHECK(route_scores.scalar_type() == torch::kFloat32 && route_scores.dim() == 1,
                "expanded route scores must be contiguous rank-1 float32");
    TORCH_CHECK(route_scores.size(0) == route_output.size(0),
                "expanded route score row count mismatch");
    TORCH_CHECK(route_scores.device() == route_output.device(), "device mismatch");
    auto output = torch::empty_like(route_output);
    if (output.numel() != 0) {
        constexpr int threads = 256;
        const auto blocks = (output.numel() + threads - 1) / threads;
        c10::cuda::CUDAGuard guard(route_output.device());
        const auto stream = at::cuda::getCurrentCUDAStream(route_output.get_device());
        sm103_expanded_post_down_scale_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(route_output.data_ptr()),
            route_scores.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            route_output.size(0),
            route_output.size(1));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor expand_compact_routes(
    const torch::Tensor& compact,
    const torch::Tensor& routes,
    const int64_t expanded_rows
) {
    check_bf16_matrix(compact, "compact");
    check_expanded_routes(routes, compact.device());
    TORCH_CHECK(expanded_rows >= 0, "expanded row count must be non-negative");
    TORCH_CHECK(routes.size(0) == compact.size(0),
                "route metadata and compact payload row counts differ");
    auto output = torch::empty(
        {expanded_rows, compact.size(1)}, compact.options().dtype(torch::kBFloat16));
    if (routes.numel() != 0) {
        constexpr int threads = 256;
        c10::cuda::CUDAGuard guard(compact.device());
        const auto stream = at::cuda::getCurrentCUDAStream(compact.get_device());
        const auto blocks = routes.numel();
        if (routes.scalar_type() == torch::kInt32) {
            sm103_expand_compact_routes_kernel<int32_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(compact.data_ptr()),
                routes.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                compact.size(0), routes.size(1), compact.size(1), expanded_rows,
                routes.stride(0), routes.stride(1));
        } else {
            sm103_expand_compact_routes_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(compact.data_ptr()),
                routes.data_ptr<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                compact.size(0), routes.size(1), compact.size(1), expanded_rows,
                routes.stride(0), routes.stride(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor collapse_expanded_routes(
    const torch::Tensor& expanded,
    const torch::Tensor& routes
) {
    check_bf16_matrix(expanded, "expanded");
    check_expanded_routes(routes, expanded.device());
    auto output = torch::empty(
        {routes.size(0), expanded.size(1)}, expanded.options().dtype(torch::kBFloat16));
    if (output.numel() != 0) {
        constexpr int threads = 256;
        c10::cuda::CUDAGuard guard(expanded.device());
        const auto stream = at::cuda::getCurrentCUDAStream(expanded.get_device());
        const auto blocks = routes.size(0);
        if (routes.scalar_type() == torch::kInt32) {
            sm103_collapse_expanded_routes_kernel<int32_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(expanded.data_ptr()),
                routes.data_ptr<int32_t>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                routes.size(0), routes.size(1), expanded.size(1), expanded.size(0),
                routes.stride(0), routes.stride(1));
        } else {
            sm103_collapse_expanded_routes_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(expanded.data_ptr()),
                routes.data_ptr<int64_t>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                routes.size(0), routes.size(1), expanded.size(1), expanded.size(0),
                routes.stride(0), routes.stride(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor expanded_post_down_score_grad(
    const torch::Tensor& route_output,
    const torch::Tensor& grad_output,
    const torch::Tensor& routes
) {
    check_bf16_matrix(route_output, "route_output");
    check_bf16_matrix(grad_output, "grad_output");
    check_expanded_routes(routes, route_output.device());
    TORCH_CHECK(route_output.sizes() == grad_output.sizes(),
                "expanded route output and gradient shapes differ");
    TORCH_CHECK(route_output.device() == grad_output.device(), "device mismatch");
    auto output = torch::empty(
        routes.sizes(), route_output.options().dtype(torch::kFloat32));
    if (routes.numel() != 0) {
        constexpr int threads = 256;
        c10::cuda::CUDAGuard guard(route_output.device());
        const auto stream = at::cuda::getCurrentCUDAStream(route_output.get_device());
        const auto blocks = routes.numel();
        if (routes.scalar_type() == torch::kInt32) {
            sm103_expanded_post_down_score_grad_kernel<int32_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(route_output.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                routes.data_ptr<int32_t>(), output.data_ptr<float>(),
                routes.size(0), routes.size(1), route_output.size(1), route_output.size(0),
                routes.stride(0), routes.stride(1));
        } else {
            sm103_expanded_post_down_score_grad_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(route_output.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
                routes.data_ptr<int64_t>(), output.data_ptr<float>(),
                routes.size(0), routes.size(1), route_output.size(1), route_output.size(0),
                routes.stride(0), routes.stride(1));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> expanded_route_scale_quantize(
    const torch::Tensor& grad_output,
    const torch::Tensor& route_scores
) {
    check_bf16_matrix(grad_output, "grad_output");
    check_sm103_device(route_scores);
    DG_CHECK_CONTIGUOUS(route_scores);
    TORCH_CHECK(route_scores.scalar_type() == torch::kFloat32 && route_scores.dim() == 1,
                "expanded route scores must be contiguous rank-1 float32");
    TORCH_CHECK(route_scores.size(0) == grad_output.size(0),
                "expanded route score row count mismatch");
    TORCH_CHECK(route_scores.device() == grad_output.device(), "device mismatch");
    const auto rows = grad_output.size(0);
    const auto hidden = grad_output.size(1);
    auto output = torch::empty(
        grad_output.sizes(), grad_output.options().dtype(torch::kFloat8_e4m3fn));
    auto scales = torch::empty(
        {rows, hidden / kBlockK}, grad_output.options().dtype(torch::kFloat32));
    if (rows != 0) {
        c10::cuda::CUDAGuard guard(grad_output.device());
        const auto stream = at::cuda::getCurrentCUDAStream(grad_output.get_device());
        sm103_expanded_route_scale_quantize_group128_kernel<<<
            rows * (hidden / kBlockK), kBlockK, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
            route_scores.data_ptr<float>(),
            reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
            scales.data_ptr<float>(), rows, hidden);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output, scales};
}

pybind11::dict capabilities() {
    namespace py = pybind11;
    py::dict result;
    result["name"] = "fp8_block128_mega_moe";
    result["architecture"] = "sm103";
    result["compute_capability"] = py::make_tuple(kRequiredMajor, kRequiredMinor);
    result["activation_dtype"] = "float8_e4m3fn";
    result["weight_dtype"] = "float8_e4m3fn";
    result["scale_dtype"] = "float32";
    result["activation_group_k"] = kBlockK;
    result["weight_block_m"] = kBlockK;
    result["weight_block_k"] = kBlockK;
    result["route_score_placement"] = "post_down";
    result["fallback"] = py::none();
    result["native_symbols"] = py::make_tuple(
        "sm103_fp8_block128_quantize",
        "sm103_fp8_block128_dequantize",
        "sm103_fp8_block128_grouped_gemm_nt",
        "sm103_fp8_block128_grouped_gemm_nn",
        "sm103_fp8_block128_grouped_w13_gemm_nt_canonical",
        "sm103_fp8_block128_grouped_gemm_nt_expanded",
        "sm103_fp8_block128_grouped_gemm_nn_expanded",
        "sm103_fp8_block128_grouped_w13_gemm_nt_canonical_expanded",
        "sm103_fp8_block128_swiglu_quantize",
        "sm103_fp8_block128_swiglu_backward",
        "sm103_fp8_block128_swiglu_backward_canonical",
        "sm103_fp8_block128_route_scale_quantize",
        "sm103_fp8_block128_post_down_combine",
        "sm103_fp8_block128_post_down_score_grad",
        "sm103_fp8_block128_route_sum",
        "sm103_fp8_block128_expanded_post_down_scale",
        "sm103_fp8_block128_expand_compact_routes",
        "sm103_fp8_block128_collapse_expanded_routes",
        "sm103_fp8_block128_expanded_post_down_score_grad",
        "sm103_fp8_block128_expanded_route_scale_quantize",
        "sm103_fp8_block128_grouped_bf16_wgrad",
        "sm103_fp8_block128_grouped_bf16_wgrad_expanded"
    );
    return result;
}

}  // namespace

void register_apis(pybind11::module_& m) {
    m.def("get_sm103_fp8_block128_capabilities", &capabilities);
    m.def("sm103_fp8_block128_quantize", &quantize_bf16, pybind11::arg("input"));
    m.def("sm103_fp8_block128_dequantize", &dequantize_fp8,
          pybind11::arg("input"), pybind11::arg("scales"));
    m.def("sm103_fp8_block128_grouped_gemm_nt", &grouped_fp8_block128_gemm_nt,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("weights"), pybind11::arg("weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_grouped_gemm_nn", &grouped_fp8_block128_gemm_nn,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("weights"), pybind11::arg("weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_grouped_w13_gemm_nt_canonical",
          &grouped_fp8_block128_w13_gemm_nt_canonical,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("canonical_weights"), pybind11::arg("canonical_weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_grouped_gemm_nt_expanded",
          &grouped_fp8_block128_gemm_nt_expanded,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("weights"), pybind11::arg("weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_grouped_gemm_nn_expanded",
          &grouped_fp8_block128_gemm_nn_expanded,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("weights"), pybind11::arg("weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_grouped_w13_gemm_nt_canonical_expanded",
          &grouped_fp8_block128_w13_gemm_nt_canonical_expanded,
          pybind11::arg("activations"), pybind11::arg("activation_scales"),
          pybind11::arg("canonical_weights"), pybind11::arg("canonical_weight_scales"),
          pybind11::arg("group_counts"));
    m.def("sm103_fp8_block128_swiglu_quantize", &swiglu_quantize,
          pybind11::arg("preactivation"));
    m.def("sm103_fp8_block128_swiglu_backward", &swiglu_backward,
          pybind11::arg("grad_output"), pybind11::arg("preactivation"));
    m.def("sm103_fp8_block128_swiglu_backward_canonical", &swiglu_backward_canonical,
          pybind11::arg("grad_output"), pybind11::arg("preactivation"));
    m.def("sm103_fp8_block128_route_scale_quantize", &route_scale_quantize,
          pybind11::arg("grad_output"), pybind11::arg("route_scores"), pybind11::arg("route_order"));
    m.def("sm103_fp8_block128_post_down_combine", &post_down_combine,
          pybind11::arg("route_output"), pybind11::arg("route_scores"));
    m.def("sm103_fp8_block128_post_down_score_grad", &post_down_score_grad,
          pybind11::arg("route_output"), pybind11::arg("grad_output"), pybind11::arg("topk"));
    m.def("sm103_fp8_block128_route_sum", &route_sum,
          pybind11::arg("route_grad"), pybind11::arg("num_tokens"), pybind11::arg("topk"));
    m.def("sm103_fp8_block128_expanded_post_down_scale", &expanded_post_down_scale,
          pybind11::arg("route_output"), pybind11::arg("route_scores"));
    m.def("sm103_fp8_block128_expand_compact_routes", &expand_compact_routes,
          pybind11::arg("compact"), pybind11::arg("routes"), pybind11::arg("expanded_rows"));
    m.def("sm103_fp8_block128_collapse_expanded_routes", &collapse_expanded_routes,
          pybind11::arg("expanded"), pybind11::arg("routes"));
    m.def("sm103_fp8_block128_expanded_post_down_score_grad",
          &expanded_post_down_score_grad,
          pybind11::arg("route_output"), pybind11::arg("grad_output"), pybind11::arg("routes"));
    m.def("sm103_fp8_block128_expanded_route_scale_quantize",
          &expanded_route_scale_quantize,
          pybind11::arg("grad_output"), pybind11::arg("route_scores"));
    m.def("sm103_fp8_block128_grouped_bf16_wgrad",
          &grouped_bf16_wgrad,
          pybind11::arg("left"), pybind11::arg("right"),
          pybind11::arg("padded_group_counts"));
    m.def("sm103_fp8_block128_grouped_bf16_wgrad_expanded",
          &grouped_bf16_wgrad_expanded,
          pybind11::arg("left"), pybind11::arg("right"), pybind11::arg("psum"));
}

}  // namespace deep_gemm::sm103_fp8_block128
