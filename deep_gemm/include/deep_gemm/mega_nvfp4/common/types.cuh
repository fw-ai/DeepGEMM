#pragma once

// Derived from sgl-project/DeepGEMM@ec226bcf400c11d62755137b916fae9ee359c684:
//   deep_gemm/include/deep_gemm/common/types.cuh
// Copyright (c) 2025 DeepSeek, with modifications by the sgl-project/DeepGEMM contributors.
// Licensed under the MIT License (see LICENSE at the repository root).
// Fireworks port: only the `MmaKind` enum and its helpers (`get_element_size`,
// `get_element_bits`, `get_sf_gran_k`) are copied, in namespace `deep_gemm::mega_nvfp4`; they are unchanged.


#include <cute/arch/mma_sm100_desc.hpp>

namespace deep_gemm::mega_nvfp4 {

enum class MmaKind {
    BF16        = 0,
    MXFP8FP4    = 1,
    MXFP4       = 2,
    MXF4        = MXFP4,
    NVFP4       = 3,
};

constexpr CUTLASS_HOST_DEVICE int get_element_size(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 2;
        case MmaKind::MXFP8FP4: return 1;
        case MmaKind::MXFP4:    return 1;
        case MmaKind::NVFP4:    return 1;
        default: return 0;
    }
}

constexpr CUTLASS_HOST_DEVICE int get_element_bits(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return 16;
        case MmaKind::MXFP8FP4: return 8;
        case MmaKind::MXFP4:    return 4;
        case MmaKind::NVFP4:    return 4;
        default: return 0;
    }
}

constexpr CUTLASS_HOST_DEVICE int get_sf_gran_k(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::MXFP8FP4: return 32;
        case MmaKind::MXFP4:    return 32;
        case MmaKind::NVFP4:    return 16;
        default: return 0;
    }
}

} // namespace deep_gemm::mega_nvfp4
