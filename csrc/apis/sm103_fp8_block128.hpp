#pragma once

#include <pybind11/pybind11.h>

namespace deep_gemm::sm103_fp8_block128 {

void register_apis(pybind11::module_& m);

}  // namespace deep_gemm::sm103_fp8_block128
