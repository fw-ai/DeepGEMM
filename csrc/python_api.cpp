#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega.hpp"
#include "apis/runtime.hpp"
#include "apis/sm103_fp8_block128.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::runtime::register_apis(m);
    deep_gemm::sm103_fp8_block128::register_apis(m);

#define DG_STRINGIFY_IMPL(value) #value
#define DG_STRINGIFY(value) DG_STRINGIFY_IMPL(value)
#ifndef DEEP_GEMM_GIT_COMMIT_TOKEN
#define DEEP_GEMM_GIT_COMMIT_TOKEN unknown
#endif
    m.attr("__git_commit__") = DG_STRINGIFY(DEEP_GEMM_GIT_COMMIT_TOKEN);
}
