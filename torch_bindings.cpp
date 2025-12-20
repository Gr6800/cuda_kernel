#include <torch/extension.h> // 包含PYBIND11_MODULE
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "rmsnorm cuda implementation");
}