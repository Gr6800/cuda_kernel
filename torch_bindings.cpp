#include <torch/extension.h> // 包含PYBIND11_MODULE
#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // base
    m.def("shared_mem", &shared_mem, "shared mem kernel");
    m.def("transpose", &transpose, "transpose kernel");
    m.def("global_mem", &global_mem, "global mem kernel");

    // transformer
    m.def("rmsnorm", &rmsnorm, "rmsnorm cuda implementation");
}