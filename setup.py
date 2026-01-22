from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

compile_kernel_list = [
    # base
    "shared_mem",
    "transpose",
    "tma",
    "global_mem",

    # transformer
    "rmsnorm",
]

def get_source(compile_kernel_list):
    source_list = ["torch_bindings.cpp"]
    if "rmsnorm" in compile_kernel_list:
        source_list.append("rmsnorm/rmsnorm.cu")
    if "shared_mem" in compile_kernel_list:
        source_list.append("base/shared_memory/shared_mem.cu")
    if "transpose" in compile_kernel_list:
        source_list.append("base/shared_memory/transpose.cu")
    if "tma" in compile_kernel_list:
        source_list.append("base/shared_memory/tma.cu")
    if "global_mem" in compile_kernel_list:
        source_list.append("base/global_memory/global_mem.cu")

    return source_list

cuda_lib_path = "/usr/local/cuda/lib64"

setup(
    ext_modules=[
        CUDAExtension(
            name="cuda_ops",
            sources=get_source(compile_kernel_list), # 编译的源文件
            define_macros=[("HOST_CUDA_ARCH", 900),], # 编译架构: hopper
            extra_compile_args={
                'nvcc': [
                    "-std=c++17",
                    "-Xcompiler",
                    "-fPIC",
                    "-gencode=arch=compute_90a,code=sm_90a",
                    "-DNDEBUG",
                ],
            }, # 编译参数
            include_dirs=[ # 头文件目录
                os.getcwd(), # 当前项目路径
            ],
            libraries=["cuda"], # 拓展库
            library_dirs=[cuda_lib_path], # 拓展库的搜索路径
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)