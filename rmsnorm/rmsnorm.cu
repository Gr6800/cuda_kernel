#include <iostream> // std::cout
#include <ATen/cuda/CUDAContext.h> // cudaStream_t

// kernel
__global__ void rmsnorm_kernel() {
    printf("there is rmsnorm kernel\n");
}

// 接口 && launch
void rmsnorm() {
    std::cout << "there is rmsnorm launch" << std::endl;
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_kernel<<<1, 1, 0, stream>>>(); // gridDim, blockDim, sharedMemSize, stream
    // rmsnorm_kernel<<<1, 1>>>(); // gridDim, blockDim, sharedMemSize, stream
}