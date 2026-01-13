#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, typename scalar_t>
__global__ void transpose_kernel(
    scalar_t *output,       // shape[N, M]
    const scalar_t *input,  // shape[M, N]
    const int M,
    const int N
) {
    int offset_m = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_n = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ scalar_t s_buf[BLOCK_SIZE_M][BLOCK_SIZE_N];

    s_buf[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[offset_m * N + offset_n];
    __syncthreads();

    int out_offset_m = blockIdx.x * blockDim.x + threadIdx.y;   // 仅通过block下标转置block，block内部通过shared mem转置
    int out_offset_n = blockIdx.y * blockDim.y + threadIdx.x;
    output[out_offset_m * M + out_offset_n] = s_buf[threadIdx.x][threadIdx.x ^ threadIdx.y]; // 取swizzling前的列元素

}

void transpose(
    torch::Tensor &output,  // shape[N, M]
    torch::Tensor &input    // shape[M, N]
){
    // std::cout << "There is in launch kernel." << std::endl;

    // TODO 确保为行主序
    const int M = input.size(-2);
    const int N = input.size(-1);

    // TODO 处理边界
    constexpr int BLOCK_SIZE_M = 32;
    constexpr int BLOCK_SIZE_N = 32;
    dim3 grid(int(N/BLOCK_SIZE_N), int(M/BLOCK_SIZE_M)); // 注意: x->N, y->M
    dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M); // 每个thread处理1个元素

    using input_dtype = float;
    transpose_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N><<<grid, block>>>(
        output.data_ptr<input_dtype>(), 
        input.data_ptr<input_dtype>(), 
        M, N
    );
}