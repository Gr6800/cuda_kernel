#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

template <typename scalar_t>
__global__ void shared_mem_kernel(
    scalar_t* output, 
    const scalar_t* input,
    int stride_s, 
    int stride_h
){
    // printf("there is in cuda kernel\n");
    
    extern __shared__ scalar_t mem_s[];
    
    int test_size = 32;
    int shared_offset_m = int(threadIdx.x / test_size);

    mem_s[shared_offset_m*warpSize + threadIdx.x%test_size] = threadIdx.x;
    __syncthreads();

    output[shared_offset_m*stride_s + threadIdx.x%test_size] = mem_s[shared_offset_m*warpSize + threadIdx.x%test_size];
}

void shared_mem(
    torch::Tensor &output,  // [seq_len, hidden_dim]
    torch::Tensor &input    // [seq_len, hidden_dim]
){
    // std::cout << "there is in launch kernel" << std::endl;
    int hidden_dim = input.size(-1);
    int seq_len = input.size(-2);
    int stride_s = hidden_dim;
    int stride_h = 1;

    // std::cout << "seq_len: " << seq_len << std::endl;
    // std::cout << "hidden_dim: " << hidden_dim << std::endl;

    dim3 grid(seq_len);
    dim3 block(hidden_dim);
    int shared_size = 32*4;

    using input_dtype = float;
    shared_mem_kernel<<<grid, block, shared_size*sizeof(input_dtype)>>>(
        output.data_ptr<input_dtype>(),
        input.data_ptr<input_dtype>(),
        stride_s, stride_h
    );
}