#include <iostream>
#include <torch/all.h>

template <int vec_size, typename scalar_t>
struct __align__(vec_size*sizeof(scalar_t)) vec_n_buf{
    scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
__global__ void global_mem_kernel(
    scalar_t *output,
    const scalar_t *input
){
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    using vec_n = vec_n_buf<vec_size, scalar_t>;
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(input);
    vec_n* vec_output = reinterpret_cast<vec_n*>(output);

    vec_output[offset] = vec_input[offset];
}

void global_mem(
    torch::Tensor &output,
    torch::Tensor &input
){  
    // TODO 保证input为行主序
    int M = input.size(-2);
    int N = input.size(-1);

    constexpr int vec_size = 4;
    int num_vec = N / vec_size;
    dim3 grid(M);
    dim3 block(num_vec);
    using input_dtype = float;
    global_mem_kernel<vec_size><<<grid, block>>>(
        output.data_ptr<input_dtype>(),
        input.data_ptr<input_dtype>()
    );
}