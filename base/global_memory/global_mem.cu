#include <iostream>
#include <torch/all.h>

template <int vec_size, typename scalar_t>
struct __align__(vec_size*sizeof(scalar_t)) vec_n_buf{
    scalar_t val[vec_size];
};

template <int vec_size, typename scalar_t>
__global__ void global_mem_kernel(
    scalar_t *output,
    const scalar_t *input,
    const scalar_t *weight
){
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    auto vec_op = [](
        vec_n_buf<vec_size, scalar_t> &vec_output,
        const vec_n_buf<vec_size, scalar_t> &vec_input,
        const vec_n_buf<vec_size, scalar_t> &vec_weight
    ){
        vec_n_buf<vec_size, scalar_t> tmp_output;
        vec_n_buf<vec_size, scalar_t> tmp_input;
        vec_n_buf<vec_size, scalar_t> tmp_weight;
        
        tmp_input = vec_input;
        tmp_weight = vec_weight;
#pragma unroll
        for(int i = 0; i < vec_size; i++){
            tmp_output.val[i] = tmp_input.val[i] * tmp_weight.val[i];
        }
        vec_output = tmp_output;
    };

    using vec_n = vec_n_buf<vec_size, scalar_t>;
    vec_n* vec_output = reinterpret_cast<vec_n*>(output);
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(input);
    const vec_n* vec_weight = reinterpret_cast<const vec_n*>(weight);

    vec_op(
        vec_output[offset],
        vec_input[offset],
        vec_weight[threadIdx.x]
    );
}

void global_mem(
    torch::Tensor &output,
    torch::Tensor &input,
    torch::Tensor &weight
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
        input.data_ptr<input_dtype>(),
        weight.data_ptr<input_dtype>()
    );
}