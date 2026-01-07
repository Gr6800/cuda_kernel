#include <iostream> // std::cout
#include <ATen/cuda/CUDAContext.h> // cudaStream_t
#include <torch/all.h> // torch::Tensor

// kernel
template <typename scalar_t, int size> // TODO 后续可以在这里改成2维
struct __align__(size*sizeof(scalar_t)) vec_n_buf{   // __align要求1,2,4,8,16B对齐
    scalar_t val[size];
};

template <int vec_size, typename scalar_t, typename VecOp>   // 类型模板参数可以由编译器推导，非类型的模型参数不能被推导
__device__ inline void vec_xx(
        const scalar_t* input, // 每行的全局下标的地址
        VecOp vec_op){
    // 每个thread处理vec个元素，目前每个thread仅处理1轮 TODO

    // 1 确认对齐
    static_assert(vec_size > 0 && (vec_size & (vec_size - 1)) == 0, 
                  "vec_size must be a power of 2");
    // uintptr_t addr_input = reinterpret_cast<uintptr_t>(input);
    // constexpr int vec_width = vec_size * sizeof(scalar_t); // 需要在编译期确认，因为scalar_t在编译期确定，所以需要定义为编译期常量。在函数外保证了input的数据类型与vec中的数据类型相同
    // bool is_align = ((addr_input & (vec_width - 1)) == 0);
    // printf("is_align: %d\n", is_align);

    // 2 为每个线程分配数据
    using vec_n = vec_n_buf<scalar_t, vec_size>;    // 使用数据类型转换直接将1xf32变为sizexf32
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(input); // reinterpret_cast不能去除const属性，要么统一都是const，要么都不是
    int vec_offset = threadIdx.x; // 后续需要处理边界 TODO //申请dim3 block时已经考虑过为每个thread分配vec_size个数据
    vec_n data = vec_input[vec_offset];

    // 3 应用vec_op
    vec_op(data); // 数据类型转换后，可以通过下标来取数，一次取size个
}

template <int vec_size, typename scalar_t, typename VecRmsnormOp>
__device__ void vec_rmsnorm(
    scalar_t* output,
    const scalar_t* input,
    const scalar_t* weight,
    float s_var,
    const int hidden_dim,
    const float eps,
    VecRmsnormOp vec_rmsnorm_op
){
    // 按一行数据一个block负责，然后重新分配给每个thread
    using vec_n = vec_n_buf<scalar_t, vec_size>;
    vec_n* vec_output = reinterpret_cast<vec_n*>(output); // 将当前block负责的所有元素都进行转换
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(input);
    const vec_n* vec_weight = reinterpret_cast<const vec_n*>(weight);
    int offset_start_h = threadIdx.x;
    vec_rmsnorm_op(
        vec_output[offset_start_h], 
        vec_input[offset_start_h], 
        vec_weight[offset_start_h], 
        s_var, hidden_dim, eps);
}

__device__ float reduce_in_warp(float val) {
    // printf("init in warp, offset: %d, val: %f\n", blockIdx.x*32+threadIdx.x, val);
    // int offset = 1;
    // float n = __shfl_down_sync(0x000000ff, val, offset);
    // printf("offset in warp: %d, offset: %d, n: %f\n", offset, blockIdx.x*32+threadIdx.x, n);
    for(int offset = 16; offset > 0; offset /= 2) {
        // 传递变量val,向下(lane0方向)偏移offset个线程数.例如,lane0.val = lane0.val + lane16.val // TODO: 这理解对吗
        // 以lane16(__shfl_down_sync中的val)为视角,lane16.val会被lane0.val获取
        // 所以__shfl_down_sync()返回的是高lane的val
        // 对于lane0来说,__shfl_down_sync()获得了lane16的val
        // 对于lane16来说，不存在lane32，获得的仍是lane16本身
        // mask不知道有什么用，不论怎么设置32个thread都参与
        val += __shfl_down_sync(0xffffffff, val, offset);
        // printf("offset in warp: %d, offset: %d, val: %f\n", offset, blockIdx.x*32+threadIdx.x, val);
    }
    return val;
}

template <typename scalar_t, int vec_size>
__global__ void rmsnorm_kernel(
    scalar_t* output,       // [seq_len, hidden_dim], __restrict__可以提示编译器该指针不与其他指针重叠,有助于进行优化
    const scalar_t* input,  // [seq_len, hidden_dim], const表示输入都只可读,不可修改数值
    const scalar_t* weight, // [hidden_dim]
    const float eps,
    const int seq_len,
    const int hidden_dim,
    const int stride_s, 
    const int stride_h
) {
    // step 1: 计算下标分配数据 load
    int offset_start_s = blockIdx.x;
    int offset_start_h = threadIdx.x;
    
    extern __shared__ float s_variance[];   // 在kernel launch时指定了shared mem的大小，此处命名
    float variance = 0.0;
    
    // step 2: 计算+=x*x，并归约
    // step 2.1: 向量化访存，先归约至每个线程的variance中
    auto vec_op = [&variance](const vec_n_buf<scalar_t, vec_size> &vec){
#pragma unroll // 循环展开 
        for(int i = 0; i < vec_size; i++){
            float x = static_cast<float>(vec.val[i]);
            variance += x * x;
        }
    };
    vec_xx<vec_size>(input+offset_start_s*stride_s, vec_op);
    // printf("offset: %d, variance: %f\n", offset_start_s*stride_s+offset_start_h, variance);
    
    // step 2.2: 归约
    // 1 在warp内归约
    variance = reduce_in_warp(variance);
    // 2 将lane0的值写到shared memory
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if(lane_id == 0){
        s_variance[warp_id] = variance;
    }
    __syncthreads();
    // 3 在shared memory上归约
    if(warp_id == 0) {  // num_thread最大为1024，所以warp个数最大为32(32=1024/32)
        int num_warp = min(32, (int(hidden_dim/vec_size) + 32 - 1) / 32);
        float sum = (lane_id < num_warp)? s_variance[lane_id] : 0;
        s_variance[0] = reduce_in_warp(sum);    // 将结果存在s_variance[0]中
    }
    __syncthreads();

    __shared__ float s_var;
    s_var = s_variance[0];
    // printf("s_var: %f\n", s_var);

    // step 3: 计算rmsnorm
    auto vec_rmsnorm_op = [](
        vec_n_buf<scalar_t, vec_size> &vec_output,
        const vec_n_buf<scalar_t, vec_size> &vec_input,
        const vec_n_buf<scalar_t, vec_size> &vec_weight,
        float s_var, 
        const int hidden_dim, 
        const float eps
    ) {
#pragma unroll
        for(int i = 0; i < vec_size; i++){
            vec_output.val[i] = vec_input.val[i] * vec_weight.val[i] * rsqrt(s_var/hidden_dim + eps);
        }
    };

    vec_rmsnorm<vec_size>(
        output + offset_start_s*stride_s,
        input + offset_start_s*stride_s,
        weight,
        s_var, hidden_dim, eps,
        vec_rmsnorm_op
    );
}

// 接口 && launch
void rmsnorm(
    torch::Tensor& output,  // [..., hidden_dim]
    torch::Tensor& input,   // [..., hidden_dim]
    torch::Tensor& weight,  // [hidden_dim]
    double eps
) {

    // TODO: 确保input为行主序

    // reshape -> ndim=2 shape[Batch, Hidden_dim]
    int hidden_dim = input.size(-1);
    torch::Tensor input_view = input.view({-1, hidden_dim});
    int seq_len = input.size(0);
    int stride_s = input.stride(0);
    int stride_h = input.stride(1);

    // 划分数据
    constexpr int vec_size = 2; // 模板参数的变量应该是编译期常量
    dim3 grid(seq_len); //按行划分,每个block处理一行,行与行之间没有数据交互
    dim3 block(std::min(int(hidden_dim/vec_size), 1024)); // TODO 目前在+=x^2阶段，每个线程处理多个数据，存在部分thread处理0个数据的情况
    int num_warp = std::min(32, (int(hidden_dim/vec_size) + 32 - 1) / 32);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    using input_dtype = float;  // TODO 暂时写成float
    rmsnorm_kernel<input_dtype, vec_size><<<grid, block, num_warp*sizeof(float), stream>>>(
        output.data_ptr<input_dtype>(),    // TODO: 应该传指针就行吧?
        input_view.data_ptr<input_dtype>(),
        weight.data_ptr<input_dtype>(),
        eps,
        seq_len, hidden_dim,
        stride_s, stride_h
    );
}