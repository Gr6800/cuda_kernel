#include <iostream> // std::cout
#include <ATen/cuda/CUDAContext.h> // cudaStream_t
#include <torch/all.h> // torch::Tensor
#include <cuda/barrier>

// kernel
template <typename scalar_t, int size> // TODO 后续可以在这里改成2维
struct __align__(size*sizeof(scalar_t)) vec_n_buf{   // __align要求1,2,4,8,16B对齐
    scalar_t val[size];
};

// 将block_size_s行的+=x*x都累加到thread中的寄存器
template <int block_size_s, int vec_size, typename scalar_t, typename VecOp>   // 类型模板参数可以由编译器推导，非类型的模型参数不能被推导
__device__ inline void vec_xx(
        const scalar_t* input, // 每个block的全局下标的地址
        float* s_input, // size: [block_size_s*blockDim.x*vec_size] TODO 支持hidden_dim=inf，此时buf有限，考虑优先按列计算应该更能利用shared mem
        const int hidden_dim,
        VecOp vec_op){
    // step 1: TMA: global mem -> shared mem
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;

    if(threadIdx.x == 0){
        init(&bar, blockDim.x);
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    if(threadIdx.x == 0){
        cuda::memcpy_async(
            s_input,
            input,
            cuda::aligned_size_t<16>(block_size_s*blockDim.x*vec_size*sizeof(scalar_t)),
            bar
        );
    }

    barrier::arrival_token token = bar.arrive();    // TODO: arrive和wait之间应该可以做点别的
    bar.wait(std::move(token));
    // TODO 存在长尾效应时，改用LDG.128

    // step 2: 计算每个thread中的+=x*x

    // 1 确认对齐
    static_assert(vec_size > 0 && (vec_size & (vec_size - 1)) == 0, 
                  "vec_size must be a power of 2");
    // uintptr_t addr_input = reinterpret_cast<uintptr_t>(input);
    // constexpr int vec_width = vec_size * sizeof(scalar_t); // 需要在编译期确认，因为scalar_t在编译期确定，所以需要定义为编译期常量。在函数外保证了input的数据类型与vec中的数据类型相同
    // bool is_align = ((addr_input & (vec_width - 1)) == 0);

    // 2 为每个线程分配数据
    using vec_n = vec_n_buf<scalar_t, vec_size>;    // 使用数据类型转换直接将1xf32变为sizexf32
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(s_input); // reinterpret_cast不能去除const属性，要么统一都是const，要么都不是

    // 3 应用vec_op
    int num_vec = hidden_dim/vec_size; // 每行
    for(int ss = 0; ss < block_size_s; ss++) {
        for(int offset_start_h = threadIdx.x; offset_start_h < num_vec; offset_start_h += blockDim.x){  // TODO TMA buf修改后此处也需要修改
            vec_op(vec_input[ss*num_vec + offset_start_h], ss);  // 后续需要处理边界 TODO 后续尝试采用double buffer
        }
    }
}

// 完成block_size_s行的rmsnorm的计算，并存放到output
// 一个block负责block_size_s行数据，行与行之间串行
template <int block_size_s, int vec_size, typename scalar_t, typename VecRmsnormOp>
__device__ void vec_rmsnorm(
    scalar_t* output,
    const scalar_t* input,
    const scalar_t* weight,
    float *s_var,
    const int hidden_dim,
    const float eps,
    VecRmsnormOp vec_rmsnorm_op
){
    // pack
    using vec_n = vec_n_buf<scalar_t, vec_size>;
    vec_n* vec_output = reinterpret_cast<vec_n*>(output); // 将当前block负责的所有元素都进行转换
    const vec_n* vec_input = reinterpret_cast<const vec_n*>(input);
    const vec_n* vec_weight = reinterpret_cast<const vec_n*>(weight);

    // 为每个thread分配数据
    int num_vec = hidden_dim/vec_size; // 每行
    for(int ss = 0; ss < block_size_s; ss++) {
        for(int offset_start_h = threadIdx.x; offset_start_h < num_vec; offset_start_h += blockDim.x) {
            vec_rmsnorm_op(
                vec_output[ss*num_vec + offset_start_h], 
                vec_input[ss*num_vec + offset_start_h], // TODO 当需要多次使用tma时，此处也需要修改
                vec_weight[offset_start_h], // TODO 考虑竖着计算，共用weight
                s_var, 
                hidden_dim, eps, ss
            );
        }
    }
}

__device__ float reduce_in_warp(float val) {
    for(int offset = 16; offset > 0; offset /= 2) {
        // 传递变量val,向下(lane0方向)偏移offset个线程数
        // 以lane16(__shfl_down_sync中的val)为视角,lane16.val会被lane0.val获取
        // 所以__shfl_down_sync()返回的是高lane的val
        // 对于lane0来说,__shfl_down_sync()获得了lane16的val
        // 对于lane16来说，不存在lane32，获得的仍是lane16本身
        // mask不知道有什么用，不论怎么设置32个thread都参与
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int block_size_s, int vec_size>
__device__ void reduce_in_block(
    float* variance, 
    float* s_variance,
    const int hidden_dim,
    int num_warp
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    for(int ss = 0; ss < block_size_s; ss++) {
        // 1 在warp内归约
        float var = variance[ss];
        var = reduce_in_warp(var);
        // 2 将lane0的值写到shared memory
        if(lane_id == 0){
            s_variance[ss*num_warp + warp_id] = var;
        }
        __syncthreads();
        // 3 在shared memory上归约
        if(warp_id == 0) {
            float sum = (lane_id < num_warp)? s_variance[ss*num_warp + lane_id] : 0;
            s_variance[ss*num_warp + 0] = reduce_in_warp(sum);    // 将结果存在s_variance[0]中
        }
        __syncthreads();  
    }
}

template <typename scalar_t, int block_size_s, int vec_size>
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
    int offset_start_s = blockIdx.x * block_size_s;
    
    float variance[block_size_s];   // size: [block_size_s]，每个thread负责block_size_s*vec，每行1个var buf
    for(int i = 0; i < block_size_s; i++) variance[i] = 0.0; // TODO 优化此处
    extern __shared__ float shared_mem[];   // size_0: [block_size_s*blockIdx.x*vec_size]，存放input
                                            // size_1: [block_size_s*num_warp]，每个block负责block_size_s*hidden_dim，共block_size_s*num_warp个warp，每个warp一个s_var buf做归约
    // int offset_start_s_input = 0;    // offset in shared mem for input
    int offset_start_s_var = block_size_s*blockDim.x*vec_size;  // offset in shared mem for reducing var

    // step 2: 计算+=x*x，并归约
    // step 2.1: 向量化访存，先归约至每个线程的variance中
    auto vec_op = [&variance](const vec_n_buf<scalar_t, vec_size> &vec, int ss){
        // 完成一个vec中的+=x*x计算
#pragma unroll // 循环展开 
        for(int i = 0; i < vec_size; i++){
            float x = static_cast<float>(vec.val[i]);
            variance[ss] += x * x;
        }
    };
    vec_xx<block_size_s, vec_size>(
        input+offset_start_s*stride_s, 
        &shared_mem[0], // offset_start_s_input
        hidden_dim, 
        vec_op
    );

    int num_warp = min(1024/warpSize, (int(hidden_dim/vec_size) + warpSize - 1) / warpSize);   // num_thread最大为1024，所以warp个数最大为32(32=1024/32)
    reduce_in_block<block_size_s, vec_size>(variance, &shared_mem[offset_start_s_var], hidden_dim, num_warp);

    __shared__ float s_var[block_size_s];   // size: [block_size_s]，每个block负责block_size_s*hidden_dim，每行一个s_var buf
    float * s_variance = &shared_mem[offset_start_s_var];
    for(int i = 0; i < block_size_s; i++) s_var[i] = s_variance[i*num_warp];    // TODO 优化此处

    // step 3: 计算rmsnorm
    auto vec_rmsnorm_op = [](
        vec_n_buf<scalar_t, vec_size> &vec_output,
        const vec_n_buf<scalar_t, vec_size> &vec_input,
        const vec_n_buf<scalar_t, vec_size> &vec_weight,
        float *s_var, 
        const int hidden_dim, 
        const float eps,
        int ss
    ) {
        // 完成一个vec中的rms计算
        vec_n_buf<scalar_t, vec_size> tmp_output;
        vec_n_buf<scalar_t, vec_size> tmp_input;
        vec_n_buf<scalar_t, vec_size> tmp_weight;
        tmp_input = vec_input;
        tmp_weight = vec_weight;
#pragma unroll
        for(int i = 0; i < vec_size; i++){
            tmp_output.val[i] = tmp_input.val[i] * tmp_weight.val[i] * rsqrt(s_var[ss]/hidden_dim + eps);
        }
        vec_output = tmp_output;
    };

    vec_rmsnorm<block_size_s, vec_size>(
        output + offset_start_s*stride_s,
        &shared_mem[0],
        weight, // TODO 尝试共用，避免多次load
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
    constexpr int vec_size = 4; // 模板参数的变量应该是编译期常量
    constexpr int block_size_s = 2; // 为4时，可能L1 cache显存爆了
    dim3 grid(int(seq_len/block_size_s)); //按行划分,每个block处理block_size_s行，行与行之间在thread中串行
    dim3 block(std::min(int(hidden_dim/vec_size), 1024)); 
    int num_warp = std::min(32, (int(hidden_dim/vec_size) + 32 - 1) / 32);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    using input_dtype = float;  // TODO 暂时写成float
    rmsnorm_kernel<input_dtype, block_size_s, vec_size><<<grid, block, 
                    (block_size_s*block.x*vec_size + block_size_s*num_warp)*sizeof(float), stream>>>(
        output.data_ptr<input_dtype>(),
        input_view.data_ptr<input_dtype>(),
        weight.data_ptr<input_dtype>(),
        eps,
        seq_len, hidden_dim,
        stride_s, stride_h
    );
}