#include <iostream> // std::cout
#include <ATen/cuda/CUDAContext.h> // cudaStream_t
#include <torch/all.h> // torch::Tensor

// kernel
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

template <typename scalar_t>
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
    // printf("there is rmsnorm kernel\n");
    // 计算下标分配数据 load
    int offset_start_s = blockIdx.x;
    int offset_start_h = threadIdx.x;
    
    float x;
    extern __shared__ float s_variance[];   // 在kernel launch时指定了shared mem的大小，此处命名
    float variance = 0.0;
    // 计算+=x^2: 不论有多少数据都处理到最多1024个thread中
    for(offset_start_h = threadIdx.x; offset_start_h < hidden_dim; offset_start_h += blockDim.x){
        int offset = offset_start_s * stride_s + offset_start_h;
        x = input[offset];
        variance += x*x;
    }
    // __syncthreads();    // 每个线程处理的数据可能不一致，这里需要做__syncthreads吗？TODO 目前不同步，在hiddendim>1024下计算正常
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
        int num_warp = min(1024, (hidden_dim + 32 - 1) / 32);
        float sum = (lane_id < num_warp)? s_variance[lane_id] : 0;
        s_variance[0] = reduce_in_warp(sum);    // 将结果存在s_variance[0]中
    }
    __syncthreads();

    __shared__ float s_var;
    s_var = s_variance[0];
    // printf("s_var: %f\n", s_var);

    // 计算rmsnorm, store
    for(offset_start_h = threadIdx.x; offset_start_h < hidden_dim; offset_start_h += blockDim.x){
        int offset = offset_start_s * stride_s + offset_start_h;
        x = input[offset]; // TODO 当hidden_dim<1024时，只用load一次x
        output[offset] = x * weight[offset_start_h%hidden_dim] * rsqrt(s_var/hidden_dim + eps);
    }
    // printf("offset: %d, weight: %f\n", offset, weight[offset_start_h]);

}

// 接口 && launch
void rmsnorm(
    torch::Tensor& output,  // [..., hidden_dim]
    torch::Tensor& input,   // [..., hidden_dim]
    torch::Tensor& weight,  // [hidden_dim]
    double eps
) {
    // std::cout << "there is rmsnorm launch" << std::endl;

    // TODO: 确保input为行主序

    // reshape -> ndim=2 shape[Batch, Hidden_dim]
    int hidden_dim = input.size(-1);
    torch::Tensor input_view = input.view({-1, hidden_dim});
    int seq_len = input.size(0);
    int stride_s = input.stride(0);
    int stride_h = input.stride(1);

    // std::cout << "seq_len: " << seq_len << std::endl;
    // std::cout << "hidden_dim: " << hidden_dim << std::endl;
    // std::cout << "stride_s: " << stride_s << std::endl;
    // std::cout << "stride_h: " << stride_h << std::endl;

    // 划分数据
    dim3 grid(seq_len); //按行划分,每个block处理一行,行与行之间没有数据交互
    dim3 block(std::min(hidden_dim, 1024)); // 每个thread处理一个元素
    int num_warp = std::min(32, (hidden_dim + 32 - 1) / 32);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    using input_dtype = float;  // TODO 暂时写成float
    rmsnorm_kernel<input_dtype><<<grid, block, num_warp*sizeof(float), stream>>>(
        output.data_ptr<input_dtype>(),    // TODO: 应该传指针就行吧?
        input_view.data_ptr<input_dtype>(),
        weight.data_ptr<input_dtype>(),
        eps,
        seq_len, hidden_dim,
        stride_s, stride_h
    );
}