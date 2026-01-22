#include <iostream>
#include <torch/all.h>
#include <cuda/barrier>

template <int shared_size, typename scalar_t>
__global__ void tma_kernel(
    scalar_t *output,
    scalar_t *input, // [M, N]
    const int M, 
    const int N
){
    using barrier = cuda::barrier<cuda::thread_scope_block>;

    __shared__ __align__(16) float smem_data[shared_size];  // TMA的使用要求目标地址和源地址都16B对齐，并且size是16的倍数，否则会退化
    
    // 1. a) 用0号线程初始化 barrier，与上面的代码示例类似。
    //    b) 插入一个fence。表示后续执行异步拷贝操作，需要在这个fence之后才执行。
    __shared__ barrier bar;
    if (threadIdx.x == 0) { 
        init(&bar, blockDim.x);                                    // a)
        cuda::device::experimental::fence_proxy_async_shared_cta();// b)
    }
    __syncthreads();

    // 2. 发起 TMA 异步拷贝。注意：TMA 操作是用单线程发起。
    if (threadIdx.x == 0) {
        // 3a. 发起异步拷贝
        // Version 1: cuda::memcpy_async
        cuda::memcpy_async(
            smem_data, 
            &input[blockIdx.x * N], 
            cuda::aligned_size_t<16>(sizeof(smem_data)),
            bar
        );

        // // Version 2: cuda::device::memcpy_async_tx
        // cuda::device::memcpy_async_tx(
        //   smem_data, &input[blockIdx.x * N], 
        //   cuda::aligned_size_t<16>(sizeof(smem_data)),
        //   bar);
        // cuda::device::barrier_expect_tx(bar, sizeof(smem_data));

        // // Version 3: cuda::device::experimental::cp_async_bulk_global_to_shared
        // cuda::device::experimental::cp_async_bulk_global_to_shared(
        //     smem_data, &input[blockIdx.x * N], cuda::aligned_size_t<16>(sizeof(smem_data)), bar);
        // cuda::device::barrier_expect_tx(bar, sizeof(smem_data));
    }
    // 3b. 所有线程到达该标记点，barrier内部的计数器会加 1。
    barrier::arrival_token token = bar.arrive();
    // 3c.等待barrier内部的计数器等于期望数值，即所有线程到达3b点时，当前线程的wait会返回，结束等待。
    bar.wait(std::move(token));

    // 4. 在 Shared Memory 上写数据。
    smem_data[threadIdx.x] += 1;

    // 5. 插入fence，保证后续的异步拷贝操作在Shared Memory写数据结束后再启动。
    cuda::device::experimental::fence_proxy_async_shared_cta();
    __syncthreads();

    // 6. 发起从 Shared Memory 到 Global Memory 的异步拷贝操作。
    if(threadIdx.x == 0) {
        // assert(((uintptr_t)smem_data % 16) == 0);
        // assert(((uintptr_t)&output[blockIdx.x * N] % 16) == 0);
        // assert((sizeof(smem_data) % 16) == 0);
        // cuda::device::experimental::cp_async_bulk_shared_to_global(
        //     &output[blockIdx.x * N], smem_data, cuda::aligned_size_t<16>(sizeof(smem_data)));
        cuda::device::experimental::cp_async_bulk_shared_to_global(
            &output[blockIdx.x * N], smem_data, cuda::aligned_size_t<16>(sizeof(smem_data)));
        // 7. 等待tma传输完成
        cuda::device::experimental::cp_async_bulk_commit_group();   // 创建bulk async-group
        cuda::device::experimental::cp_async_bulk_wait_group_read<0>(); // 等待传输完成
    }

}

void tma(
    torch::Tensor &output,
    torch::Tensor &input
){
    const int M = input.size(-2);
    const int N = input.size(-1);

    dim3 gird(M);
    dim3 block(N);

    using input_dtype = float;
    constexpr int shared_size = 1024;
    tma_kernel<shared_size><<<gird, block>>>(
        output.data_ptr<input_dtype>(), 
        input.data_ptr<input_dtype>(),
        M, N
    );
}