import torch
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/nfsmnt/guanrui012/github/hopper/cuda_kernel")
import cuda_ops

from utils.utils import benchmark_kernel

torch.manual_seed(0)
# INPUT_ARGS = [[4096*24, 16*2**i] for i in range(10)] # 16, ..., 8192
# INPUT_ARGS = [[4, 1024]]
INPUT_ARGS = [[4096*24, 16*2**i] for i in range(8)] # 16, ..., 8192

def rmsnorm_ref(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)

def test_rmsnorm():
    B = 2 # batch
    H = 1024 # hidden_dim
    input = torch.randn(B, H, device="cuda").to(torch.float32)
    # input = torch.ones(B, H, device="cuda").to(torch.float32)
    # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
    # input = torch.arange(B*H+1, device="cuda").to(torch.float32)[1:].reshape(B,H)   # 切片后导致非对齐
    output = torch.empty_like(input)
    # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
    # weight = torch.arange(0, H, device="cuda").to(torch.float32)
    weight = torch.randn(H, device="cuda").to(torch.float32)
    eps = 1e-6

    # print(input)
    cuda_ops.rmsnorm(output, input, weight, eps)
    cuda_ops.rmsnorm(output, input, weight, eps)
    # print(f"cuda_output: {output}")

def test_rmsnorm_accuracy():
    for INPUT_ARG in INPUT_ARGS:
        B, H = INPUT_ARG # batch, hidden_dim
        input = torch.randn(B, H, device="cuda").to(torch.float32)
        # input = torch.ones(B, H, device="cuda").to(torch.float32)
        # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
        output = torch.empty_like(input)
        weight = torch.randn(H, device="cuda").to(torch.float32)
        # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
        # weight = torch.arange(0, H, device="cuda").to(torch.float32)
        eps = 1e-6

        # print(input)
        cuda_ops.rmsnorm(output, input, weight, eps)
        # print(f"cuda_output: {output}")
        ref_output = rmsnorm_ref(input, weight, eps)
        # print(f"ref_output: {ref_output}")
        torch.testing.assert_close(ref_output, output)
    print("OK!")

def test_rmsnorm_performance():
    for INPUT_ARG in INPUT_ARGS:
        B, H = INPUT_ARG # batch, hidden_dim
        input = torch.randn(B, H, device="cuda").to(torch.float32)
        # input = torch.ones(B, H, device="cuda").to(torch.float32)
        # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
        output = torch.empty_like(input)
        # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
        # weight = torch.arange(0, H, device="cuda").to(torch.float32)
        weight = torch.randn(H, device="cuda").to(torch.float32)
        eps = 1e-6

        duration_cuda, _ = benchmark_kernel("rmsnorm", cuda_ops.rmsnorm, 100, output, input, weight, eps)
        # duration_c, _ = benchmark_kernel("rmsnorm", cuda_ops.rmsnorm, 100, output, input, weight, eps)
        print(f"input_args[B={B},H={H}], duration_cuda: {duration_cuda * 1000} ms")

if __name__ == "__main__":
    # test_rmsnorm()
    # test_rmsnorm_accuracy()
    test_rmsnorm_performance()