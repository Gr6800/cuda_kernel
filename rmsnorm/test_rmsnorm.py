import torch

import sys
sys.path.append("/nfsmnt/guanrui012/github/hopper/cuda_kernel")
import cuda_ops

from utils.utils import benchmark_kernel

def rmsnorm_ref(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: float):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)

def test_rmsnorm():
    B = 4 # batch
    H = 1024 # hidden_dim
    # input = torch.randn(B, H, device="cuda").to(torch.float32)
    input = torch.ones(B, H, device="cuda").to(torch.float32)
    # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
    output = torch.empty_like(input)
    # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
    weight = torch.arange(0, H, device="cuda").to(torch.float32)
    # weight = torch.randn(H, device="cuda").to(torch.float32) # TODO 换了weight之后输出有误
    eps = 1e-6

    # print(input)
    cuda_ops.rmsnorm(output, input, weight, eps)
    # print(f"cuda_output: {output}")

def test_rmsnorm_accuracy():
    B = 4096*24 # batch
    H = 128 # hidden_dim
    input = torch.randn(B, H, device="cuda").to(torch.float32)
    # input = torch.ones(B, H, device="cuda").to(torch.float32)
    # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
    output = torch.empty_like(input)
    # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
    # weight = torch.arange(0, H, device="cuda").to(torch.float32)
    weight = torch.randn(H, device="cuda").to(torch.float32) # TODO 换了weight之后输出有误
    eps = 1e-6

    # print(input)
    cuda_ops.rmsnorm(output, input, weight, eps)
    # print(f"cuda_output: {output}")
    ref_output = rmsnorm_ref(input, weight, eps)
    # print(f"ref_output: {ref_output}")
    torch.testing.assert_close(ref_output, output)
    print("OK!")

def test_rmsnorm_performance():
    B = 4096*24 # batch
    H = 128 # hidden_dim
    # input = torch.randn(B, H, device="cuda").to(torch.float32)
    input = torch.ones(B, H, device="cuda").to(torch.float32)
    # input = torch.arange(H, device="cuda").to(torch.float32).repeat(B, 1)
    output = torch.empty_like(input)
    # weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
    weight = torch.arange(0, H, device="cuda").to(torch.float32)
    # weight = torch.randn(H, device="cuda").to(torch.float32) # TODO 换了weight之后输出有误
    eps = 1e-6

    duration_cuda, _ = benchmark_kernel("rmsnorm", cuda_ops.rmsnorm, 100, output, input, weight, eps)
    # duration_c, _ = benchmark_kernel("rmsnorm", cuda_ops.rmsnorm, 100, output, input, weight, eps)
    print(f"input_args[B={B},H={H}], duration_cuda: {duration_cuda * 1000} ms")

if __name__ == "__main__":
    # test_rmsnorm()
    test_rmsnorm_accuracy()
    # test_rmsnorm_performance()