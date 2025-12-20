import torch

import sys
sys.path.append("/nfsmnt/guanrui012/github/hopper/cuda")
import cuda_ops 

def test_rmsnorm():
    B = 4 # batch
    H = 256 # hidden_dim
    input = torch.ones(B, H, device="cuda").to(torch.bfloat16)
    output = torch.empty_like(input)
    weight = torch.ones(H, device="cuda").to(torch.float32) * 0.5
    eps = 1e-6

    cuda_ops.rmsnorm()
    pass

def test_rmsnorm_accuracy():
    pass

def test_rmsnorm_performance():
    pass

if __name__ == "__main__":
    test_rmsnorm()