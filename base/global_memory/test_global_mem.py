import torch

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
import cuda_ops

from utils.utils import benchmark_kernel

M, N = 1024, 1024*4
input = torch.randn(M, N, device="cuda", dtype=torch.float32)
weight = torch.randn(N, device="cuda", dtype=torch.float32) # dim=-1相同时共用weight
output = torch.empty_like(input)

cuda_ops.global_mem(output, input, weight)

# torch.testing.assert_close(input*weight, output)

pass