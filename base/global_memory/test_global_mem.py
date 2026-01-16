import torch

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
import cuda_ops

from utils.utils import benchmark_kernel

M, N = 1024, 1024*4
input = torch.randn(M, N, device="cuda", dtype=torch.float32)
output = torch.empty_like(input)

cuda_ops.global_mem(output, input)

# torch.testing.assert_close(input, output)

pass