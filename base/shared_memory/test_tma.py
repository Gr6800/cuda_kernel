import torch

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
import cuda_ops

M, N = 4, 1024
input = torch.randn(M, N, device="cuda", dtype=torch.float32)
output = torch.empty_like(input)

cuda_ops.tma(output, input)

torch.testing.assert_close(input+1, output)

pass