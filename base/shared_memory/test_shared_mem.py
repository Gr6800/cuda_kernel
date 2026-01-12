import torch
import os
import sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
import cuda_ops

S, H = 1024, 32
input = torch.randn(S, H, device="cuda").to(torch.float32)
output = torch.empty_like(input)

cuda_ops.shared_mem(output, input)
cuda_ops.shared_mem(output, input)

# torch.testing.assert_close(input, output)
pass