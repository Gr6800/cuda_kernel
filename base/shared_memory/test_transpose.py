import torch

import os, sys
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)
import cuda_ops

from utils.utils import benchmark_kernel

INPUT_ARGS = [
    [1024, 1024],
    [512, 1024],
    [1024, 512],
    [10240, 10240],
]

def test_transpose():
    M, N = 1024, 1024
    # M, N = 32, 64
    input = torch.randn(M, N, device="cuda", dtype=torch.float32)
    # input = torch.arange(M*N, device="cuda", dtype=torch.float32).view(M, N)
    output = torch.empty(N, M, device="cuda", dtype=torch.float32)

    cuda_ops.transpose(output, input)

    pass


def test_transpose_accuracy():
    unpass = 0
    for INPUT_ARG in INPUT_ARGS:
        M, N = INPUT_ARG
        input = torch.randn(M, N, device="cuda", dtype=torch.float32)
        output = torch.empty(N, M, device="cuda", dtype=torch.float32)

        cuda_ops.transpose(output, input)

        try:
            torch.testing.assert_close(input.transpose(0,1), output)
        except Exception as e:
            unpass += 1
            print(f"ERROR in input_args[M={M},N={N}]")

    print(f"case {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} is OK!")

def test_rmsnorm_performance():
    for INPUT_ARG in INPUT_ARGS:
        M, N = INPUT_ARG
        input = torch.randn(M, N, device="cuda", dtype=torch.float32)
        output = torch.empty(N, M, device="cuda", dtype=torch.float32)

        duration_cuda, _ = benchmark_kernel("rmsnorm", cuda_ops.transpose, 100, output, input)
        print(f"input_args[M={M},N={N}], duration_cuda: {duration_cuda * 1000} ms")

if __name__ == "__main__":
    # test_transpose()
    test_transpose_accuracy()
    # test_rmsnorm_performance()