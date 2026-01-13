import torch
from typing import Callable

def benchmark_kernel(op_name: str, impl_func: Callable, iterations: int = 50, *args, **kwargs):
    """Kernel Performance Benchmark"""
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # warm up
    for _ in range(3):
        impl_func(*args, **kwargs)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        result = impl_func(*args, **kwargs)
    torch.cuda.synchronize()
    duration = (time.perf_counter() - start) / iterations
    
    return duration, result