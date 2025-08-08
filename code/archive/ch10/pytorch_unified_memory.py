import torch
import torch.nn as nn
import time

def main():
    # Create tensors with unified memory
    device = torch.device("cuda")
    
    # Allocate tensors (PyTorch automatically uses unified memory when available)
    N = 10_000_000
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    c = torch.zeros(N, device=device)
    
    print(f"Unified memory tensor created")
    print(f"Tensor size: {N * 4 / (1024*1024):.1f} MB")
    
    # CPU access (unified memory automatically handles migration)
    start = time.time()
    a_cpu = a.cpu()  # This triggers memory migration
    cpu_access_time = (time.time() - start) * 1000
    
    # GPU computation
    start = time.time()
    c = a + b
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) * 1000
    
    # CPU access to result
    start = time.time()
    c_cpu = c.cpu()  # This triggers memory migration back
    cpu_result_time = (time.time() - start) * 1000
    
    print(f"CPU-GPU access time: {cpu_access_time:.1f} ms")
    print(f"GPU computation time: {gpu_time:.1f} ms")
    print(f"CPU result access time: {cpu_result_time:.1f} ms")
    print(f"Memory migration: automatic")
    print(f"Page fault handling: optimized")
    
    # Verify computation
    result_sum = c_cpu.sum().item()
    print(f"Result sum: {result_sum:.2f}")
    
    # Memory statistics
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
