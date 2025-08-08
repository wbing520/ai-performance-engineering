# add_parallel.py
# Proper PyTorch code using vectorized operations

import torch
import time

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A

torch.cuda.synchronize()

start_time = time.time()

# Proper parallel approach using vectorized operation
# Launches a single GPU kernel that adds all elements in parallel
C = A + B

torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) * 1000

print(f"Parallel PyTorch time: {elapsed_time:.2f} ms")
print(f"Result: C[0] = {C[0]}, C[N-1] = {C[N-1]}")
