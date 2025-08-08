# add_sequential.py
# Naive PyTorch code that performs sequential GPU operations

import torch
import time

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

# Ensure all previous work is done
torch.cuda.synchronize()

start_time = time.time()

# Naive, Sequential GPU operations - DO NOT DO THIS
for i in range(N):
    C[i] = A[i] + B[i]  # This launches N tiny GPU operations serially

torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) * 1000

print(f"Sequential PyTorch time: {elapsed_time:.2f} ms")
print(f"Result: C[0] = {C[0]}, C[N-1] = {C[N-1]}")
