import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

torch.cuda.synchronize()
# Naive, sequential GPU operations (DO NOT DO THIS)
for i in range(N):
    C[i] = A[i] + B[i]  # launches N tiny GPU ops serially
torch.cuda.synchronize()
