import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

torch.cuda.synchronize()
for i in range(N):
    C[i] = A[i] + B[i]
torch.cuda.synchronize()
print("Done sequential add")
