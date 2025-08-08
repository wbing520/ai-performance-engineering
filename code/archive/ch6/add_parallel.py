import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A

torch.cuda.synchronize()
# Proper parallel approach using a vectorized operation
C = A + B  # launches one GPU kernel that adds all elements in parallel
torch.cuda.synchronize()
