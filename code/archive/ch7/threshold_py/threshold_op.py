import torch

N = 1 << 20
X = torch.randn(N, device='cuda')
Y = torch.maximum(X, torch.zeros_like(X))
torch.cuda.synchronize()
print("Threshold operation complete")
