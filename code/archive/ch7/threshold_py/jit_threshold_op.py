import torch

@torch.compile(fullgraph=True)
def threshold_op(X):
    return torch.maximum(X, torch.zeros_like(X))

N = 1 << 20
X = torch.randn(N, device='cuda')
Y = threshold_op(X)
torch.cuda.synchronize()
print("JIT threshold operation complete")
