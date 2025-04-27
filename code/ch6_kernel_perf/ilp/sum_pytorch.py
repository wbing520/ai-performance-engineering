# sum_pytorch.py
# Python:3.11, PyTorch nightly 2.8.0+, CUDA 13.0, Triton 2.5.0
import torch
N = 1<<20
data = torch.ones(N,device='cuda',dtype=torch.float32)
torch.cuda.synchronize()
print("Sum =", data.sum().item())