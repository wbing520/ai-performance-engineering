# matrix_multiply_amp.py
# Hardware: GB200/H100, CUDA 13.0, Python 3.11
# PyTorch: nightly 2.8.0+, OpenAI Triton 2.5.0
import torch

N, K, M = 1024, 1024, 1024
A = torch.randn(N, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, M, device='cuda', dtype=torch.float16)

with torch.cuda.amp.autocast(dtype=torch.float16):
    C = torch.matmul(A, B)
torch.cuda.synchronize()
print("C[0,0] =", C[0,0].item())