# overlap_pytorch.py
# Python:3.11, PyTorch nightly 2.8.0+, CUDA 13.0, Triton 2.5.0
import torch

N, ITER = 1<<20, 10
h_buf = torch.ones(N, pin_memory=True)
d_buf = torch.empty(N, device='cuda')
stream = torch.cuda.Stream()

for i in range(ITER):
    h_buf.fill_(i)
    with torch.cuda.stream(stream):
        d_buf.copy_(h_buf, non_blocking=True)
        d_buf.mul_(2.0)
        h_buf.copy_(d_buf, non_blocking=True)
torch.cuda.synchronize()
print("Completed")