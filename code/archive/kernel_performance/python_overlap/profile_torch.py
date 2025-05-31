# profile_torch.py
import torch
from torch.profiler import profile, record_function, ProfilerActivity

N = 1<<20
h_buf = torch.ones(N, pin_memory=True)
d_buf = torch.empty(N, device='cuda')
stream = torch.cuda.Stream()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for i in range(5):
        with record_function("iter"):
            with torch.cuda.stream(stream):
                d_buf.copy_(h_buf, non_blocking=True)
                d_buf.mul_(2.0)
                h_buf.copy_(d_buf, non_blocking=True)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))