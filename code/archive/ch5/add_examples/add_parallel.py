import torch
from torch.profiler import profile, record_function, ProfilerActivity

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A

torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, with_stack=True) as prof:
    with record_function("vector_add"):
        C = A + B
prof.step()
print(prof.key_averages().table(sort_by="cuda_time_total"))
