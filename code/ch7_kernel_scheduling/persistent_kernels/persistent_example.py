import torch, time
from torch.utils.cpp_extension import load

ext = load(name="persistent_ext", sources=["persistent_kernel_ext.cu"], verbose=False)
x = torch.rand(1<<10, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
ext.run_persistent(x, 1000)
torch.cuda.synchronize()
print("Time:", time.time() - t0)
