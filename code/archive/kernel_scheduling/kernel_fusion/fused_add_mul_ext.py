import torch, time
from torch.utils.cpp_extension import load

ext = load(name="fusion_ext", sources=["fused_add_mul.cu"], verbose=False)
a = torch.ones(1<<20, device='cuda')
b = torch.ones(1<<20, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
ext.fusedKernel(a.data_ptr(), b.data_ptr(), a.data_ptr(), a.numel())
torch.cuda.synchronize()
print("Fused time:", time.time() - t0)
