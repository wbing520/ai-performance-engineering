import torch, time
N = 1<<20
a = torch.ones(N, device='cuda')
b = torch.ones(N, device='cuda')
torch.cuda.synchronize()
t0 = time.time()
c = a + b
d = c * b
torch.cuda.synchronize()
print("Naive time:", time.time() - t0)
