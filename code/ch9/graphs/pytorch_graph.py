# pytorch_graph.py
import torch
import time

N = 1<<20
X = torch.randn(N, device='cuda')

def opA(x): return x * 1.1
def opB(x): return x + 2.0
def opC(x): return x.sqrt()

# Warm-up
_ = opC(opB(opA(X)))
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
with torch.cuda.graph(g, stream=stream):
    Y = opA(X); Z = opB(Y); W = opC(Z)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    g.replay()
torch.cuda.synchronize()
print("CUDA Graph total time:", time.time() - t0)