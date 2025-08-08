import torch
import time

# Create tensors on GPU
X = torch.randn(1<<20, device='cuda')

# Define operations (for example purposes, treat these as our opA, opB, opC)
def opA(x): 
    return x * 1.1

def opB(x): 
    return x + 2.0

def opC(x): 
    return x.sqrt()

# Warm-up the sequence once (required before capture to initialize CUDA code)
_ = opC(opB(opA(X)))
torch.cuda.synchronize()

# Capture the graph
g = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
torch.cuda.synchronize()

with torch.cuda.graph(g, stream=stream):
    # Record the sequence A->B->C in the graph
    Y = opA(X)
    Z = opB(Y)
    W = opC(Z)

torch.cuda.synchronize()

# Replay the captured graph 100 times
t0 = time.time()
for i in range(100):
    g.replay()
torch.cuda.synchronize()
print("Total time with CUDA Graph:", time.time() - t0)

# Compare with non-graph version
t0 = time.time()
for i in range(100):
    Y = opA(X)
    Z = opB(Y)
    W = opC(Z)
torch.cuda.synchronize()
print("Total time without CUDA Graph:", time.time() - t0)
