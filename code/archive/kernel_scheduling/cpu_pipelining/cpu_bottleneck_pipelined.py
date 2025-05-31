import torch
from concurrent.futures import ThreadPoolExecutor
import math

def cpu_work(i):
    x=0
    for j in range(10000):
        x += math.sin(j)*math.cos(j)
    return x

def pipelined(x):
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(100):
            fut = executor.submit(cpu_work, i)
            x = x + 1
            fut.result()
    return x

if __name__ == "__main__":
    x=torch.zeros(1<<20,device='cuda')
    pipelined(x)
