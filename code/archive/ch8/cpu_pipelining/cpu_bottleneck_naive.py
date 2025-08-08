import torch
import math

def cpu_work(i):
    x=0
    for j in range(10000):
        x += math.sin(j)*math.cos(j)
    return x

def naive(x):
    for i in range(100):
        cpu_work(i)
        x = x + 1
    return x

if __name__ == "__main__":
    x=torch.zeros(1<<20,device='cuda')
    naive(x)
