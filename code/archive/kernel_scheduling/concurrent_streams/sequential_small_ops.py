import torch

def run_sequential(x):
    for _ in range(100):
        x = x + 1
    return x

if __name__ == "__main__":
    x = torch.zeros(1<<20, device='cuda', dtype=torch.int32)
    run_sequential(x)
