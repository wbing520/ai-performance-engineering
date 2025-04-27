import torch

def run_concurrent(x):
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    for _ in range(100):
        with torch.cuda.stream(s1):
            x[:x.size(0)//2] = x[:x.size(0)//2] + 1
        with torch.cuda.stream(s2):
            x[x.size(0)//2:] = x[x.size(0)//2:] + 1
    torch.cuda.synchronize()

if __name__ == "__main__":
    x = torch.zeros(1<<20, device='cuda', dtype=torch.int32)
    run_concurrent(x)
