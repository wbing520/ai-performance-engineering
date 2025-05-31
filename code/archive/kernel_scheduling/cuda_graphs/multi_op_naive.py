import torch, time

def naive(x):
    for _ in range(10):
        x += 1

if __name__ == "__main__":
    x = torch.zeros(1<<20, device='cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    naive(x)
    torch.cuda.synchronize()
    print("Naive time:", time.time() - t0)
