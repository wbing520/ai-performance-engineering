#!/usr/bin/env python3
# add.py — PyTorch 2.8.0-nightly-cu130 + Triton 3.3.0
import torch, time, triton, triton.language as tl

@triton.jit
def triton_vecadd(A, B, C, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    tl.store(C + offs, a + b, mask=mask)

def run_triton(N=1<<20, BLOCK=1024):
    A = torch.arange(N, dtype=torch.float32, device='cuda')
    B = torch.arange(N, 0, -1, dtype=torch.float32, device='cuda')
    C = torch.empty_like(A)
    t0 = time.perf_counter()
    triton_vecadd[(N + BLOCK - 1) // BLOCK](A, B, C, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

if __name__ == "__main__":
    N = 1<<20
    A = torch.arange(N)
    B = torch.arange(N, 0, -1)
    t0 = time.perf_counter(); C_cpu = A + B; t1 = time.perf_counter()
    A_gpu, B_gpu = A.cuda(), B.cuda()
    t2 = time.perf_counter(); C_gpu = A_gpu + B_gpu; torch.cuda.synchronize(); t3 = time.perf_counter()
    t_t = run_triton(N)
    print(f"CPU add     : {(t1-t0)*1e3:.3f} ms")
    print(f"PyTorch add : {(t3-t2)*1e3:.3f} ms")
    print(f"Triton add  : {t_t:.3f} ms")
