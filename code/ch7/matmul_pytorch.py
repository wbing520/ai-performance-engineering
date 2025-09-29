"""PyTorch naive vs vectorized matmul benchmark."""

from __future__ import annotations

import time
import torch

M = 512
N = 512
K = 512


def benchmark(op) -> float:
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    torch.cuda.synchronize()
    start = time.time()
    op(a, b)
    torch.cuda.synchronize()
    return (time.time() - start) * 1_000


def main() -> None:
    torch.cuda.init()
    naive_time = benchmark(lambda x, y: torch.einsum("ik,kj->ij", x, y))
    optimized_time = benchmark(lambda x, y: torch.matmul(x, y))
    print(f"naive einsum: {naive_time:.2f} ms")
    print(f"torch.matmul: {optimized_time:.2f} ms")


if __name__ == "__main__":
    main()
