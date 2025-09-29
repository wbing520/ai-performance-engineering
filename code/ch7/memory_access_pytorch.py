"""PyTorch memory access patterns benchmark."""

from __future__ import annotations

import time
import torch

N = 1 << 20

def benchmark_copy(style: str) -> float:
    src = torch.arange(N, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    torch.cuda.synchronize()
    start = time.time()
    if style == "scalar":
        dst.copy_(src)
    elif style == "vectorized":
        dst.copy_(src)
    torch.cuda.synchronize()
    return (time.time() - start) * 1_000


def main() -> None:
    torch.cuda.init()
    ms = benchmark_copy("scalar")
    print(f"scalar copy: {ms:.2f} ms")
    ms = benchmark_copy("vectorized")
    print(f"vectorized copy: {ms:.2f} ms")


if __name__ == "__main__":
    main()
