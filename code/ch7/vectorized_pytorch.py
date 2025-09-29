"""PyTorch vectorized vs. naive additions benchmark."""

from __future__ import annotations

import time
import torch

N = 1 << 20

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.arange(N, device=device, dtype=torch.float32)
    b = 2 * a
    c = torch.empty_like(a)

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.time()
    for i in range(N):
        c[i] = a[i] + b[i]
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    sequential_ms = (time.time() - start) * 1_000

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.time()
    c = a + b
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    vector_ms = (time.time() - start) * 1_000

    print(f"naive loop: {sequential_ms:.2f} ms, vectorized: {vector_ms:.2f} ms")


if __name__ == "__main__":
    main()
