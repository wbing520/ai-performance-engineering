"""GPU gather benchmark comparing naive vs. coalesced access.

This replicates the Chapter 7 guidance with simple timing.
"""

from __future__ import annotations

import time
import torch

N = 1 << 20


def run(indices: torch.Tensor) -> float:
    table = torch.arange(N, device=indices.device, dtype=torch.float32)
    out = torch.empty_like(table)
    torch.cuda.synchronize()
    start = time.time()
    out = table[indices]
    torch.cuda.synchronize()
    return (time.time() - start) * 1_000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indices = torch.randint(0, N, (N,), device=device)
    ms = run(indices)
    print(f"random gather: {ms:.2f} ms")
    indices = torch.arange(N, device=device)
    ms = run(indices)
    print(f"coalesced gather: {ms:.2f} ms")


if __name__ == "__main__":
    main()
