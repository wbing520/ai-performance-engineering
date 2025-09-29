"""Simple benchmarking script for Chapter 1 (goodput measurement)."""

from __future__ import annotations

import time
import torch


def measure_goodput(model: torch.nn.Module, device: torch.device, iterations: int = 20) -> None:
    model.eval()
    data = torch.randn(32, 256, device=device)
    target = torch.randint(0, 10, (32,), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    useful = 0.0
    overhead = 0.0

    for _ in range(iterations):
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        useful += time.time() - start

    total = useful + overhead
    ratio = useful / total if total else 0.0
    print(f"goodput={ratio * 100:.1f}% (useful={useful:.3f}s total={total:.3f}s)")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)
    measure_goodput(model, device)


if __name__ == "__main__":
    main()
