"""torch.compile benchmarking utilities (Chapter 13/14 best practices).

Demonstrates:
- Warmup iterations before timing.
- Safer compile settings (no unconditional fullgraph).
- Optional AMP usage and fused optimizers.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self, input_dim: int = 256, hidden: int = 256, out_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def benchmark_compile(mode: str = "default", amp: bool = True, use_fused: bool = True) -> None:
    assert mode in {"default", "reduce-overhead", "max-autotune"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    optimizer_cls = optim.AdamW if not use_fused or device.type != "cuda" else (lambda params: optim.AdamW(params, lr=1e-3, fused=True))
    optimizer = optimizer_cls(model.parameters())

    data = torch.randn(32, 256, device=device)
    target = torch.randint(0, 10, (32,), device=device)

    compile_kwargs = dict(mode=mode, dynamic=True)
    compiled = torch.compile(model, **compile_kwargs)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    autocast_cm = torch.cuda.amp.autocast(device_type="cuda") if amp and device.type == "cuda" else nullcontext()

    with torch.no_grad():
        for _ in range(3):
            compiled(data)

    torch.cuda.synchronize(device) if device.type == "cuda" else None

    iters = 10
    losses = []
    start = time.time()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        with autocast_cm:
            logits = compiled(data)
            loss = nn.functional.cross_entropy(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.time() - start) / iters * 1000
    print(f"mode={mode}, amp={amp}, fused={use_fused} -> {elapsed:.2f} ms/iter, loss={losses[-1]:.4f}")


def main() -> None:
    for mode in ("default", "reduce-overhead", "max-autotune"):
        benchmark_compile(mode)


if __name__ == "__main__":
    main()
