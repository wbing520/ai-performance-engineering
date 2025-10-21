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
    autocast_cm = torch.autocast(device_type="cuda") if amp and device.type == "cuda" else nullcontext()

    # Warmup iterations
    with torch.no_grad():
        for _ in range(3):
            compiled(data)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Use CUDA Events for accurate GPU timing (preferred over time.time())
    iters = 10
    losses = []
    
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
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
        end_event.record()
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event) / iters
    else:
        elapsed = (time.time() - start) / iters * 1000
    
    print(f"mode={mode}, amp={amp}, fused={use_fused} -> {elapsed:.2f} ms/iter, loss={losses[-1]:.4f}")


def compile_with_strict_graph() -> None:
    """
    PyTorch 2.9 feature: explicit graph break control.
    Use torch._dynamo.error_on_graph_break() to enforce full-graph compilation.
    """
    import torch._dynamo
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    data = torch.randn(32, 256, device=device)
    
    print("\n" + "=" * 80)
    print("PyTorch 2.9 Graph Break Control Example")
    print("=" * 80)
    
    # Example 1: Error on graph break (strict mode for debugging)
    print("\nExample 1: Strict mode - error on any graph break")
    try:
        with torch._dynamo.error_on_graph_break():
            compiled = torch.compile(model, mode="reduce-overhead")
            # This will raise an error if any graph breaks occur
            _ = compiled(data)
            print("✓ No graph breaks detected - full graph compilation successful")
    except RuntimeError as e:
        print(f"✗ Graph break detected: {e}")
        print("  Use this mode during development to identify compilation issues")
    
    # Example 2: Allow graph breaks (permissive mode)
    print("\nExample 2: Permissive mode - explicitly allow graph breaks")
    with torch._dynamo.allow_graph_break():
        compiled = torch.compile(model, mode="default")
        _ = compiled(data)
        print("✓ Graph breaks allowed - compilation may split into subgraphs")
    
    # Example 3: Default behavior (no explicit control)
    print("\nExample 3: Default behavior - automatic handling")
    compiled = torch.compile(model, mode="default")
    _ = compiled(data)
    print("✓ Default mode - torch.compile handles breaks automatically")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("- Use error_on_graph_break() during development to catch issues")
    print("- Use allow_graph_break() when you expect/accept partial compilation")
    print("- Default mode is usually best for production (automatic handling)")
    print("- Graph breaks reduce optimization potential but allow more Python code")
    print("=" * 80)


def demonstrate_graph_break_scenarios() -> None:
    """
    Show common scenarios that cause graph breaks and how to handle them.
    """
    import torch._dynamo
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print("Common Graph Break Scenarios")
    print("=" * 80)
    
    # Scenario 1: Dynamic control flow based on tensor values
    class ModelWithDynamicControl(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            out = self.fc(x)
            # This causes a graph break because it depends on tensor values
            if out.sum() > 0:
                return out * 2
            return out
    
    print("\nScenario 1: Dynamic control flow")
    model1 = ModelWithDynamicControl().to(device)
    data1 = torch.randn(4, 10, device=device)
    
    # This will have graph breaks
    compiled1 = torch.compile(model1, mode="default")
    _ = compiled1(data1)
    print("✓ Model with dynamic control compiled (with graph breaks)")
    
    # Scenario 2: Python print statements (common during debugging)
    class ModelWithPrint(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            out = self.fc(x)
            # print() causes a graph break
            # print(f"Output shape: {out.shape}")  # Uncomment to see break
            return out
    
    print("\nScenario 2: Python side effects (print, logging)")
    model2 = ModelWithPrint().to(device)
    compiled2 = torch.compile(model2, mode="default")
    _ = compiled2(data1)
    print("✓ Model without print compiled successfully")
    print("  (Uncommenting print would cause graph break)")
    
    print("\n" + "=" * 80)
    print("Tips to Avoid Graph Breaks:")
    print("1. Use torch operations instead of Python control flow")
    print("2. Remove debug print statements in compiled functions")
    print("3. Use torch.cond() for conditional execution")
    print("4. Keep Python side effects outside the compiled region")
    print("5. Profile with torch._dynamo.explain() to see break points")
    print("=" * 80)


def main() -> None:
    # Original benchmarks
    for mode in ("default", "reduce-overhead", "max-autotune"):
        benchmark_compile(mode)
    
    # PyTorch 2.9 graph break control examples
    compile_with_strict_graph()
    demonstrate_graph_break_scenarios()


if __name__ == "__main__":
    main()
