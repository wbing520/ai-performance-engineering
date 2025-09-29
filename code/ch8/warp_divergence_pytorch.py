"""PyTorch-side patterns for avoiding warp divergence.

These snippets intentionally highlight how to keep GPU control flow uniform by
preferring vectorized tensor ops over Python loops, masking work, and (optionally)
fusing conditionals with torch.compile.
"""

from __future__ import annotations

import time
import torch


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int = 20) -> float:
    """Run fn `iters` times and report average latency in milliseconds."""
    _sync_if_needed()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_if_needed()
    elapsed_ms = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<28}: {elapsed_ms:7.3f} ms")
    return elapsed_ms


def compare_threshold_ops() -> None:
    """Compare common ReLU-style thresholding implementations."""
    device = _device()
    if device.type != "cuda":
        print("CUDA not available; skipping threshold comparison")
        return

    print("\n=== Thresholding without divergence ===")
    n = 1 << 18
    x = torch.randn(n, device=device)
    zeros = torch.zeros_like(x)

    y_max = None
    def op_max():
        nonlocal y_max
        y_max = torch.maximum(x, zeros)

    y_where = None
    def op_where():
        nonlocal y_where
        y_where = torch.where(x > 0, x, zeros)

    y_clamp = None
    def op_clamp():
        nonlocal y_clamp
        y_clamp = torch.clamp_min(x, 0.0)

    y_relu = None
    def op_relu():
        nonlocal y_relu
        y_relu = torch.nn.functional.relu(x)

    _benchmark("torch.maximum", op_max)
    _benchmark("torch.where", op_where)
    _benchmark("torch.clamp_min", op_clamp)
    _benchmark("torch.nn.functional.relu", op_relu)

    numerically_equal = (
        torch.equal(y_max, y_where)
        and torch.equal(y_where, y_clamp)
        and torch.equal(y_clamp, y_relu)
    )
    print(f"Outputs identical: {numerically_equal}")


def compare_vectorized_conditionals() -> None:
    """Show benefit of vectorized conditionals versus Python control flow."""
    device = _device()
    if device.type != "cuda":
        print("CUDA not available; skipping conditional comparison")
        return

    print("\n=== Vectorized versus Python conditionals ===")
    n = 200_000
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)

    x_small = x[:2_000].cpu()
    y_small = y[:2_000].cpu()

    def python_loop():
        out = torch.empty_like(x_small)
        for i in range(x_small.numel()):
            out[i] = x_small[i] * 2 if x_small[i] > 0 else y_small[i] * 3
        return out

    def vectorized():
        return torch.where(x > 0, x * 2, y * 3)

    loop_start = time.perf_counter()
    loop_out = python_loop()
    loop_elapsed = (time.perf_counter() - loop_start) * 1_000
    print(f"Python loop (2K elems)       : {loop_elapsed:7.3f} ms")

    torch_out = vectorized()
    print("Vectorized (200K elems)    ", end="")
    _benchmark("", vectorized, iters=10)

    matches = torch.allclose(loop_out.cuda(), torch_out[: loop_out.numel()])
    print(f"Results match (first 2K): {matches}")


def compare_mask_strategies() -> None:
    """Contrast processing all data versus masking active subsets only."""
    device = _device()
    if device.type != "cuda":
        print("CUDA not available; skipping mask comparison")
        return

    print("\n=== Sparse mask handling ===")
    n = 200_000
    torch.manual_seed(42)
    data = torch.randn(n, device=device)
    mask = torch.rand(n, device=device) < 0.1  # 10% active

    zeros = torch.zeros_like(data)
    active_indices = mask.nonzero(as_tuple=False).squeeze()

    def process_all():
        processed = torch.sin(data) * torch.cos(data)
        return torch.where(mask, processed, zeros)

    def process_active_only():
        processed = torch.sin(data[active_indices]) * torch.cos(data[active_indices])
        result = zeros.clone()
        result[active_indices] = processed
        return result

    res_all = None
    def run_all():
        nonlocal res_all
        res_all = process_all()

    res_active = None
    def run_active():
        nonlocal res_active
        res_active = process_active_only()

    time_all = _benchmark("Process all elements", run_all)
    time_active = _benchmark("Process active subset", run_active)

    diff = torch.max(torch.abs(res_all - res_active)).item()
    print(f"Max elementwise difference: {diff:.2e}")
    if time_active > 0:
        print(f"Speedup (all / active): {time_all / time_active:5.2f}x")


def compiled_conditionals() -> None:
    """Fuse multiple conditionals with torch.compile when available."""
    if not torch.cuda.is_available():
        print("CUDA not available; skipping torch.compile example")
        return
    if not hasattr(torch, "compile"):
        print("torch.compile not present in this build; skipping")
        return

    print("\n=== torch.compile fused conditionals ===")
    device = _device()
    x = torch.randn(200_000, device=device)
    y = torch.randn(200_000, device=device)
    threshold = 0.5

    def uncompiled(x, y, thresh):
        mask_hi = x > thresh
        mask_lo = x < -thresh
        out = torch.where(mask_hi, x * 2.0, x)
        out = torch.where(mask_lo, y * 0.5, out)
        return torch.where(~(mask_hi | mask_lo), x + y, out)

    compiled = torch.compile(uncompiled, fullgraph=True, mode="reduce-overhead")

    # Warm-up to trigger compilation
    compiled(x, y, threshold)

    _benchmark("Uncompiled", lambda: uncompiled(x, y, threshold))
    _benchmark("Compiled", lambda: compiled(x, y, threshold))

    max_diff = torch.max(torch.abs(uncompiled(x, y, threshold) - compiled(x, y, threshold))).item()
    print(f"Max difference post-compile: {max_diff:.2e}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; demonstrations require a GPU")
        return

    print("=== PyTorch warp-divergence avoidance examples ===")
    print(f"Device: {torch.cuda.get_device_name()}\n")

    compare_threshold_ops()
    compare_vectorized_conditionals()
    compare_mask_strategies()
    compiled_conditionals()

    print("\nKey takeaways:")
    print("- Prefer vectorized tensor ops; they map to uniform GPU kernels.")
    print("- Use masking or indexing to skip inactive work instead of branching per element.")
    print("- torch.compile can fuse chains of conditionals to reduce launches and control-flow overhead.")


if __name__ == "__main__":
    main()
