"""NUMA-aware affinity helpers for Chapter 3 examples.

This script demonstrates pinned CPU/memory affinity for the current process and
optionally for DataLoader workers. It incorporates the best practices captured in
perf_review_findings.md:

- CPU masks are derived from the GPU's NUMA node (not the current process).
- Memory binding uses libnuma when available, with graceful degradation.
- DataLoader workers receive their NUMA node via closure; no CUDA API calls occur
  inside worker processes.
- All system calls have basic error handling.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import psutil
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
except ImportError:  # DDP not strictly required for the demo
    DDP = None
    dist = None

# ---------------------------------------------------------------------------
# libnuma helpers
# ---------------------------------------------------------------------------

NUMA_AVAILABLE = False
_libnuma: Optional[ctypes.CDLL] = None

for candidate in ("libnuma.so", "libnuma.so.1", "/usr/lib/aarch64-linux-gnu/libnuma.so.1"):
    try:
        _libnuma = ctypes.CDLL(candidate)
        if _libnuma.numa_available() >= 0:
            _libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
            _libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
            NUMA_AVAILABLE = True
            break
    except OSError:
        continue

# ---------------------------------------------------------------------------
# NUMA parsing utilities
# ---------------------------------------------------------------------------

_CPU_LIST_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


def _expand_cpu_list(spec: str) -> List[int]:
    """Expand strings like ``"0-3,8,12-13"`` into explicit CPU lists."""
    cpus: List[int] = []
    for part in spec.split(","):
        match = _CPU_LIST_RE.match(part.strip())
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        cpus.extend(range(start, end + 1))
    return cpus


def cpus_for_numa_node(node: int) -> List[int]:
    """Return a CPU list for ``node`` using sysfs (preferred) or numactl fallback."""
    sysfs_path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if sysfs_path.exists():
        spec = sysfs_path.read_text().strip()
        cpus = _expand_cpu_list(spec)
        if cpus:
            return cpus

    # Fallback: use numactl --show physcpubind if available
    try:
        output = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return list(range(psutil.cpu_count() or 1))

    # numactl prints lines like: "node 0 cpus: 0 1 2 3"
    for line in output.stdout.splitlines():
        if f"node {node} cpus:" in line:
            cpu_list = [int(tok) for tok in line.split()[3:]]
            if cpu_list:
                return cpu_list

    return list(range(psutil.cpu_count() or 1))


def gpu_numa_node(device_index: int) -> int:
    """Best-effort detection of the NUMA node that hosts ``device_index``."""
    try:
        props = torch.cuda.get_device_properties(device_index)
        pci_bus = props.pci_bus_id
        sysfs_path = Path(f"/sys/bus/pci/devices/{pci_bus}/numa_node")
        if sysfs_path.exists():
            value = int(sysfs_path.read_text().strip())
            if value >= 0:
                return value
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: unable to query GPU NUMA node: {exc}")

    # Fallback if sysfs is unavailable
    try:
        output = subprocess.run(["numactl", "--show"], capture_output=True, text=True)
        match = re.search(r"preferred node:\s*(\d+)", output.stdout)
        if match:
            return int(match.group(1))
    except Exception:  # pylint: disable=broad-except
        pass

    return 0


def bind_current_process(node: int) -> None:
    """Bind the current process' CPU and memory policy to ``node``."""
    cpus = cpus_for_numa_node(node)
    try:
        psutil.Process(os.getpid()).cpu_affinity(cpus)
    except psutil.Error as exc:
        print(f"Warning: unable to set CPU affinity to node {node}: {exc}")

    if NUMA_AVAILABLE and _libnuma is not None:
        _libnuma.numa_run_on_node(node)
        _libnuma.numa_set_preferred(node)
    else:
        print("NUMA library unavailable; memory policy not updated")

    print(f"PID {os.getpid()} bound to NUMA node {node} (CPUs={cpus})")


# ---------------------------------------------------------------------------
# Dataset / worker demo
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    def __init__(self, length: int = 1024, feature_dim: int = 224 * 224 * 3) -> None:
        self.length = length
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        del index
        return torch.randn(self.feature_dim), torch.randint(0, 10, (1,)).item()


def make_worker_init(numa_node: int):
    """Return a worker init function that binds to ``numa_node`` without touching CUDA."""
    def _init(worker_id: int):
        del worker_id
        bind_current_process(numa_node)
    return _init


# ---------------------------------------------------------------------------
# Main demo routine
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Chapter 3 NUMA affinity demo ===")
    if not torch.cuda.is_available():
        print("CUDA not available; exiting")
        raise SystemExit(0)

    device_index = torch.cuda.current_device()
    node = gpu_numa_node(device_index)
    bind_current_process(node)

    dataset = DummyDataset()
    num_workers = min(4, os.cpu_count() or 1)  # safe default; adjust as needed
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=make_worker_init(node) if num_workers > 0 else None,
    )

    model = torch.nn.Linear(dataset.feature_dim, 10).to("cuda")

    if dist is not None and "RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            model = DDP(model, device_ids=[device_index])
            print("Initialized torch.distributed")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: unable to initialize DDP ({exc}); continuing in single-process mode")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()

    for batch_idx, (inputs, target) in enumerate(dataloader, start=1):
        inputs = inputs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}")
        if batch_idx == 50:  # keep the demo short
            break

    print("Demo complete")
