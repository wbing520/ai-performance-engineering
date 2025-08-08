import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "9.0":
        return "hopper"  # H100/H200
    elif compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return {
            "name": "Hopper H100/H200",
            "compute_capability": "9.0",
            "sm_version": "sm_90",
            "memory_bandwidth": "3.35 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3", "Transformer Engine", "Dynamic Programming"]
        }
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "3.2 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    else:
        return {
            "name": "Other",
            "compute_capability": "Unknown",
            "sm_version": "Unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": []
        }
# bind_numa_affinity.py
import os
import re
import subprocess
import psutil
import ctypes
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

# Load libnuma for memory binding
_libnuma = ctypes.CDLL("libnuma.so")

if _libnuma.numa_available() < 0:
    raise RuntimeError("NUMA not available on this system")

_libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
_libnuma.numa_set_preferred.argtypes = [ctypes.c_int]

def parse_physical_cpu_list(phys_str: str):
    """Parse numactl --show physcpubind output like '0-3,8-11' into a list of ints."""
    cpus = []
    for part in phys_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus

def get_numa_cpus_and_memory():
    """Run `numactl --show` and parse CPU and memory policy information."""
    out = subprocess.run(
        ["numactl", "--show"],
        capture_output=True, text=True
    ).stdout
    
    # physcpubind: CPU ranges
    phys = re.search(r"physcpubind:\s*([\d,-]+)", out).group(1)
    cpus = parse_physical_cpu_list(phys)
    
    # preferred node: <n>
    node = int(re.search(r"preferred node:\s*(\d+)", out).group(1))
    
    return cpus, node

def get_gpu_numa_node(device: int):
    """Determine the NUMA node for the given GPU using its PCI bus ID."""
    props = torch.cuda.get_device_properties(device)
    pci = props.pci_bus_id  # e.g., '0000:03:00.0'
    sysfs_path = f"/sys/bus/pci/devices/{pci}/numa_node"
    
    try:
        with open(sysfs_path, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        # Fallback to preferred node from numactl if sysfs entry missing
        _, node = get_numa_cpus_and_memory()
        return node

def set_numa_affinity(node: int):
    """Bind current process to CPUs and memory of the given NUMA node."""
    # CPU affinity
    cpus, _ = get_numa_cpus_and_memory()
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    
    # Memory affinity
    _libnuma.numa_run_on_node(node)
    _libnuma.numa_set_preferred(node)
    
    print(f"PID={os.getpid()} bound to NUMA node {node} (CPUs={cpus})")

def worker_init_fn(worker_id: int):
    """Initialize DataLoader workers with NUMA bindings."""
    gpu_node = get_gpu_numa_node(torch.cuda.current_device())
    set_numa_affinity(gpu_node)
    print(f"Worker {worker_id} (PID={os.getpid()}) bound to NUMA node {gpu_node}")

class MyDataset(Dataset):
    """Example dataset class."""
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return dummy data
        return torch.randn(224*224*3), torch.randint(0, 10, (1,)).item()

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    
    # Determine and bind to the GPU's NUMA node
    device = torch.cuda.current_device()
    gpu_node = get_gpu_numa_node(device)
    set_numa_affinity(gpu_node)
    
    # Prepare data loader with NUMA-bound workers
    dataset = MyDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    # Build and wrap model
    model = torch.nn.Linear(224*224*3, 10).to("cuda")
    ddp_model = DDP(model, device_ids=[device])
    
    # Training loop
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to("cuda"), target.to("cuda")
        outputs = ddp_model(data)
        # ... backpropagation, optimizer steps, etc.
        if batch_idx >= 5:  # Just run a few batches for demo
            break

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == "10.0":  # Blackwell B200/B300
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.8 features
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.autotune_mode = "max-autotune"
    torch._dynamo.config.automatic_dynamic_shapes = True
