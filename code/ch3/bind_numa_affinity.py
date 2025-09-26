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
    return "blackwell" if compute_capability == "10.0" else "other"


def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": []
    }

# Load libnuma for memory binding
try:
    # Try different possible library names
    for libname in ["libnuma.so", "libnuma.so.1", "/usr/lib/aarch64-linux-gnu/libnuma.so.1"]:
        try:
            _libnuma = ctypes.CDLL(libname)
            break
        except OSError:
            continue
    else:
        raise OSError("Could not load libnuma library")
    
    if _libnuma.numa_available() < 0:
        raise RuntimeError("NUMA not available on this system")
    
    _libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
    _libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
    NUMA_AVAILABLE = True
except (OSError, RuntimeError) as e:
    print(f"Warning: NUMA not available: {e}")
    NUMA_AVAILABLE = False
    _libnuma = None

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
    try:
        out = subprocess.run(
            ["numactl", "--show"],
            capture_output=True, text=True
        ).stdout
        
        # physcpubind: CPU ranges
        phys_match = re.search(r"physcpubind:\s*([\d,-]+)", out)
        if phys_match:
            phys = phys_match.group(1)
            cpus = parse_physical_cpu_list(phys)
        else:
            cpus = list(range(psutil.cpu_count()))
        
        # preferred node: <n>
        node_match = re.search(r"preferred node:\s*(\d+)", out)
        if node_match:
            node = int(node_match.group(1))
        else:
            node = 0  # Default to node 0
        
        return cpus, node
    except Exception as e:
        print(f"Warning: Could not get NUMA information: {e}")
        # Return default values
        return list(range(psutil.cpu_count())), 0

def get_gpu_numa_node(device: int):
    """Determine the NUMA node for the given GPU using its PCI bus ID."""
    try:
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
    except Exception as e:
        print(f"Warning: Could not determine GPU NUMA node: {e}")
        return 0  # Default to node 0

def set_numa_affinity(node: int):
    """Bind current process to CPUs and memory of the given NUMA node."""
    # CPU affinity
    cpus, _ = get_numa_cpus_and_memory()
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    
    # Memory affinity
    if NUMA_AVAILABLE and _libnuma is not None:
        _libnuma.numa_run_on_node(node)
        _libnuma.numa_set_preferred(node)
    else:
        print(f"NUMA not available, skipping memory binding for node {node}")
    
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
    """Main function to demonstrate NUMA affinity binding."""
    print("=== Chapter 3: NUMA Affinity Binding Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU NUMA binding")
        return
    
    try:
        # Try to initialize distributed training if environment variables are set
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend="nccl", init_method="env://")
            print("Distributed training initialized")
        else:
            print("Distributed training not configured, running in single-GPU mode")
    except Exception as e:
        print(f"Could not initialize distributed training: {e}")
        print("Running in single-GPU mode")
    
    # Determine and bind to the GPU's NUMA node
    device = torch.cuda.current_device()
    gpu_node = get_gpu_numa_node(device)
    set_numa_affinity(gpu_node)
    
    # Prepare data loader with NUMA-bound workers (disable multiprocessing for CUDA compatibility)
    dataset = MyDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
        pin_memory=True,
        # worker_init_fn=worker_init_fn  # Disable worker init function
    )
    
    # Build model
    model = torch.nn.Linear(224*224*3, 10).to("cuda")
    
    # Wrap with DDP if distributed training is available
    if dist.is_initialized():
        ddp_model = DDP(model, device_ids=[device])
    else:
        ddp_model = model
    
    print("Starting training loop with NUMA affinity...")
    
    # Training loop
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to("cuda"), target.to("cuda")
        outputs = ddp_model(data)
        print(f"Batch {batch_idx}: Processed {data.shape[0]} samples")
        if batch_idx >= 2:  # Just run a few batches for demo
            break
    
    print("NUMA affinity binding demo completed successfully!")

if __name__ == "__main__":
    main()

# Note: Architecture-specific optimizations are handled in arch_config.py
# The following configuration options are not available in the current PyTorch version:
# - torch._inductor.config.triton.use_hopper_optimizations
# - torch._inductor.config.triton.hbm3_optimizations
# - torch._inductor.config.triton.use_blackwell_optimizations
# - torch._inductor.config.triton.hbm3e_optimizations
# - torch._inductor.config.triton.tma_support
# - torch._inductor.config.triton.autotune_mode
# - torch._dynamo.config.automatic_dynamic_shapes
