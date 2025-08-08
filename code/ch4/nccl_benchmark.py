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
# nccl_benchmark.py
"""
Comprehensive NCCL benchmark for testing different collective operations
and communication patterns with PyTorch 2.8 and CUDA 12.9.
"""
import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def benchmark_collective(rank, world_size, op_type, data_size, dtype, num_warmup=5, num_trials=10):
    """Benchmark a specific collective operation."""
    torch.cuda.set_device(rank)
    
    # Create test tensor
    if dtype == "float32":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.float32)
    elif dtype == "float16":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.float16)
    elif dtype == "bfloat16":
        tensor = torch.randn(data_size, device=f"cuda:{rank}", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Warmup
    for _ in range(num_warmup):
        if op_type == "allreduce":
            dist.all_reduce(tensor.clone())
        elif op_type == "allgather":
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output_tensors, tensor)
        elif op_type == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_type == "reduce":
            dist.reduce(tensor, dst=0)
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.time()
        
        if op_type == "allreduce":
            dist.all_reduce(tensor.clone())
        elif op_type == "allgather":
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output_tensors, tensor)
        elif op_type == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_type == "reduce":
            dist.reduce(tensor, dst=0)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate bandwidth (for allreduce, assume 2*(N-1)/N efficiency)
    data_bytes = tensor.numel() * tensor.element_size()
    if op_type == "allreduce":
        # Allreduce algorithm bandwidth calculation
        bandwidth_gbps = (data_bytes * 2 * (world_size - 1) / world_size) / avg_time / 1e9
    elif op_type == "allgather":
        bandwidth_gbps = data_bytes * world_size / avg_time / 1e9
    else:
        bandwidth_gbps = data_bytes / avg_time / 1e9
    
    if rank == 0:
        print(f"{op_type.upper()} {dtype} {data_size} elements:")
        print(f"  Avg: {avg_time*1000:.2f} ms, Min: {min_time*1000:.2f} ms, Max: {max_time*1000:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        print(f"  Data size: {data_bytes/1024/1024:.1f} MB")

def run_benchmarks(rank, world_size, args):
    """Run comprehensive NCCL benchmarks."""
    dist.init_process_group(backend="nccl", init_method="env://")
    
    if rank == 0:
        print(f"NCCL Benchmark - World Size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print("=" * 60)
    
    # Test different data sizes and operations
    data_sizes = [1024, 1024*1024, 16*1024*1024, 64*1024*1024]  # 4KB to 256MB
    operations = ["allreduce", "allgather", "broadcast"]
    dtypes = ["float32", "float16", "bfloat16"]
    
    for op in operations:
        if args.operation and op not in args.operation:
            continue
            
        for dtype in dtypes:
            if args.dtype and dtype not in args.dtype:
                continue
                
            for size in data_sizes:
                if args.max_size and size * 4 > args.max_size * 1024 * 1024:
                    continue
                    
                benchmark_collective(rank, world_size, op, size, dtype,
                                   args.warmup, args.trials)
                
                if rank == 0:
                    print("-" * 40)
    
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="NCCL Benchmark Suite")
    parser.add_argument("--world-size", type=int, default=2,
                       help="Number of processes (default: 2)")
    parser.add_argument("--operation", nargs="+", 
                       choices=["allreduce", "allgather", "broadcast"],
                       help="Operations to benchmark (default: all)")
    parser.add_argument("--dtype", nargs="+",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data types to test (default: all)")
    parser.add_argument("--max-size", type=int, default=256,
                       help="Maximum data size in MB (default: 256)")
    parser.add_argument("--warmup", type=int, default=5,
                       help="Number of warmup iterations (default: 5)")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of benchmark trials (default: 10)")
    
    args = parser.parse_args()
    
    # Set environment variables for distributed training
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    
    world_size = min(args.world_size, torch.cuda.device_count())
    if world_size < 2:
        print("This benchmark requires at least 2 GPUs")
        return
    
    print(f"Running benchmark with {world_size} GPUs")
    mp.spawn(run_benchmarks, args=(world_size, args), nprocs=world_size)

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
