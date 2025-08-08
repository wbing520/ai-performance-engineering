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
#!/usr/bin/env python3
"""
Comprehensive Profiling Tools Demo
Latest profiling tools for PyTorch 2.8, CUDA 12.9, and Blackwell B200/B300

This script demonstrates:
- Nsight Systems (nsys) timeline analysis
- Nsight Compute (ncu) kernel-level analysis
- PyTorch Profiler with enhanced features
- HTA (Holistic Tracing Analysis) for multi-GPU
- Perf system-level analysis
- Memory profiling
- FLOP counting
- Module-level analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import time
import psutil
import GPUtil
import subprocess
import os
import json
from typing import Dict, List, Any
import numpy as np


class ProfilingDemoModel(nn.Module):
    """A model for demonstrating comprehensive profiling."""
    
    def __init__(self, input_size=1024, hidden_size=512, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        with nvtx.annotate("fc1_forward"):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        with nvtx.annotate("fc2_forward"):
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        with nvtx.annotate("fc3_forward"):
            x = self.fc3(x)
        
        return x


def configure_blackwell_optimizations():
    """Configure PyTorch 2.8 nightly optimizations for Blackwell B200/B300."""
    if torch.cuda.is_available():
        # Enable Blackwell B200/B300 specific optimizations
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.triton.autotune_mode = "max-autotune"
        
        # Enable advanced optimizations
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.triton.use_blackwell_tensor_cores = True
        
        # Memory optimizations for HBM3e
        torch._inductor.config.triton.hbm3e_memory_optimizations = True
        
        # Enhanced profiling configuration
        torch._inductor.config.triton.profiler_mode = "max-autotune"
        torch._inductor.config.triton.enable_blackwell_features = True
        
        # Enable dynamic shapes for better performance
        torch._dynamo.config.automatic_dynamic_shapes = True
        
        # Enable advanced memory optimizations
        torch._inductor.config.triton.enable_advanced_memory_optimizations = True
        
        print("Blackwell B200/B300 optimizations enabled")


def demonstrate_pytorch_profiler():
    """Demonstrate enhanced PyTorch profiler with latest features."""
    print("\n=== PyTorch Profiler Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    model.train()
    
    # Create sample data
    batch_size = 32
    x = torch.randn(batch_size, 1024, device=device, requires_grad=True)
    target = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Enhanced profiler configuration for PyTorch 2.8
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )
    ) as prof:
        for i in range(10):
            with record_function("training_iteration"):
                optimizer.zero_grad()
                
                with nvtx.annotate("forward_pass"):
                    output = model(x)
                
                with nvtx.annotate("loss_computation"):
                    loss = criterion(output, target)
                
                with nvtx.annotate("backward_pass"):
                    loss.backward()
                
                with nvtx.annotate("optimizer_step"):
                    optimizer.step()
    
    print("PyTorch Profiler Results:")
    print("=" * 50)
    
    print("\n1. Top operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\n2. Memory profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\n3. FLOP analysis:")
    print(prof.key_averages().table(sort_by="flops", row_limit=5))
    
    print("\n4. Module-level analysis:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5))
    
    # Export results
    prof.export_chrome_trace("pytorch_profiler_trace.json")
    print("\n✓ Chrome trace exported to pytorch_profiler_trace.json")


def demonstrate_nsight_systems():
    """Demonstrate Nsight Systems timeline analysis."""
    print("\n=== Nsight Systems Demo ===")
    
    print("Nsight Systems (nsys) Commands:")
    print("=" * 40)
    
    print("\n1. Basic timeline profiling:")
    print("nsys profile -t cuda,nvtx,osrt -o timeline_profile python script.py")
    
    print("\n2. Enhanced timeline with Python sampling:")
    print("nsys profile --force-overwrite=true -o profile_report -t cuda,nvtx,osrt,cudnn,cublas -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true python script.py")
    
    print("\n3. Multi-GPU profiling:")
    print("nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl -o multi_gpu_profile torchrun --nnodes=1 --nproc_per_node=8 script.py")
    
    print("\n4. HTA (Holistic Tracing Analysis):")
    print("nsys profile --force-overwrite=true -o hta_report -t cuda,nvtx,osrt,cudnn,cublas,nccl -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true --capture-range=cudaProfilerApi --capture-range-end=stop --capture-range-op=both --multi-gpu=all python script.py")
    
    print("\n5. Memory profiling:")
    print("nsys profile --force-overwrite=true -t cuda,cudamemcpy -o memory_profile python script.py")
    
    print("\nKey Features:")
    print("• Timeline analysis with GPU and CPU events")
    print("• Python backtrace sampling")
    print("• CUDA backtrace integration")
    print("• Multi-GPU support")
    print("• Hardware metrics collection")
    print("• Blackwell B200/B300 specific metrics")


def demonstrate_nsight_compute():
    """Demonstrate Nsight Compute kernel-level analysis."""
    print("\n=== Nsight Compute Demo ===")
    
    print("Nsight Compute (ncu) Commands:")
    print("=" * 40)
    
    print("\n1. Basic kernel profiling:")
    print("ncu --mode=launch --target-processes=python3 --set full --kernel-regex \".*\" --sampling-interval 1 --sampling-max-passes 5 --sampling-period 1000000 --export csv -o ncu_report python script.py")
    
    print("\n2. Specific kernel profiling:")
    print("ncu --kernel-name regex:gemm* -o gemm_profile python script.py")
    print("ncu --kernel-name regex:attention* -o attention_profile python script.py")
    
    print("\n3. Performance metrics:")
    print("ncu --metrics achieved_occupancy,warp_execution_efficiency -o occupancy_profile python script.py")
    print("ncu --metrics dram_read_throughput,dram_write_throughput -o memory_profile python script.py")
    
    print("\n4. Blackwell B200/B300 specific metrics:")
    print("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput,sm__throughput.avg.pct_of_peak_sustained_elapsed -o blackwell_profile python script.py")
    
    print("\n5. Comprehensive analysis:")
    print("ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput,sm__cycles_elapsed.avg,dram__bytes.sum -o comprehensive_profile python script.py")
    
    print("\nKey Features:")
    print("• SM100 architecture support")
    print("• Tensor Core metrics")
    print("• HBM3e memory analysis")
    print("• Advanced occupancy analysis")
    print("• TMA (Tensor Memory Accelerator) metrics")


def demonstrate_hta_profiling():
    """Demonstrate HTA (Holistic Tracing Analysis) for multi-GPU."""
    print("\n=== HTA (Holistic Tracing Analysis) Demo ===")
    
    print("HTA Commands:")
    print("=" * 30)
    
    print("\n1. Basic HTA profiling:")
    print("nsys profile --force-overwrite=true -o hta_report -t cuda,nvtx,osrt,cudnn,cublas,nccl -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true python script.py")
    
    print("\n2. Multi-GPU HTA:")
    print("nsys profile --force-overwrite=true -o hta_multi_gpu -t cuda,nvtx,osrt,cudnn,cublas,nccl -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true --capture-range=cudaProfilerApi --capture-range-end=stop --capture-range-op=both --multi-gpu=all python script.py")
    
    print("\n3. Distributed training HTA:")
    print("nsys profile --force-overwrite=true -o hta_distributed -t cuda,nvtx,osrt,cudnn,cublas,nccl -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true torchrun --nnodes=1 --nproc_per_node=8 script.py")
    
    print("\nKey Features:")
    print("• NCCL communication analysis")
    print("• Cross-GPU synchronization")
    print("• Load balancing analysis")
    print("• Memory transfer optimization")
    print("• NVLink-C2C analysis")


def demonstrate_perf_profiling():
    """Demonstrate Perf system-level analysis."""
    print("\n=== Perf Profiling Demo ===")
    
    print("Perf Commands:")
    print("=" * 20)
    
    print("\n1. Basic system profiling:")
    print("perf record -g -p $(pgrep python) -o perf.data")
    print("perf report -i perf.data")
    
    print("\n2. CPU profiling:")
    print("perf record -e cpu-cycles -g -p $(pgrep python) -o cpu_perf.data")
    print("perf report -i cpu_perf.data")
    
    print("\n3. Memory profiling:")
    print("perf record -e cache-misses -g -p $(pgrep python) -o memory_perf.data")
    print("perf report -i memory_perf.data")
    
    print("\n4. System-wide profiling:")
    print("perf record -a -g -o system_perf.data")
    print("perf report -i system_perf.data")
    
    print("\nKey Features:")
    print("• CPU performance analysis")
    print("• System call analysis")
    print("• Hardware event monitoring")
    print("• Call graph analysis")


def demonstrate_memory_profiling():
    """Demonstrate comprehensive memory profiling."""
    print("\n=== Memory Profiling Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Memory Profiling Techniques:")
    print("=" * 40)
    
    print("\n1. PyTorch Memory Profiling:")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        print(f"Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Current cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Peak cached: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        
        # Memory stats
        memory_stats = torch.cuda.memory_stats()
        print(f"Number of allocations: {memory_stats.get('num_alloc_retries', 0)}")
        print(f"Number of OOM: {memory_stats.get('num_ooms', 0)}")
    
    print("\n2. Nsight Systems Memory Profiling:")
    print("nsys profile --force-overwrite=true -t cuda,cudamemcpy -o memory_profile python script.py")
    
    print("\n3. Nsight Compute Memory Metrics:")
    print("ncu --metrics dram_read_throughput,dram_write_throughput,dram__bytes.sum -o memory_metrics python script.py")
    
    print("\n4. Memory Snapshot:")
    if torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(True)
        snapshot = torch.cuda.memory._snapshot()
        print(f"Memory snapshot contains {len(snapshot)} entries")
        torch.cuda.memory._record_memory_history(False)


def demonstrate_comprehensive_profiling():
    """Demonstrate comprehensive profiling with all tools."""
    print("\n=== Comprehensive Profiling Demo ===")
    
    print("Comprehensive Profiling Workflow:")
    print("=" * 50)
    
    print("\n1. Start with Nsight Systems:")
    print("   Get system-level overview")
    print("   nsys profile -t cuda,nvtx,osrt -o timeline_profile python script.py")
    
    print("\n2. Use PyTorch Profiler:")
    print("   Identify framework bottlenecks")
    print("   Enhanced profiler with memory, FLOPs, modules")
    
    print("\n3. Run Nsight Compute:")
    print("   Analyze specific kernels")
    print("   ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py")
    
    print("\n4. Enable HTA for Multi-GPU:")
    print("   Analyze distributed performance")
    print("   nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile python script.py")
    
    print("\n5. Monitor Memory:")
    print("   Track memory usage patterns")
    print("   nsys profile -t cuda,cudamemcpy -o memory_profile python script.py")
    
    print("\n6. Use Perf:")
    print("   System-level analysis")
    print("   perf record -g -p $(pgrep python) -o perf.data")
    
    print("\n7. Comprehensive Analysis:")
    print("   Combine all tools for complete picture")
    print("   Export results for detailed analysis")


def demonstrate_blackwell_specific_profiling():
    """Demonstrate Blackwell B200/B300 specific profiling features."""
    print("\n=== Blackwell B200/B300 Specific Profiling ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell profiling")
        return
    
    device_props = torch.cuda.get_device_properties(0)
    is_blackwell = device_props.major >= 10
    
    if is_blackwell:
        print("Blackwell B200/B300 Profiling Features:")
        print("=" * 50)
        
        print("\n1. HBM3e Memory Profiling:")
        print("• High-bandwidth memory analysis")
        print("• Memory bandwidth optimization")
        print("• Unified memory profiling")
        
        print("\n2. Tensor Core Profiling:")
        print("• 4th Generation Tensor Core metrics")
        print("• FP8/FP4 precision analysis")
        print("• Matrix operation optimization")
        
        print("\n3. TMA (Tensor Memory Accelerator) Profiling:")
        print("• Advanced memory access patterns")
        print("• Memory latency analysis")
        print("• Data movement optimization")
        
        print("\n4. NVLink-C2C Profiling:")
        print("• Direct GPU-to-GPU communication")
        print("• Multi-GPU synchronization")
        print("• Communication overhead analysis")
        
        print("\n5. SM100 Architecture Profiling:")
        print("• Compute Capability 10.0 features")
        print("• Advanced occupancy analysis")
        print("• Kernel optimization for SM100")
        
        print("\nProfiling Commands:")
        print("• ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed -o blackwell_throughput python script.py")
        print("• nsys profile -t cuda,nvtx,osrt -o blackwell_timeline python script.py")
        print("• ncu --metrics dram_read_throughput,dram_write_throughput -o blackwell_memory python script.py")
    else:
        print("This GPU does not support Blackwell B200/B300 specific profiling")


def demonstrate_profiling_automation():
    """Demonstrate automated profiling workflows."""
    print("\n=== Automated Profiling Workflows ===")
    
    print("Automated Profiling Scripts:")
    print("=" * 40)
    
    print("\n1. Basic Profiling Script:")
    script_content = '''#!/bin/bash
# Basic profiling script
echo "Running basic profiling..."

# PyTorch profiler
python -c "
import torch
from torch.profiler import profile, ProfilerActivity
import torch.cuda.nvtx as nvtx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Linear(1024, 1024).to(device)
x = torch.randn(32, 1024, device=device)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, with_flops=True, profile_memory=True) as prof:
    for _ in range(100):
        with nvtx.annotate('forward'):
            output = model(x)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
"

# Nsight Systems
nsys profile -t cuda,nvtx,osrt -o basic_timeline python script.py

# Nsight Compute
ncu --metrics achieved_occupancy,warp_execution_efficiency -o basic_kernel python script.py

echo "Basic profiling complete!"
'''
    
    print(script_content)
    
    print("\n2. Comprehensive Profiling Script:")
    comprehensive_script = '''#!/bin/bash
# Comprehensive profiling script
echo "Running comprehensive profiling..."

# HTA profiling
nsys profile --force-overwrite=true -o hta_report -t cuda,nvtx,osrt,cudnn,cublas,nccl -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true python script.py

# Memory profiling
nsys profile --force-overwrite=true -t cuda,cudamemcpy -o memory_profile python script.py

# Kernel profiling
ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel python script.py

# System profiling
perf record -g -p $(pgrep python) -o perf.data
perf report -i perf.data > perf_report.txt

echo "Comprehensive profiling complete!"
'''
    
    print(comprehensive_script)


def main():
    """Main function to demonstrate comprehensive profiling."""
    print("=== Comprehensive Profiling Tools Demo ===")
    print("Latest profiling tools for PyTorch 2.8, CUDA 12.9, and Blackwell B200/B300")
    print("=" * 80)
    
    # Configure Blackwell optimizations
    configure_blackwell_optimizations()
    
    # Run demonstrations
    demonstrate_pytorch_profiler()
    demonstrate_nsight_systems()
    demonstrate_nsight_compute()
    demonstrate_hta_profiling()
    demonstrate_perf_profiling()
    demonstrate_memory_profiling()
    demonstrate_comprehensive_profiling()
    demonstrate_blackwell_specific_profiling()
    demonstrate_profiling_automation()
    
    print("\n=== Summary ===")
    print("This demo shows comprehensive profiling capabilities:")
    print("1. PyTorch Profiler with enhanced features")
    print("2. Nsight Systems timeline analysis")
    print("3. Nsight Compute kernel-level analysis")
    print("4. HTA for multi-GPU analysis")
    print("5. Perf system-level analysis")
    print("6. Memory profiling techniques")
    print("7. Blackwell B200/B300 specific features")
    print("8. Automated profiling workflows")
    print("9. Comprehensive analysis tools")
    print("10. Latest profiling tool integration")
    
    print("\nKey Benefits:")
    print("• Complete performance analysis")
    print("• Multi-level profiling (system, framework, kernel)")
    print("• Blackwell B200/B300 optimizations")
    print("• Memory and bandwidth analysis")
    print("• Automated profiling workflows")
    print("• Latest tool integration")


if __name__ == "__main__":
    main()
