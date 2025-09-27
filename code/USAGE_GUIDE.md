# AI Systems Performance Engineering – Usage Guide

This guide documents all the profiling scripts and command examples for the latest CUDA 12.9, PyTorch 2.9 nightly, Triton 3.4, and dedicated Blackwell B200/B300 GPU support.

## Chapter Overview

The book is organized into 20 chapters covering all aspects of AI systems performance engineering:

### Foundation Chapters (1-5)
- **Chapter 1**: Introduction to AI Systems Performance Engineering
- **Chapter 2**: AI System Hardware Overview  
- **Chapter 3**: NUMA Affinity and System Optimization
- **Chapter 4**: Communication-Computation Overlap and Distributed Training
- **Chapter 5**: GPU-based Storage I/O Optimizations

### Core CUDA Chapters (6-10)
- **Chapter 6**: Basic CUDA Kernels and Occupancy
- **Chapter 7**: Shared-Memory Tiling and Kernel Optimization
- **Chapter 8**: Kernel Scheduling and Orchestration
- **Chapter 9**: Dynamic Multi-Kernel Orchestration and Graph Scheduling
- **Chapter 10**: Advanced CUDA Features (Graphs, Dynamic Parallelism, Unified Memory)

### Framework Chapters (11-13)
- **Chapter 11**: PyTorch Optimization and Compilation
- **Chapter 12**: Inference GPU Cluster Sizing
- **Chapter 13**: PyTorch Profiling of Large Models

### Advanced Topics (14-17)
- **Chapter 14**: Advanced CUDA Programming and Optimization
- **Chapter 15**: Model Serving and Inference Optimization
- **Chapter 16**: Agentic AI Systems and Multi-Agent Optimization
- **Chapter 17**: Large-Scale Model Training Optimization

### System and Future Chapters (18-20)
- **Chapter 18**: System-Level Performance Monitoring and Optimization
- **Chapter 19**: Emerging Technologies and Future Trends
- **Chapter 20**: Case Studies and Real-World Applications

## Quick Start

### Prerequisites

```bash
# Install latest dependencies
pip install -r requirements_latest.txt

# Verify CUDA installation
nvcc --version  # Should show CUDA 12.9
python -c "import torch; print(torch.version.cuda)"  # Should show 12.9

# Check architecture support
python arch_config.py  # Shows current architecture
```

### Environment Setup

```bash
# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_PLUGIN=ucx  # For RDMA networks
```

## Chapter-by-Chapter Usage

### Chapter 1: Introduction to AI Systems Performance Engineering

**Purpose**: Foundation concepts, performance measurement, and system characterization.

```bash
cd code/ch1

# Basic performance measurement
python performance_basics.py
```

**Profiling Commands**:
```bash
# Profile basic performance measurement
nsys profile -t cuda,osrt -o basic_performance_profile python performance_basics.py
```

**Expected Output**:
```
GPU: NVIDIA B200/B300
Compute Capability: 10.0
Memory: 192GB/288GB HBM3e
Memory Bandwidth: 3.2 TB/s
Peak FP32 Performance: 4.0 PFLOPS
GPU Utilization: 95.2%
Memory Utilization: 78.4%
```

### Chapter 2: AI System Hardware Overview

**Purpose**: Deep dive into Blackwell B200/B300 architecture and hardware optimization.

```bash
cd code/ch2

# Hardware information and benchmarking
python hardware_info.py
```

**Profiling Commands**:
```bash
# Profile hardware information
nsys profile -t cuda,osrt -o hardware_info_profile python hardware_info.py
```

**Expected Output**:
```
NVIDIA Blackwell B200/B300 Information:
GPU: B200/B300
Compute Capability: 10.0 (SM100)
Memory: 192GB/288GB HBM3e
Memory Bandwidth: 3.2 TB/s
Tensor Cores: 4th Generation
TMA: Tensor Memory Accelerator
NVLink-C2C: Direct GPU-to-GPU communication
```

### Chapter 3: NUMA Affinity

**Purpose**: Bind processes to NUMA nodes local to GPUs to minimize cross-NUMA traffic.

```bash
cd code/ch3

# NUMA-aware launch script
bash numa_bind.sh

# Programmatic NUMA binding
python bind_numa_affinity.py
```

**Profiling Commands**:
```bash
# Profile NUMA binding effectiveness
nsys profile -t cuda,osrt -o numa_binding_profile bash numa_bind.sh
```

**Expected Output**:
```
PID=[X] bound to NUMA node [Y] (CPUs=[Z])
Worker [X] (PID=[Y]) bound to NUMA node [Z]
```

### Chapter 4: Overlap AllReduce

**Purpose**: Demonstrate communication-computation overlap and distributed training optimizations.

```bash
cd code/ch4

# Gradient overlap comparison
python after_ddp.py
python after_overlap_ddp.py
```

**Profiling Commands**:
```bash
# Profile overlap vs no-overlap timeline
nsys profile -t cuda,nvtx -o overlap_ddp_profile python after_overlap_ddp.py
```

**Expected Output**:
```
# With DDP overlap
Forward time: [X] ms
Backward + All-reduce time: [Y] ms
Total iteration: [Z] ms
```

### Chapter 5: GPU-based Storage I/O Optimizations

**Purpose**: Optimize storage I/O for AI workloads including GPUDirect Storage.

```bash
cd code/ch5

# Storage optimization examples
python gpudirect_storage_example.py
```

**Profiling Commands**:
```bash
# Profile storage I/O performance
nsys profile -t cuda,osrt -o storage_profile python gpudirect_storage_example.py
```

**Expected Output**:
```
Storage I/O Performance:
Sequential Read: [X] GB/s
Random Read: [Y] GB/s
GPUDirect Storage: [Z] GB/s
```

### Chapter 6: CUDA Basics

**Purpose**: Demonstrate basic CUDA kernels, occupancy, and parallelism principles.

```bash
cd code/ch6

# Compile CUDA programs
make clean && make

# Run CUDA programs
./add_parallel
./add_sequential

# Run Python examples
python add_parallel.py
python add_sequential.py
```

**Profiling Commands**:
```bash
# Profile kernel execution timelines
nsys profile -t cuda,osrt -o parallel_add_profile ./add_parallel
nsys profile -t cuda,osrt -o sequential_add_profile ./add_sequential

# Profile kernel efficiency and occupancy
ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile ./add_parallel
```

**Expected Output**:
```
# Sequential kernel
Sequential addition completed
Time: [X] ms (1 thread, low occupancy)

# Parallel kernel
Parallel addition completed
Time: [Y] ms ([Z] threads, high occupancy)
```

### Chapter 7: Matmul Tiling

**Purpose**: Demonstrate shared-memory tiling for improved matrix multiplication performance.

```bash
cd code/ch7

# Compile CUDA program
make clean && make

# Run examples
python matmul_pytorch.py
./tiled_matmul
```

**Profiling Commands**:
```bash
# Profile naive vs tiled implementations
nsys profile -t cuda,osrt -o tiled_matmul_profile ./tiled_matmul
ncu --metrics achieved_occupancy,warp_execution_efficiency -o matmul_kernel_profile ./tiled_matmul
```

**Expected Output**:
```
# Tiled implementation
Tiled matrix multiplication completed
Matrix size: 1024x1024
Time: [X] ms
Performance: [Y] TFLOPS
```

### Chapter 10: Advanced CUDA Features

**Purpose**: Demonstrate advanced CUDA features including CUDA Graphs, Dynamic Parallelism, Unified Memory, and Stream-ordered Memory.

```bash
cd code/ch10

# Compile CUDA examples
make clean && make

# Run CUDA examples
./cuda_graphs
./dynamic_parallelism
./unified_memory
./stream_ordered_memory
```

**Profiling Commands**:
```bash
# Profile CUDA Graph execution
nsys profile -t cuda,osrt -o cuda_graphs_profile ./cuda_graphs

# Profile dynamic parallelism
nsys profile -t cuda,osrt -o dynamic_parallelism_profile ./dynamic_parallelism

# Profile unified memory
nsys profile -t cuda,osrt -o unified_memory_profile ./unified_memory

# Profile stream-ordered memory
nsys profile -t cuda,osrt -o stream_ordered_memory_profile ./stream_ordered_memory
```

**Expected Output**:
```
# CUDA Graphs
CUDA Graph created successfully
Graph execution time: [X] ms
Regular execution time: [Y] ms
Speedup: [Z]x

# Stream-ordered Memory
Stream-ordered allocation: [X] MB
Allocation time: [Y] ms
Deallocation time: [Z] ms
```

### Chapter 11: PyTorch Optimization

**Purpose**: Demonstrate PyTorch-specific optimizations and techniques.

```bash
cd code/ch11

# Compile CUDA examples
make clean && make

# Run examples
./basic_streams
./stream_ordered_allocator
./multi_stream_pipeline
```

**Profiling Commands**:
```bash
# Profile stream examples
nsys profile -t cuda,osrt -o basic_streams_profile ./basic_streams
nsys profile -t cuda,osrt -o stream_alloc_profile ./stream_ordered_allocator
nsys profile -t cuda,osrt -o multi_stream_profile ./multi_stream_pipeline
```

**Expected Output**:
```
# Stream examples
Stream overlap achieved
Memory allocation optimized
Pipeline efficiency: [X]%
```

### Chapter 13: PyTorch Profiling

**Purpose**: Profile large transformer models and identify performance bottlenecks.

```bash
cd code/ch13

# Basic profile run
python train_deepseek_v3.py

# Memory profiling
python memory_profiling.py

# Custom allocator
python custom_allocator.py

# FSDP example
python fsdp_example.py
```

**Profiling Commands**:
```bash
# Profile with NVTX markers and GPU timeline (generate stats after capture)
nsys profile -t cuda,nvtx -o deepseek_v3_profile python train_deepseek_v3.py
nsys stats --report summary,nvtx_gpu_proj_sum,cuda_api --format csv,sqlite deepseek_v3_profile -o deepseek_v3_profile

# Profile specific kernels
ncu --target-processes all --kernel-name regex:gemm* -o deepseek_gemm_profile python train_deepseek_v3.py

# Profile with detailed metrics
ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed -o deepseek_detailed_profile python train_deepseek_v3.py
```

**Expected Output**:
```
PyTorch Profiler Results:
----------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
                                   Name    Self CPU %      Self CPU Avg     CPU total %      CPU total Avg    CPU time avg     CUDA total %     CUDA total Avg    CUDA time avg       Number of Calls  
----------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  
                            aten::matmul            0.0%           0.000ms            0.0%           0.000ms           0.000ms           43.2%          43.200ms          43.200ms                   128  
                            aten::linear            0.0%           0.000ms            0.0%           0.000ms           0.000ms           28.5%          28.500ms          28.500ms                   256  
```

### Chapter 14: PyTorch Compiler

**Purpose**: Demonstrate torch.compile optimizations and Triton integration.

```bash
cd code/ch14

# PyTorch compiler examples
python torch_compiler_examples.py

# Triton examples
python triton_examples.py
```

**Profiling Commands**:
```bash
# Profile torch.compile examples
nsys profile -t cuda,osrt -o torch_compiler_profile python torch_compiler_examples.py

# Profile Triton examples
nsys profile -t cuda,osrt -o triton_profile python triton_examples.py
```

**Expected Output**:
```
# torch.compile
Compiled model performance: [X] ms
Eager model performance: [Y] ms
Speedup: [Z]x

# Triton kernels
Custom kernel performance: [X] ms
PyTorch kernel performance: [Y] ms
Speedup: [Z]x
```

### Chapter 15: Model Serving

**Purpose**: High-performance model serving and inference optimization.

```bash
cd code/ch15

# Disaggregated inference
python disaggregated_inference.py
```

**Profiling Commands**:
```bash
# Profile inference performance
nsys profile -t cuda,osrt -o inference_profile python disaggregated_inference.py
```

**Expected Output**:
```
Inference Performance:
Throughput: [X] tokens/sec
Latency: [Y] ms
Memory usage: [Z] MB
```

### Chapter 16: Advanced Profiling

**Purpose**: Advanced profiling techniques and performance analysis.

```bash
cd code/ch16

# Inference profiling
python inference_profiling.py

# NVTX profiling
make clean && make
./nvtx_profiling

# Radix attention
python radix_attention_example.py
```

**Profiling Commands**:
```bash
# Profile inference profiling
nsys profile -t cuda,osrt -o inference_profiling_profile python inference_profiling.py

# Profile NVTX profiling
nsys profile -t cuda,nvtx -o nvtx_profile ./nvtx_profiling

# Profile radix attention
nsys profile -t cuda,osrt -o radix_attention_profile python radix_attention_example.py
```

**Expected Output**:
```
# NVTX profiling
NVTX markers added
Performance analysis complete
Bottlenecks identified: [X, Y, Z]
```

### Chapter 8: Kernel Optimization

**Purpose**: Advanced kernel optimization, occupancy tuning, and instruction-level parallelism.

```bash
cd code/ch8

# Compile CUDA examples
make clean && make

# Run examples
./independent_ops
./loop_unrolling
./threshold_naive
./threshold_predicated
./occupancy_api_example
./occupancy_tuning

# Run Python examples
python warp_divergence_pytorch.py
python occupancy_pytorch.py
python ilp_pytorch.py
```

**Profiling Commands**:
```bash
# Profile kernel optimization
nsys profile -t cuda,osrt -o independent_ops_profile ./independent_ops
nsys profile -t cuda,osrt -o loop_unrolling_profile ./loop_unrolling
nsys profile -t cuda,osrt -o threshold_profile ./threshold_predicated

# Profile occupancy
ncu --metrics achieved_occupancy,warp_execution_efficiency -o occupancy_profile ./occupancy_tuning

# Profile Python examples
nsys profile -t cuda,osrt -o warp_divergence_profile python warp_divergence_pytorch.py
nsys profile -t cuda,osrt -o occupancy_pytorch_profile python occupancy_pytorch.py
nsys profile -t cuda,osrt -o ilp_pytorch_profile python ilp_pytorch.py
```

**Expected Output**:
```
# Kernel optimization
Independent operations completed
Time: [X] ms
Performance: [Y] GFLOPS

# Occupancy tuning
Optimal occupancy achieved: [X]%
Thread block size: [Y]x[Z]
```

### Chapter 9: Advanced Kernel Fusion

**Purpose**: Demonstrate kernel fusion techniques and advanced CUDA optimizations.

```bash
cd code/ch9

# Compile CUDA examples
make clean && make

# Run examples
./cutlass_gemm_example
./fusedL2Norm
./fused_l2norm
./inline_ptx_example

# Run Python examples
python fusion_pytorch.py
```

**Profiling Commands**:
```bash
# Profile kernel fusion
nsys profile -t cuda,osrt -o cutlass_gemm_profile ./cutlass_gemm_example
nsys profile -t cuda,osrt -o fused_l2norm_profile ./fusedL2Norm
nsys profile -t cuda,osrt -o inline_ptx_profile ./inline_ptx_example

# Profile Python fusion
nsys profile -t cuda,osrt -o fusion_pytorch_profile python fusion_pytorch.py
```

**Expected Output**:
```
# CUTLASS GEMM
CUTLASS GEMM completed
Matrix size: 1024x1024
Time: [X] ms
Performance: [Y] TFLOPS

# Fused L2Norm
Fused L2Norm completed
Time: [X] ms
Speedup: [Y]x over separate kernels
```

### Chapter 12: Advanced CUDA Features

**Purpose**: Demonstrate advanced CUDA features including atomic work queues, CUDA Graphs, and dynamic parallelism.

```bash
cd code/ch12

# Compile CUDA examples
make clean && make

# Run examples
./atomic_work_queue
./cuda_graphs
./dynamic_parallelism
```

**Profiling Commands**:
```bash
# Profile atomic work queue
nsys profile -t cuda,osrt -o atomic_queue_profile ./atomic_work_queue

# Profile CUDA Graphs
nsys profile -t cuda,osrt -o cuda_graphs_profile ./cuda_graphs

# Profile dynamic parallelism
nsys profile -t cuda,osrt -o dynamic_parallelism_profile ./dynamic_parallelism
```

**Expected Output**:
```
# Atomic work queue
Work queue processing completed
Tasks processed: [X]
Time: [Y] ms
Throughput: [Z] tasks/sec

# CUDA Graphs
Graph execution time: [X] ms
Regular execution time: [Y] ms
Speedup: [Z]x

# Dynamic parallelism
Dynamic parallelism completed
Parent threads: [X]
Child threads: [Y]
Total work: [Z]
```

### Chapter 17: Dynamic Routing

**Purpose**: Demonstrate dynamic routing and early rejection techniques for AI systems.

```bash
cd code/ch17

# Dynamic routing example
python dynamic_routing.py

# Early rejection example
python early_rejection.py
```

**Profiling Commands**:
```bash
# Profile dynamic routing
nsys profile -t cuda,osrt -o dynamic_routing_profile python dynamic_routing.py

# Profile early rejection
nsys profile -t cuda,osrt -o early_rejection_profile python early_rejection.py
```

**Expected Output**:
```
# Dynamic routing
Dynamic routing completed
Routes processed: [X]
Average latency: [Y] ms
Throughput: [Z] routes/sec

# Early rejection
Early rejection completed
Tokens processed: [X]
Rejected tokens: [Y]
Rejection rate: [Z]%
```

### Chapter 18: FlashMLA and Optimization

**Purpose**: Demonstrate FlashMLA kernels and advanced optimization techniques.

```bash
cd code/ch18

# Compile FlashMLA kernel
make clean && make

# Run FlashMLA example
./flashmla_kernel

# Run FlexDecoding example
python flexdecoding_example.py
```

**Profiling Commands**:
```bash
# Profile FlashMLA kernel
nsys profile -t cuda,osrt -o flashmla_profile ./flashmla_kernel

# Profile FlexDecoding
nsys profile -t cuda,osrt -o flexdecoding_profile python flexdecoding_example.py
```

**Expected Output**:
```
# FlashMLA kernel
FlashMLA kernel completed
Matrix size: [X]x[Y]
Time: [Z] ms
Performance: [W] TFLOPS

# FlexDecoding
FlexDecoding completed
Tokens generated: [X]
Average latency: [Y] ms
Throughput: [Z] tokens/sec
```

### Chapter 19: Dynamic Parallelism

**Purpose**: Demonstrate dynamic parallelism and token precision switching.

```bash
cd code/ch19

# Dynamic parallelism example
python dynamic_parallelism.py

# Token precision switching
python token_precision_switch.py
```

**Profiling Commands**:
```bash
# Profile dynamic parallelism
nsys profile -t cuda,osrt -o dynamic_parallelism_profile python dynamic_parallelism.py

# Profile token precision switching
nsys profile -t cuda,osrt -o precision_switch_profile python token_precision_switch.py
```

**Expected Output**:
```
# Dynamic parallelism
Dynamic parallelism completed
Workload distributed: [X] tasks
Execution time: [Y] ms
Efficiency: [Z]%

# Token precision switching
Precision switching completed
FP16 tokens: [X]
FP32 tokens: [Y]
Mixed precision efficiency: [Z]%
```

### Chapter 20: AI Kernel Generation

**Purpose**: Demonstrate AI-assisted kernel generation and optimization.

```bash
cd code/ch20

# AI kernel generator
python ai_kernel_generator.py
```

**Profiling Commands**:
```bash
# Profile AI kernel generation
nsys profile -t cuda,osrt -o ai_kernel_profile python ai_kernel_generator.py
```

**Expected Output**:
```
# AI kernel generation
AI kernel generation completed
Kernels generated: [X]
Optimization iterations: [Y]
Performance improvement: [Z]%
Generated kernel performance: [W] GFLOPS
```

## Advanced Profiling Scripts

### Nsight Systems (nsys)

**Basic Timeline Profiling**:
```bash
nsys profile -t cuda,osrt -o timeline_profile your_script.py
```

**With NVTX Markers**:
```bash
nsys profile -t cuda,nvtx -o nvtx_profile your_script.py
```

**Multi-GPU Profiling**:
```bash
nsys profile -t cuda,nvtx,nccl -o multi_gpu_profile torchrun --nnodes=1 --nproc_per_node=8 your_script.py
```

### Nsight Compute (ncu)

**Kernel-Specific Profiling**:
```bash
ncu --kernel-name regex:gemm* -o gemm_profile your_script.py
ncu --kernel-name regex:attention* -o attention_profile your_script.py
```

**Performance Metrics**:
```bash
ncu --metrics achieved_occupancy,warp_execution_efficiency -o occupancy_profile your_script.py
ncu --metrics dram_read_throughput,dram_write_throughput -o memory_profile your_script.py
```

### PyTorch Profiler

**Basic Profiling**:
```bash
python -m torch.utils.bottleneck your_script.py
```

**Advanced Profiling**:
```bash
python -c "
import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

## Performance Tuning Tips

### General Optimization

1. **Memory Management**:
   - Use `cudaMallocAsync` for stream-ordered allocation
   - Monitor memory fragmentation with `torch.cuda.memory_stats()`
   - Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

2. **Communication Optimization**:
   - Use NCCL backend for GPU-direct communication
   - Enable UCX for RDMA networks: `NCCL_P2P_PLUGIN=ucx`
   - Overlap communication with computation using DDP

3. **Kernel Optimization**:
   - Aim for high occupancy (80%+)
   - Use coalesced memory access patterns
   - Leverage shared memory for data reuse

### Blackwell B200/B300 Considerations

1. **Architecture Features**:
   - Compute Capability: SM100 (10.0)
   - HBM3e Memory: Up to 3.2TB/s bandwidth
   - Tensor Cores: 4th Generation
   - TMA: Tensor Memory Accelerator
   - Multi-GPU: Direct GPU-to-GPU communication

2. **CUDA 12.9 Optimizations**:
   - Stream-ordered memory allocation
   - CUDA Graphs for kernel replay
   - Unified Memory with HBM3e
   - TMA for advanced memory access

### Multi-GPU Considerations

1. **Load Balancing**:
   - Distribute work evenly across GPUs
   - Use `torch.distributed` for communication
   - Monitor GPU utilization with `nvidia-smi`

2. **Communication Patterns**:
   - Profile NCCL operations
   - Check network bandwidth
   - Use `nsys` with `-t cuda,nvtx,nccl`

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**:
   ```bash
  nvcc --version  # Should show CUDA 12.9
  python -c "import torch; print(torch.version.cuda)"  # Should show 12.9
   ```

2. **Memory Issues**:
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Monitor memory usage
   python -c "import torch; print(torch.cuda.memory_allocated()/1e6)"
   ```

3. **Profiler Overhead**:
   - Use sampling mode for production runs
   - Limit profiling duration
   - Use `--capture-range` for specific sections

4. **Multi-GPU Issues**:
   ```bash
   # Check GPU topology
   nvidia-smi topo -m
   
   # Verify NCCL setup
   python -c "import torch.distributed as dist; print(dist.is_nccl_available())"
   ```

### Performance Debugging

1. **Low Occupancy**:
   - Check thread block size
   - Verify shared memory usage
   - Profile with `ncu --metrics achieved_occupancy`

2. **Memory Bandwidth**:
   - Check memory access patterns
   - Use coalesced access
   - Profile with `ncu --metrics dram_read_throughput`

3. **Communication Bottlenecks**:
   - Profile NCCL operations
   - Check network bandwidth
   - Use `nsys` with `-t cuda,nvtx,nccl`

## Architecture-Specific Notes

### Blackwell B200/B300 with Grace CPU

- **NUMA Nodes**: 2-4 per socket
- **GPU Affinity**: Each GPU local to one NUMA node
- **Memory Bandwidth**: HBM3e ~3.2TB/s per GPU
- **Grace CPU**: ARM-based with high GPU bandwidth

### CUDA 12.9 Features

- **Stream-ordered Memory**: `cudaMallocAsync`/`cudaFreeAsync`
- **CUDA Graphs**: Capture and replay kernel sequences
- **Multi-GPU**: Direct GPU-to-GPU memory access
- **Unified Memory**: HBM3e provides faster CPU-GPU access
- **TMA**: Tensor Memory Accelerator for efficient data movement

This comprehensive profiling setup provides the tools needed to achieve maximum performance on the latest Blackwell B200/B300 architecture with CUDA 12.9, PyTorch 2.9 nightly, and Triton 3.4.
