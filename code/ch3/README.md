# Chapter 3: OS, Docker, and Kubernetes Tuning for GPU-based Environments

This directory contains code examples and configuration files for optimizing operating systems, containers, and orchestration platforms for GPU-based AI workloads.

## Files Overview

### Core Scripts and Utilities

- **`bind_numa_affinity.py`** - Python script demonstrating NUMA-aware process binding for PyTorch training with automatic GPU-to-NUMA-node detection
- **`numa_topology_script.sh`** - Shell script for dynamically binding training processes to appropriate NUMA nodes based on GPU topology
- **`gpu_setup_commands.sh`** - Collection of system commands for GPU performance optimization
- **`system_tuning.sh`** - Comprehensive system-wide performance tuning script

### Container and Orchestration

- **`docker_gpu_optimized.dockerfile`** - Multi-stage Docker build optimized for GPU performance with PyTorch 2.8 and CUDA 12.9
- **`kubernetes_mig_pod.yaml`** - Kubernetes pod configuration for Multi-Instance GPU (MIG) workloads
- **`kubernetes_topology_pod.yaml`** - Advanced Kubernetes pod with topology awareness and NUMA affinity

### Dependencies

- **`requirements.txt`** - Python dependencies including PyTorch 2.8, Triton 3.4, and performance monitoring tools

## Key Optimizations Covered

### NUMA Awareness and CPU Affinity
- Automatic detection of GPU-to-NUMA-node mapping
- CPU and memory binding to reduce cross-NUMA latency
- Worker process affinity for data loading

### GPU Driver Optimizations
- Persistence mode for reduced startup latency
- Multi-Process Service (MPS) for concurrent GPU sharing
- Multi-Instance GPU (MIG) partitioning for isolation

### Memory Optimizations
- Pinned (page-locked) memory for faster GPU transfers
- Transparent Huge Pages configuration
- Memory allocator tuning (jemalloc, tcmalloc)

### Container Runtime
- NVIDIA Container Toolkit integration
- Host networking for RDMA/InfiniBand access
- Volume mounts to bypass overlay filesystem overhead

### Kubernetes Orchestration
- Topology Manager for hardware-aware scheduling
- GPU Operator for automated driver management
- Resource isolation and QoS guarantees

## Usage Examples

### Basic NUMA Binding
```bash
# Run training with automatic NUMA binding
python bind_numa_affinity.py

# Manual NUMA binding for specific GPU
numactl --cpunodebind=1 --membind=1 python train.py --gpu 4
```

### Docker with GPU Support
```bash
# Build optimized container
docker build -f docker_gpu_optimized.dockerfile -t gpu-optimized .

# Run with host networking and resource binding
docker run --gpus all --privileged \
    --network=host \
    --ulimit memlock=-1 \
    --cpuset-cpus="0-7" \
    gpu-optimized
```

### Kubernetes Deployment
```bash
# Deploy MIG-enabled pod
kubectl apply -f kubernetes_mig_pod.yaml

# Deploy topology-aware distributed training
kubectl apply -f kubernetes_topology_pod.yaml
```

### System Tuning
```bash
# Apply system-wide optimizations (run as root)
sudo bash system_tuning.sh

# Enable GPU persistence mode
sudo nvidia-smi -pm ENABLED
```

## Hardware Requirements

- NVIDIA GPUs with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Blackwell)
- CUDA 12.9+ compatible drivers
- Linux kernel with NUMA support
- InfiniBand/Ethernet networking for multi-node setups

## Software Stack

- **PyTorch 2.8** with CUDA 12.9 support
- **Triton 3.4** for custom kernel development
- **NVIDIA Container Toolkit** for containerized GPU access
- **Kubernetes 1.28+** with GPU device plugin
- **NVIDIA GPU Operator** for automated management

## Performance Impact

These optimizations typically provide:
- 5-15% training throughput improvement through NUMA awareness
- 10-20% faster data loading with pinned memory
- 2-5% memory access speedup with huge pages
- Near-zero container overhead (~1-2%) with proper configuration
- Linear scaling up to 72 GPUs within NVLink domains

## Monitoring and Validation

Use these tools to verify optimizations:
```bash
# Check NUMA binding
numactl --show
cat /proc/self/numa_maps

# Monitor GPU utilization
nvidia-smi dmon
nvidia-smi topo -m

# Check memory usage
cat /proc/meminfo | grep -i huge
torch.cuda.memory_summary()

# Network bandwidth testing
ib_write_bw  # InfiniBand
iperf3       # Ethernet
```

## Troubleshooting

Common issues and solutions:
- **Cross-NUMA memory access**: Verify CPU and memory binding with `numactl --show`
- **Container GPU access**: Ensure NVIDIA Container Toolkit is installed and configured
- **Memory allocation failures**: Check `ulimit -l` and increase locked memory limits
- **Network performance**: Verify RDMA drivers and test with bandwidth utilities
- **Kubernetes scheduling**: Check node labels and device plugin logs

For more detailed information, refer to Chapter 3 of the AI Performance Engineering book.
