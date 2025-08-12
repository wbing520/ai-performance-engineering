# 🚀 AI Performance Engineering

A comprehensive guide to optimizing AI systems for maximum performance, efficiency, and scalability. This repository contains practical examples and code for performance engineering on modern AI hardware, including NVIDIA's Grace Blackwell superchips and NVL72 systems.

## 🚀 Overview

This repository supports the latest AI hardware and software stack with architecture switching:

- **PyTorch 2.8**: Latest PyTorch with enhanced compiler support and architecture-specific optimizations
- **CUDA 12.8**: Latest CUDA toolkit with Hopper H100/H200 and Blackwell B200/B300 support
- **Triton 3.3**: OpenAI's Triton for custom GPU kernel development
- **Architecture Switching**: Support for both Hopper H100/H200 (SM90) and Blackwell B200/B300 (SM100)
- **Grace Blackwell Superchip**: Unified memory architecture examples
- **NVL72 Systems**: Multi-GPU cluster optimization examples
- **Enhanced Profiling**: Latest Nsight Systems, Nsight Compute, HTA, and Perf integration

## 📚 Book Chapters

Each chapter contains practical code examples demonstrating key performance engineering concepts:

### Chapter 1: Introduction and AI System Overview
- **`code/ch1/performance_basics.py`**: Basic performance measurement and goodput analysis
- Demonstrates unified memory architecture, Tensor Core performance, and Transformer Engine
- Enhanced with PyTorch 2.8 optimizations and architecture-specific features

### Chapter 2: AI System Hardware Overview  
- **`code/ch2/hardware_info.py`**: Grace Blackwell superchip hardware analysis
- NVLink bandwidth testing, unified memory capabilities, and Blackwell specifications
- Latest hardware detection and optimization features

### Chapter 3: OS, Docker, and Kubernetes Tuning
- **`code/ch3/bind_numa_affinity.py`**: NUMA binding and CPU pinning for GPU optimization
- **`code/ch3/numa_bind.sh`**: Shell scripts for NUMA topology management
- Memory pinning, CPU affinity, and DataLoader optimization
- Enhanced for latest system configurations

### Chapter 4: Distributed Networking Communication
- **`code/ch4/after_ddp.py`**: Distributed training with DDP communication overlap
- NCCL vs Gloo backend comparison, gradient compression, and RDMA optimization
- Latest distributed training optimizations

### Chapter 5: CUDA Programming Fundamentals
- **`code/ch5/`**: CUDA kernel development and optimization
- Memory coalescing, shared memory usage, and kernel fusion techniques
- Updated for CUDA 12.8 and latest features

### Chapter 6: GPU Memory Hierarchy Optimization
- **`code/ch6/`**: Memory bandwidth optimization and cache utilization
- Global memory access patterns, shared memory optimization, and L2 cache usage
- Enhanced for HBM3/HBM3e memory systems

### Chapter 7: Tensor Core and Matrix Operations
- **`code/ch7/`**: Tensor Core optimization and matrix multiplication
- FP8/FP4 precision, GEMM optimization, and custom kernel development
- Latest Tensor Core features and optimizations

### Chapter 8: CUDA Streams and Asynchronous Programming
- **`code/ch8/`**: Stream-based parallelism and asynchronous execution
- Kernel fusion, pipeline parallelism, and communication overlap
- Enhanced stream management and optimization

### Chapter 9: Dynamic Parallelism and CUDA Graphs
- **`code/ch9/`**: Dynamic parallelism and CUDA graph optimization
- Persistent kernels, graph capture, and dynamic workload distribution
- Latest CUDA graph features and optimizations

### Chapter 10: Advanced CUDA Features
- **`code/ch10/`**: Advanced CUDA programming techniques
- Cooperative groups, warp-level primitives, and custom atomic operations
- TMA (Tensor Memory Accelerator) support and optimizations

### Chapter 11: PyTorch Optimization
- **`code/ch11/`**: PyTorch-specific optimizations and techniques
- Compiler optimizations, memory management, and distributed training
- Enhanced with PyTorch 2.8 features and optimizations

### Chapter 12: Triton for Custom GPU Kernels
- **`code/ch12/`**: OpenAI Triton for high-performance GPU kernel development
- Custom attention mechanisms, fused operations, and kernel autotuning
- Updated for Triton 3.3 and latest features

### Chapter 13: Distributed Training Optimization
- **`code/ch13/`**: Large-scale distributed training techniques
- Model parallelism, pipeline parallelism, and memory optimization
- Latest distributed training features and optimizations

### Chapter 14: Inference Optimization
- **`code/ch14/`**: High-throughput inference optimization
- Model serving, quantization, and batch processing
- Enhanced inference optimization techniques

### Chapter 15: Model Compression and Quantization
- **`code/ch15/`**: Model compression techniques for efficiency
- Pruning, quantization, and knowledge distillation
- Latest compression and quantization techniques

### Chapter 16: Memory Optimization
- **`code/ch16/`**: Advanced memory management techniques
- Gradient checkpointing, activation recomputation, and memory-efficient training
- Enhanced memory optimization strategies

### Chapter 17: Profiling and Debugging
- **`code/ch17/`**: Performance profiling and debugging tools
- NVIDIA Nsight, PyTorch profiler, and custom profiling utilities
- Latest profiling tools and techniques

### Chapter 18: Model Serving and Deployment
- **`code/ch18/`**: Production model serving optimization
- vLLM, TensorRT, and high-throughput inference
- Enhanced serving and deployment strategies

### Chapter 19: Advanced Optimization Techniques
- **`code/ch19/`**: Cutting-edge optimization techniques
- FlashAttention, sparse computation, and novel architectures
- Latest advanced optimization techniques

### Chapter 20: Future Trends and Emerging Technologies
- **`code/ch20/`**: Future directions in AI performance engineering
- AI-assisted optimization, automated tuning, and emerging hardware
- Latest trends and emerging technologies

## 🛠️ Installation

### Prerequisites

- **NVIDIA GPU**: Hopper H100/H200, Blackwell B200/B300, or compatible
- **CUDA 12.8**: Latest CUDA toolkit
- **Python 3.9+**: Python environment
- **Linux**: Ubuntu 22.04+ recommended

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/ai-performance-engineering.git
   cd ai-performance-engineering
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_latest.txt
   ```

3. **Verify installation**:
   ```bash
   python code/ch1/performance_basics.py
   ```

### System Dependencies

Install system packages for optimal performance:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y numactl nvidia-container-toolkit infiniband-diags perftest

# CentOS/RHEL
sudo yum install -y numactl nvidia-container-toolkit infiniband-diags perftest
```

## 🚀 Quick Examples

### Basic Performance Analysis
```bash
# Run basic performance measurement
python code/ch1/performance_basics.py

# Test hardware capabilities
python code/ch2/hardware_info.py
```

### NUMA Optimization
```bash
# Test NUMA binding
python code/ch3/bind_numa_affinity.py

# Run NUMA topology analysis
bash code/ch3/numa_bind.sh
```

### Distributed Training
```bash
# Test DDP communication overlap
python code/ch4/after_ddp.py --test ddp

# Compare NCCL vs Gloo backends
python code/ch4/after_ddp.py --test nccl
```

### Architecture Switching
```bash
# Switch to Hopper H100/H200 (sm_90)
bash code/switch_architecture.sh sm_90

# Switch to Blackwell B200/B300 (sm_100)
bash code/switch_architecture.sh sm_100

# Auto-detect and build for current architecture
bash code/build_all.sh
```

### Enhanced Profiling
```bash
# Comprehensive profiling
bash code/profiler_scripts/comprehensive_profile.sh script.py

# PyTorch profiling
bash code/profiler_scripts/pytorch_profile.sh script.py

# Architecture-specific profiling
bash code/profiler_scripts/pytorch_profile.sh script.py blackwell
```

## 🔧 Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# NCCL optimization
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# PyTorch optimization
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Enhanced profiling
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Docker Support

For containerized environments:

```bash
# Build Docker image
docker build -t ai-perf-eng .

# Run with GPU support
docker run --gpus all --rm -it ai-perf-eng
```

## 📈 Performance Monitoring

### Real-time Monitoring

```bash
# GPU utilization and memory
watch -n 1 nvidia-smi

# Network performance
ibstat
ibv_devinfo

# System performance
htop
iostat
```

### Profiling Tools

- **NVIDIA Nsight Systems 2025.1**: Timeline analysis
- **NVIDIA Nsight Compute 2025.1**: Kernel profiling  
- **PyTorch Profiler 2.8**: Framework-level profiling
- **HTA (Holistic Tracing Analysis)**: Multi-GPU analysis
- **Perf**: System-level analysis
- **Triton 3.3**: Custom kernel profiling

### Enhanced Profiling Commands

```bash
# Comprehensive profiling
nsys profile -t cuda,nvtx,osrt,triton -o timeline_profile python script.py

# Kernel analysis
ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py

# HTA for multi-GPU
nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton -o hta_profile python script.py

# System analysis
perf record -g -p $(pgrep python) -o perf.data
perf report -i perf.data
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run tests
pytest tests/

# Format code
black code/
flake8 code/
```

## 📖 Documentation

- **Book Chapters**: Each chapter contains detailed explanations and code examples
- **API Reference**: Comprehensive documentation for all functions and classes
- **Performance Guides**: Step-by-step optimization guides
- **Troubleshooting**: Common issues and solutions

## 🏆 Acknowledgments

This repository is based on the comprehensive AI Performance Engineering book, covering:

- **Hardware Optimization**: Grace Blackwell superchips, NVL72 systems
- **Software Optimization**: PyTorch 2.8, CUDA 12.8, Triton 3.3
- **System Optimization**: NUMA binding, memory pinning, network tuning
- **Algorithm Optimization**: Distributed training, model parallelism, quantization
- **Profiling Optimization**: Latest profiling tools and techniques

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-zone
- **PyTorch**: https://pytorch.org/
- **OpenAI Triton**: https://github.com/openai/triton
- **NVIDIA Magnum IO**: https://developer.nvidia.com/magnum-io

---

**Note**: This repository is designed for educational and research purposes. For production deployments, ensure proper testing and validation of all optimizations in your specific environment.
