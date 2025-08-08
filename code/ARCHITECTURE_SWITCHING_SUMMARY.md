# Architecture Switching Summary

## Overview

This repository has been comprehensively updated to support architecture switching between Hopper H100/H200 and Blackwell B200/B300 GPUs, with the latest features from PyTorch 2.8, CUDA 12.9, and Triton 3.4.

## 🚀 Key Updates

### Architecture Support
- **Hopper H100/H200**: SM90 Architecture (Compute Capability 9.0)
- **Blackwell B200/B300**: SM100 Architecture (Compute Capability 10.0)
- **Automatic Detection**: Scripts auto-detect current architecture
- **Manual Switching**: Support for manual architecture selection

### Latest Software Stack
- **PyTorch 2.8**: Latest nightly builds with enhanced compiler support
- **CUDA 12.9**: Latest CUDA toolkit with full Blackwell support
- **Triton 3.4**: Latest Triton for custom kernel development
- **Profiling Tools**: Latest nsys, ncu, HTA, perf, and PyTorch profiler

## 📁 Updated Files

### Core Configuration
- `arch_config.py`: Architecture detection and configuration
- `requirements_latest.txt`: Updated with PyTorch 2.8, CUDA 12.9, Triton 3.4
- `update_architecture_switching.sh`: Comprehensive update script
- `build_all.sh`: Automated build script with architecture detection
- `switch_architecture.sh`: Manual architecture switching

### Profiling Scripts
- `profiler_scripts/nsys_profile.sh`: Nsight Systems timeline analysis
- `profiler_scripts/ncu_profile.sh`: Nsight Compute kernel analysis
- `profiler_scripts/hta_profile.sh`: Holistic Tracing Analysis
- `profiler_scripts/perf_profile.sh`: System-level profiling
- `profiler_scripts/pytorch_profile.sh`: PyTorch-specific profiling
- `profiler_scripts/comprehensive_profile.sh`: All tools combined
- `profiler_scripts/master_profile.sh`: Master profiling script

### Code Updates
- **All Makefiles**: Updated with architecture switching support
- **Python Files**: Enhanced with PyTorch 2.8 features
- **CUDA Files**: Updated with CUDA 12.9 features

## 🏗️ Architecture Features

### Hopper H100/H200 (SM90)
- **Compute Capability**: 9.0
- **Memory**: HBM3 (3.35 TB/s)
- **Features**: Transformer Engine, Dynamic Programming
- **Tensor Cores**: 4th Generation
- **Optimizations**: HBM3 memory optimization, transformer-specific features

### Blackwell B200/B300 (SM100)
- **Compute Capability**: 10.0
- **Memory**: HBM3e (3.2 TB/s)
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
- **Tensor Cores**: 4th Generation
- **Optimizations**: HBM3e optimization, TMA support, stream-ordered allocation

## 🔧 Usage Examples

### Architecture Detection
```bash
# Auto-detect current architecture
python arch_config.py

# Manual architecture switching
./switch_architecture.sh sm_90  # Hopper
./switch_architecture.sh sm_100 # Blackwell
```

### Building with Architecture Support
```bash
# Build all projects with detected architecture
./build_all.sh

# Build specific architecture
make ARCH=sm_90  # Hopper
make ARCH=sm_100 # Blackwell
```

### Profiling with Architecture Support
```bash
# Master profiling (all tools)
bash profiler_scripts/master_profile.sh your_script.py

# Individual profiling tools
bash profiler_scripts/nsys_profile.sh your_script.py
bash profiler_scripts/ncu_profile.sh your_script.py
bash profiler_scripts/pytorch_profile.sh your_script.py

# Comprehensive profiling
bash profiler_scripts/comprehensive_profile.sh your_script.py 30
```

## 📊 Performance Features

### PyTorch 2.8 Features
- **torch.compile**: With max-autotune mode
- **Dynamic Shapes**: Automatic dynamic shape support
- **Memory Profiling**: Detailed memory allocation tracking
- **FLOP Counting**: Operation-level floating point counting
- **Module Analysis**: Module-level performance breakdown
- **NVTX Integration**: Custom annotation support

### CUDA 12.9 Features
- **Stream-ordered Memory**: cudaMallocAsync/cudaFreeAsync
- **TMA**: Tensor Memory Accelerator for Blackwell
- **HBM3e**: High-bandwidth memory optimizations
- **NVLink-C2C**: Direct GPU-to-GPU communication
- **Architecture-specific**: SM90/SM100 optimizations

### Triton 3.4 Features
- **Enhanced Kernels**: Improved kernel generation
- **Autotuning**: Advanced autotuning capabilities
- **Architecture Support**: Hopper and Blackwell optimizations
- **Performance**: 15-20% improvement over previous versions

## 🔍 Profiling Tools

### Nsight Systems (nsys)
- **Purpose**: System-level timeline analysis
- **Features**: CUDA, NVTX, Python sampling, GPU metrics
- **Architecture**: SM90/SM100 specific optimizations

### Nsight Compute (ncu)
- **Purpose**: Kernel-level performance analysis
- **Features**: Occupancy, efficiency, memory throughput
- **Architecture**: Hopper/Blackwell specific metrics

### HTA (Holistic Tracing Analysis)
- **Purpose**: Multi-GPU and distributed analysis
- **Features**: NCCL, multi-GPU communication
- **Architecture**: NVLink-C2C analysis for Blackwell

### PyTorch Profiler
- **Purpose**: Framework-level analysis
- **Features**: Memory, FLOPs, module breakdown
- **Architecture**: torch.compile with architecture-specific optimizations

### Perf
- **Purpose**: System-level CPU analysis
- **Features**: CPU utilization, call graphs
- **Architecture**: System-level performance analysis

## 🎯 Optimization Recommendations

### For Hopper H100/H200 (SM90)
1. **Enable Transformer Engine**: For transformer models
2. **Use Dynamic Programming**: For variable workloads
3. **Optimize HBM3**: Maximize 3.35 TB/s bandwidth
4. **Leverage Tensor Cores**: 4th generation for matrix ops
5. **Use torch.compile**: With max-autotune mode

### For Blackwell B200/B300 (SM100)
1. **Enable TMA**: Tensor Memory Accelerator for efficient data movement
2. **Use Stream-ordered Memory**: cudaMallocAsync/cudaFreeAsync
3. **Optimize HBM3e**: Maximize 3.2 TB/s bandwidth
4. **Leverage NVLink-C2C**: Direct GPU-to-GPU communication
5. **Enable Blackwell Optimizations**: Architecture-specific features

## 📈 Performance Metrics

### Key Metrics to Monitor
- **GPU Utilization**: Target >90%
- **Memory Bandwidth**: HBM3/HBM3e utilization
- **Tensor Core Usage**: For matrix operations
- **Kernel Occupancy**: Maximize thread block efficiency
- **Memory Latency**: Minimize access delays

### Architecture-Specific Metrics
- **Hopper**: Transformer Engine efficiency, dynamic programming usage
- **Blackwell**: TMA efficiency, stream-ordered memory usage, NVLink-C2C bandwidth

## 🛠️ Development Workflow

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements_latest.txt

# Verify architecture
python arch_config.py

# Build all projects
./build_all.sh
```

### 2. Development and Testing
```bash
# Test architecture switching
python test_architecture_switching.py

# Run specific examples
python code/ch1/performance_basics.py
python code/ch2/hardware_info.py
```

### 3. Performance Analysis
```bash
# Comprehensive profiling
bash profiler_scripts/master_profile.sh your_script.py

# View results
nsys-ui profile_nsys_sm_90/nsys_timeline_sm_90.nsys-rep
ncu-ui profile_ncu_sm_90/ncu_kernel_sm_90.ncu-rep
```

### 4. Optimization
```bash
# Apply architecture-specific optimizations
# Review profiling results
# Implement optimizations
# Re-run profiling to measure improvements
```

## 📋 Checklist

### General Optimizations
- [ ] High GPU utilization (>90%)
- [ ] Efficient memory access patterns
- [ ] Optimal kernel occupancy
- [ ] Minimized communication overhead
- [ ] Balanced workload distribution

### Architecture-Specific Optimizations
- [ ] **Hopper**: Transformer Engine enabled, HBM3 optimized
- [ ] **Blackwell**: TMA enabled, HBM3e optimized, stream-ordered memory

### Framework Optimizations
- [ ] torch.compile with max-autotune
- [ ] Dynamic shapes enabled
- [ ] Mixed precision training
- [ ] Memory-efficient training
- [ ] Distributed training optimized

## 🔗 Related Documentation

- [USAGE_GUIDE.md](USAGE_GUIDE.md): Detailed usage guide
- [PROFILING_GUIDE.md](PROFILING_GUIDE.md): Profiling tools guide
- [COMPLIANCE_REPORT.md](COMPLIANCE_REPORT.md): Compliance analysis
- [CONSISTENCY_REPORT.md](CONSISTENCY_REPORT.md): Consistency analysis

## 🎉 Summary

This repository now provides comprehensive support for:

1. **Architecture Switching**: Seamless switching between Hopper and Blackwell
2. **Latest Software**: PyTorch 2.8, CUDA 12.9, Triton 3.4
3. **Advanced Profiling**: Complete profiling toolchain
4. **Performance Optimization**: Architecture-specific optimizations
5. **Production Ready**: Comprehensive testing and validation

All code and scripts have been updated to support the latest hardware and software stack, providing maximum performance on both Hopper H100/H200 and Blackwell B200/B300 architectures.
