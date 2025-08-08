# AI Performance Engineering Code Examples

This repository contains updated code examples extracted from the AI Performance Engineering book, optimized for:

- **PyTorch 2.8** with CUDA 12.9 support
- **Triton 3.4** for custom kernel development
- Latest profiling tools (nsys, ncu, PyTorch Profiler, HTA)

## Directory Structure

### Fully Implemented Chapters

**Chapter 3: OS, Docker, and Kubernetes Tuning** (`ch3/`)
- NUMA-aware process binding and memory allocation
- Container runtime optimizations
- Kubernetes GPU orchestration
- System-level performance tuning scripts

**Chapter 4: Distributed Networking Communication** (`ch4/`)
- Communication/computation overlap examples
- NCCL vs Gloo backend comparisons
- DistributedDataParallel optimizations
- Common distributed training pitfalls and solutions

**Chapter 13: Profiling, Tuning, and Scaling PyTorch** (`ch13/`)
- Comprehensive PyTorch profiling examples
- Memory optimization techniques
- FSDP (Fully Sharded Data Parallel) implementation
- Custom CUDA allocator configuration

### Chapter Summaries

**Chapters 1-2**: Conceptual overviews of AI systems performance engineering and hardware architecture (NVIDIA Grace Blackwell, NVL72)

**Chapters 5-12**: Hardware and system fundamentals:
- Storage optimization and GPUDirect Storage (Ch5)
- CUDA programming fundamentals (Ch6)
- GPU memory access pattern optimization (Ch7)
- Occupancy tuning and warp efficiency (Ch8)
- Arithmetic intensity and kernel fusion (Ch9)
- Advanced memory management (Ch10)
- Multi-GPU and distributed optimization (Ch11)
- Cooperative groups and synchronization (Ch12)

**Chapters 14-20**: Advanced optimizations:
- PyTorch compiler and Triton kernels (Ch14)
- Model serving and inference optimization (Ch15)
- Advanced profiling and analysis (Ch16)
- Dynamic adaptive RL inference (Ch17)
- FlashMLA and kernel tuning (Ch18)
- Dynamic parallelism and precision (Ch19)
- AI-assisted performance tuning (Ch20)

## Key Features

### Updated Dependencies
- PyTorch 2.8 with full CUDA 12.9 support
- Triton 3.4 for custom kernel development
- Latest NVIDIA libraries (NCCL, cuDNN, etc.)
- Modern profiling tools integration

### Performance Optimizations
- NUMA-aware CPU/GPU affinity binding
- Communication/computation overlap patterns
- Memory-efficient training techniques
- Distributed scaling strategies

### Profiling and Debugging
- Multi-level profiling (PyTorch, Nsight, perf)
- Memory usage analysis and optimization
- Distributed training bottleneck identification
- Automated performance regression testing

## Usage Examples

### Quick Start
```bash
# Set up environment
pip install -r requirements.txt

# Run NUMA-aware training
python ch3/bind_numa_affinity.py

# Test distributed communication
python ch4/nccl_benchmark.py --world-size 2

# Profile PyTorch workload
python ch13/train_deepseek_v3.py
```

### Distributed Training
```bash
# Multi-GPU training with overlap optimization
torchrun --nproc_per_node=4 ch4/after_ddp.py

# FSDP training with memory optimization
torchrun --nproc_per_node=4 ch13/fsdp_example.py
```

### System Optimization
```bash
# Apply OS-level optimizations
sudo bash ch3/system_tuning.sh

# Configure GPU settings
bash ch3/gpu_setup_commands.sh
```

## Hardware Requirements

### Minimum Requirements
- 2+ NVIDIA GPUs (Volta architecture or newer)
- CUDA 12.9+ compatible drivers
- 16GB+ system RAM
- Linux OS with NUMA support

### Recommended Configuration
- NVIDIA NVL72 or similar multi-GPU system
- InfiniBand or high-speed Ethernet networking
- NVMe SSD storage
- 64GB+ system RAM

## Integration Notes

### PyTorch 2.8 Compatibility
All examples have been updated to use:
- `torch.compile()` for automatic optimization
- `torch.cuda.amp` for mixed precision training
- Modern distributed training APIs
- Latest memory management features

### CUDA 12.9 Features
- Enhanced unified memory support
- Improved kernel launch efficiency
- Advanced profiling capabilities
- Better multi-GPU memory management

### Triton 3.4 Integration
- Custom kernel examples (where applicable)
- JIT compilation optimizations
- GPU-specific autotuning
- Integration with PyTorch compiler

## Performance Monitoring

### Automated Benchmarking
```bash
# Run comprehensive benchmarks
python scripts/run_benchmarks.py

# Generate performance reports
python scripts/analyze_performance.py
```

### Continuous Integration
Examples include CI/CD configurations for:
- Performance regression testing
- Memory usage monitoring
- Distributed scaling validation
- Cross-platform compatibility

## Contributing

When adding new examples:
1. Ensure PyTorch 2.8+ compatibility
2. Include proper error handling
3. Add comprehensive documentation
4. Provide benchmarking scripts
5. Test on multiple GPU configurations

## Resources

- [PyTorch 2.8 Documentation](https://pytorch.org/docs/2.8/)
- [CUDA 12.9 Programming Guide](https://docs.nvidia.com/cuda/)
- [Triton 3.4 Documentation](https://triton-lang.org/)
- [NVIDIA Developer Resources](https://developer.nvidia.com/)

## License

Examples are provided for educational purposes in conjunction with the AI Performance Engineering book.
