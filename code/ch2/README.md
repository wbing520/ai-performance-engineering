# Chapter 2: AI System Hardware Overview

This directory contains code examples that demonstrate the core hardware concepts from Chapter 2 of the AI Performance Engineering book.

## Key Concepts Demonstrated

### 1. GPU Architecture and Memory Hierarchy
- Understanding GPU compute capabilities and memory organization
- Memory hierarchy: Registers → Shared Memory → L1 Cache → L2 Cache → HBM
- Unified memory architecture between CPU and GPU

### 2. Hardware Monitoring and Utilization
- Real-time system resource monitoring
- GPU utilization, temperature, and memory usage
- CPU and memory utilization tracking

### 3. Tensor Cores and Precision Formats
- Support for different precision formats (FP32, FP16, FP8, FP4)
- Tensor Core capabilities and compute capability detection
- Performance differences between precision formats

### 4. NVLink and Interconnect Technologies
- NVLink status and capabilities
- High-speed GPU-to-GPU communication
- Network fabric information

## Files

- `hardware_info.py`: Main demonstration script showing hardware monitoring and architecture concepts
- `requirements.txt`: Required Python dependencies
- `README.md`: This file

## Running the Examples

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the hardware demonstration:
```bash
python hardware_info.py
```

## Expected Output

The script will demonstrate:
- Comprehensive GPU architecture information
- Memory hierarchy and utilization
- System resource monitoring
- Tensor Core support and precision formats
- NVLink status and capabilities
- Memory transfer demonstrations
- Compute capability benchmarks

## Key Hardware Concepts

### Grace Blackwell Superchip
- Unified CPU-GPU architecture with shared memory
- NVLink-C2C for high-speed CPU-GPU communication
- ~900 GB of unified memory per superchip

### NVL72 Rack Architecture
- 72 Blackwell GPUs in a single rack
- NVLink 5 and NVSwitch for GPU-to-GPU communication
- 130 kW power consumption with liquid cooling
- 1.44 exaFLOPS theoretical peak performance

### Memory Hierarchy
1. **Registers**: Fastest, per-thread storage
2. **Shared Memory**: Per-SM, configurable cache
3. **L1 Cache**: Per-SM, automatic cache
4. **L2 Cache**: Shared across all SMs
5. **HBM**: High-bandwidth memory, off-chip

### Tensor Cores
- Specialized units for matrix operations
- Support for reduced precision (FP16, FP8, FP4)
- Automatic precision selection with Transformer Engine
- Significant speedup for AI workloads

## Dependencies

- PyTorch 2.8+ for GPU operations and CUDA support
- psutil for system monitoring
- GPUtil for GPU metrics
- numpy for numerical operations

## Hardware Requirements

- NVIDIA GPU with CUDA support (for full functionality)
- Linux/macOS/Windows with Python 3.8+
- nvidia-smi for NVLink information (optional)

This chapter provides the foundation for understanding modern AI hardware architecture and how to monitor and utilize it effectively.
