# AI Performance Engineering

[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

> **O'Reilly Book - Fall 2025**  
> [Available on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

---

## 📋 Performance Checklist

The book includes a comprehensive **175+ item performance checklist** covering:

- ✅ Performance Tuning Mindset and Cost Optimization
- ✅ Reproducibility and Documentation Best Practices
- ✅ System Architecture and Hardware Planning
- ✅ Operating System and Driver Optimizations
- ✅ GPU Programming and CUDA Tuning
- ✅ Distributed Training and Network Optimization
- ✅ Efficient Inference and Serving
- ✅ Power and Thermal Management
- ✅ Latest Profiling Tools and Techniques
- ✅ Architecture-Specific Optimizations

---

## 📚 About This Repository

This repository contains comprehensive code examples, tools, and resources for AI Systems Performance Engineering. It accompanies the O'Reilly book covering GPU optimization, distributed training, inference scaling, and performance tuning for modern AI workloads.

### 🎯 Key Focus Areas
- **GPU Architecture & CUDA Programming**
- **Distributed Training & Inference**
- **Memory Optimization & Profiling**
- **PyTorch Performance Tuning**
- **Multi-Node Scaling Strategies**

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch with CUDA
- Docker (optional)

### Getting Started
```bash
# Clone the repository
git clone https://github.com/your-repo/ai-performance-engineering.git
cd ai-performance-engineering

# Install dependencies for a specific chapter
cd code/ch1
pip install -r requirements.txt

# Run examples
python performance_basics.py
```

### 🏗️ Architecture Switching

This repository supports multiple NVIDIA GPU architectures. Switch between Hopper (H100/H200) and Blackwell (B200/B300) architectures:

```bash
# Switch to Hopper H100/H200 (sm_90)
./code/switch_architecture.sh sm_90

# Switch to Blackwell B200/B300 (sm_100)  
./code/switch_architecture.sh sm_100

# Auto-detect and build for current architecture
./code/build_all.sh
```

**Supported Architectures:**
- **Hopper H100/H200** (`sm_90`): 80-141GB memory, 4-6 PFLOPS
- **Blackwell B200/B300** (`sm_100`): 192-288GB memory, 20-30 PFLOPS

For detailed architecture specifications and performance benchmarks, see [`code/README.md`](code/README.md).

### 🔧 Latest Features

**Updated for PyTorch 2.8, CUDA 12.9, and Triton 3.4:**

- **PyTorch 2.8**: Enhanced compiler, dynamic shapes, improved profiler
- **CUDA 12.9**: Latest CUDA features, improved kernel performance
- **Triton 3.4**: Latest Triton optimizations, architecture-specific kernels
- **Enhanced Profiling**: Nsight Systems 2024.1, Nsight Compute 2024.1
- **HTA**: Holistic Tracing Analysis for multi-GPU systems
- **Perf**: Enhanced system-level analysis
- **Architecture Optimizations**: Hopper/Blackwell-specific features

---

## 📖 Book Chapters Overview

### **Chapter 1: Introduction and AI System Overview**
- The AI Systems Performance Engineer
- Benchmarking and Profiling
- Scaling Distributed Training and Inference
- Managing Resources Efficiently
- Cross-Team Collaboration
- Transparency and Reproducibility

### **Chapter 2: AI System Hardware Overview**
- The CPU and GPU "Superchip"
- NVIDIA Grace CPU & Blackwell GPU
- NVIDIA GPU Tensor Cores and Transformer Engine
- Streaming Multiprocessors, Threads, and Warps
- Ultra-Scale Networking
- NVLink and NVSwitch
- Multi-GPU Programming

### **Chapter 3: OS, Docker, and Kubernetes Tuning**
- Operating System Configuration
- GPU Driver and Software Stack
- NUMA Awareness and CPU Pinning
- Container Runtime Optimizations
- Kubernetes for Topology-Aware Orchestration
- Memory Isolation and Resource Management

### **Chapter 4: Tuning Distributed Networking Communication**
- Overlapping Communication and Computation
- NCCL for Distributed Multi-GPU Communication
- Topology Awareness in NCCL
- Distributed Data Parallel Strategies
- NVIDIA Inference Transfer Library (NIXL)
- In-Network SHARP Aggregation

### **Chapter 5: GPU-based Storage I/O Optimizations**
- Fast Storage and Data Locality
- NVIDIA GPUDirect Storage
- Distributed, Parallel File Systems
- Multi-Modal Data Processing with NVIDIA DALI
- Creating High-Quality LLM Datasets

### **Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy**
- Understanding GPU Architecture
- Threads, Warps, Blocks, and Grids
- CUDA Programming Refresher
- Understanding GPU Memory Hierarchy
- Maintaining High Occupancy and GPU Utilization
- Roofline Model Analysis

### **Chapter 7: Profiling and Tuning GPU Memory Access Patterns**
- Coalesced vs. Uncoalesced Global Memory Access
- Vectorized Memory Access
- Tiling and Data Reuse Using Shared Memory
- Warp Shuffle Intrinsics
- Asynchronous Memory Prefetching

### **Chapter 8: Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism**
- Profiling and Diagnosing GPU Bottlenecks
- Nsight Systems and Compute Analysis
- Tuning Occupancy
- Improving Warp Execution Efficiency
- Exposing Instruction-Level Parallelism

### **Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity**
- Multi-Level Micro-Tiling
- Kernel Fusion
- Mixed Precision and Tensor Cores
- Using CUTLASS for Optimal Performance
- Inline PTX and SASS Tuning

### **Chapter 10: Intra-Kernel Pipelining and Cooperative Thread Block Clusters**
- Intra-Kernel Pipelining Techniques
- Warp-Specialized Producer-Consumer Model
- Persistent Kernels and Megakernels
- Thread Block Clusters and Distributed Shared Memory
- Cooperative Groups

### **Chapter 11: Inter-Kernel Pipelining and CUDA Streams**
- Using Streams to Overlap Compute with Data Transfers
- Stream-Ordered Memory Allocator
- Fine-Grained Synchronization with Events
- Zero-Overhead Launch with CUDA Graphs

### **Chapter 12: Dynamic and Device-Side Kernel Orchestration**
- Dynamic Scheduling with Atomic Work Queues
- Batch Repeated Kernel Launches with CUDA Graphs
- Dynamic Parallelism
- Orchestrate Across Multiple GPUs with NVSHMEM

### **Chapter 13: Profiling, Tuning, and Scaling PyTorch**
- NVTX Markers and Profiling Tools
- PyTorch Compiler (torch.compile)
- Profiling and Tuning Memory in PyTorch
- Scaling with PyTorch Distributed
- Multi-GPU Profiling with HTA

### **Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends**
- PyTorch Compiler Deep Dive
- Writing Custom Kernels with OpenAI Triton
- PyTorch XLA Backend
- Advanced Triton Kernel Implementations

### **Chapter 15: Multi-Node Inference Parallelism and Routing**
- Disaggregated Prefill and Decode Architecture
- Parallelism Strategies for MoE Models
- Speculative and Parallel Decoding Techniques
- Dynamic Routing Strategies

### **Chapter 16: Profiling, Debugging, and Tuning Inference at Scale**
- Workflow for Profiling and Tuning Performance
- Dynamic Request Batching and Scheduling
- Systems-Level Optimizations
- Quantization Approaches for Real-Time Inference
- Application-Level Optimizations

### **Chapter 17: Scaling Disaggregated Prefill and Decode**
- Prefill-Decode Disaggregation Benefits
- Prefill Workers Design
- Decode Workers Design
- Disaggregated Routing and Scheduling Policies
- Scalability Considerations

### **Chapter 18: Advanced Prefill-Decode and KV Cache Tuning**
- Optimized Decode Kernels (FlashMLA, ThunderMLA, FlexDecoding)
- Tuning KV Cache Utilization and Management
- Heterogeneous Hardware and Parallelism Strategies
- SLO-Aware Request Management

### **Chapter 19: Dynamic and Adaptive Inference Engine Optimizations**
- Adaptive Parallelism Strategies
- Dynamic Precision Changes
- Kernel Auto-Tuning
- Reinforcement Learning Agents for Runtime Tuning
- Adaptive Batching and Scheduling

### **Chapter 20: AI-Assisted Performance Optimizations**
- AlphaTensor AI-Discovered Algorithms
- Automated GPU Kernel Optimizations
- Self-Improving AI Agents
- Scaling Toward Multi-Million GPU Clusters

---

## 🛠️ Tools and Utilities

### Profiling Scripts
- `code/profiler_scripts/comprehensive_profile.sh` - Comprehensive GPU profiling
- `code/profiler_scripts/enhanced_profiling.sh` - Enhanced profiling with Nsight
- `code/profiler_scripts/hta_profile.sh` - Holistic Tracing Analysis

### Performance Analysis Tools
- `tools/comprehensive_profiling.py` - Python-based profiling utilities
- `tools/compare_nsight/` - Nsight Systems comparison tools
- `tools/inference_gpu_cluster_sizing/` - Cluster sizing notebooks

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

---

## 🎥 Community Resources

### Monthly Meetups (100,000+ Global Members, 20+ Cities)
- **Meetup Group**: [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering)
- **YouTube Channel**: [AI Performance Engineering](https://www.youtube.com/@AIPerformanceEngineering)

### Recent Meetups

#### July 21, 2025
- [YouTube Video](https://youtu.be/jaiMotxv8ck)
- [Dynamic Adaptive RL Inference CUDA Kernel Tuning](resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf)

#### June 16, 2025
- [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)

#### May 19, 2025
- [YouTube Video](https://youtu.be/F8jJwI9xHTE)
- [PyTorch Optimizations: Data Loader Pipeline](resources/PyTorch_Model_Optimization_Data_Loader.pdf)
- [Cross-Architecture CUDA and ROCm Kernel Development](resources/ai_perf_eng_meetup.pdf)

#### April 21, 2025
- [YouTube Video](https://youtu.be/XoZcY_fDUKA)
- [AI Performance Engineering Meetup](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf)
- [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf)



---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:
- Code examples and improvements
- Documentation updates
- Performance optimization techniques
- Bug reports and feature requests

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Book**: [AI Systems Performance Engineering on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)
- **Meetup**: [AI Performance Engineering Meetup Group](https://www.meetup.com/ai-performance-engineering)
- **YouTube**: [AI Performance Engineering Channel](https://www.youtube.com/@AIPerformanceEngineering)

---

*Built with ❤️ for the AI performance engineering community*