# AI Performance Engineering

## About This Repo

AI Systems Performance Engineering code, tooling, and resources for the O'Reilly book covering GPU optimization, distributed training, inference scaling, and full-stack performance tuning for modern AI workloads.

[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

> **O'Reilly Book – Fall 2025**  
> [Available on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

### AI Systems Performance Engineering Book
Modern AI systems demand more than raw FLOPs—they need goodput‑driven, profile‑first engineering across hardware, software, and algorithms. This hands‑on guide shows how to turn GPUs, interconnects, and runtime stacks into efficient, reliable training and inference pipelines. 

You’ll learn to diagnose real bottlenecks with Nsight and PyTorch profilers, squeeze bandwidth and memory, and use compiler stacks (PyTorch + OpenAI Triton) to craft high‑impact kernels. On the serving side, master high‑throughput inference with vLLM/SGLang, TensorRT‑LLM, and NVIDIA Dynamo—including disaggregated prefill/decode and paged KV cache—then scale across racks without blowing the budget.

Using a hands‑on, empirical methodology with case studies, profiling data, this book is useful for AI/ML engineers, systems engineers, researchers, and platform teams building or operating training/inference at scale. The book contains thousands of lines of PyTorch and CUDA C++ code examples for modern NVIDIA GPUs.

* Profile for goodput, not just utilization—use Nsight Systems/Compute and the PyTorch profiler to find the real stall points. 

* Exploit memory & bandwidth—optimize layouts, caching, and data movement to feed the GPU continuously. 

* Tune with compilers—leverage the PyTorch compiler stack and Triton to generate high‑impact kernels without C++ boilerplate. 

* Scale training sanely—apply parallelism strategies (DP, FSDP, TP, PP, CP, and MoE) and overlap computation/communication to minimize bubbles. 

* Serve trillion parameter models efficiently—use vLLM, SGLang, TensorRT‑LLM and NVIDIA Dynamo with disaggregated prefill/decode and KV‑cache movement.

* Reduce cost per token—engineer for performance‑per‑watt and throughput per dollar, not just peak speed.

* Adopt AI‑assisted optimization—let AI help synthesize and tune kernels as systems outgrow manual tweaking
 
* Ship with confidence—apply the 175+ item checklist to reproduce wins and prevent regressions across teams.

### Author Bio

Chris Fregly is a performance engineer and AI product leader who has driven innovations at Netflix, Databricks, and Amazon Web Services (AWS). He has led performance‑focused engineering teams that built AI/ML products, scaled go‑to‑market initiatives, and reduced cost for large‑scale generative‑AI and analytics workloads. 

Chris is the author of two other O’Reilly books: Data Science on AWS and Generative AI on AWS. He's also the creator of the O’Reilly course “High‑Performance AI in Production with NVIDIA GPUs.” 

His work spans kernel‑level tuning, compiler‑driven acceleration, distributed training, and high‑throughput inference. Chris hosts a monthly meetup called [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering).

### 175+ Item Performance Checklist

The book ships with a **175+ item performance checklist** that captures field‑tested optimizations covering the entire lifecycle. You can apply these immediately:

- ✅ Performance tuning mindset and cost optimization
- ✅ Reproducibility and documentation best practices
- ✅ System architecture and hardware planning
- ✅ Operating system and driver optimizations
- ✅ GPU programming and CUDA tuning
- ✅ Distributed training and network optimization
- ✅ Efficient inference and serving
- ✅ Power and thermal management
- ✅ Latest profiling tools and techniques
- ✅ Architecture-specific optimizations

### Links

- **Book**: [AI Systems Performance Engineering on Amazon](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)
- **Meetup**: [AI Performance Engineering](https://www.meetup.com/ai-performance-engineering)
- **YouTube**: [AI Performance Engineering Channel](https://www.youtube.com/@AIPerformanceEngineering)

> *Built in San Francisco for the AI performance engineering community*

### Key Focus Areas

- **GPU Architecture, PyTorch, CUDA, and OpenAI Triton Programming**
- **Distributed Training & Inference**
- **Memory Optimization & Profiling**
- **PyTorch Performance Tuning**
- **Multi-Node Scaling Strategies**

## Quick Start

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

# Profiling-friendly workloads
# Most examples use modest tensor sizes and short iteration counts so Nsight
# Systems, Nsight Compute, and the PyTorch profiler finish quickly. Increase the
# sizes if you need larger-scale numbers.
```

### Blackwell Workflow

The repository targets **NVIDIA Blackwell B200/B300 (SM100)**. CUDA builds,
Nsight workflows, and PyTorch stacks assume CUDA 12.9, PyTorch 2.9 nightlies, and
Triton 3.5.0. Helper scripts keep everything aligned:

```bash
# Build CUDA samples and run sanity checks
./code/build_all.sh

# Profile the entire codebase with Nsight + PyTorch profiler
python scripts/profile_harness.py --profile nsys --profile pytorch --output-root profiles/full_run

# Reset generated profiling artifacts
./clean_profiles.sh
```

For environment variables, validation scripts, and hardware guidance, see
`docs/environment.md`.

### Latest Stack Updates

- **PyTorch 2.9**: Enhanced compiler, dynamic shapes, improved profiler
- **CUDA 12.9**: Latest Blackwell features and kernel performance updates
- **Triton 3.5.0**: Architecture-specific kernels and optimizations
- **Nsight 2024.x**: Refreshed kernel and timeline analysis capabilities
- **HTA**: Holistic Tracing Analysis for multi-GPU systems
- **perf**: Enhanced system-level sampling workflows
- **Unified Profiling Harness**: One command covers Nsight Systems/Compute and the PyTorch profiler

## Documentation Map

- **Chapter guide**: See the section below for chapter-by-chapter themes and learning goals
- `docs/tooling-and-profiling.md`: Nsight, HTA, perf, and harness workflows
- `docs/environment.md`: Blackwell stack requirements, env vars, validation, and tooling

## Chapter Guide

This guide captures the context, themes, and focus areas for each chapter in the
book. It mirrors the narrative that previously lived in `docs/chapter-guide.md`.

### Chapter 1: Introduction and AI System Overview

- The AI Systems Performance Engineer
- Benchmarking and Profiling
- Scaling Distributed Training and Inference
- Managing Resources Efficiently
- Cross-Team Collaboration
- Transparency and Reproducibility

### Chapter 2: AI System Hardware Overview

- The CPU and GPU "Superchip"
- NVIDIA Grace CPU & Blackwell GPU
- NVIDIA GPU Tensor Cores and Transformer Engine
- Streaming Multiprocessors, Threads, and Warps
- Ultra-Scale Networking
- NVLink and NVSwitch
- Multi-GPU Programming

### Chapter 3: OS, Docker, and Kubernetes Tuning

- Operating System Configuration
- GPU Driver and Software Stack
- NUMA Awareness and CPU Pinning
- Container Runtime Optimizations
- Kubernetes for Topology-Aware Orchestration
- Memory Isolation and Resource Management

### Chapter 4: Tuning Distributed Networking Communication

- Overlapping Communication and Computation
- NCCL for Distributed Multi-GPU Communication
- Topology Awareness in NCCL
- Distributed Data Parallel Strategies
- NVIDIA Inference Transfer Library (NIXL)
- In-Network SHARP Aggregation

### Chapter 5: GPU-based Storage I/O Optimizations

- Fast Storage and Data Locality
- NVIDIA GPUDirect Storage
- Distributed, Parallel File Systems
- Multi-Modal Data Processing with NVIDIA DALI
- Creating High-Quality LLM Datasets

### Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy

- Understanding GPU Architecture
- Threads, Warps, Blocks, and Grids
- CUDA Programming Refresher
- Understanding GPU Memory Hierarchy
- Maintaining High Occupancy and GPU Utilization
- Roofline Model Analysis

### Chapter 7: Profiling and Tuning GPU Memory Access Patterns

- Coalesced vs. Uncoalesced Global Memory Access
- Vectorized Memory Access
- Tiling and Data Reuse Using Shared Memory
- Warp Shuffle Intrinsics
- Asynchronous Memory Prefetching

### Chapter 8: Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism

- Profiling and Diagnosing GPU Bottlenecks
- Nsight Systems and Compute Analysis
- Tuning Occupancy
- Improving Warp Execution Efficiency
- Exposing Instruction-Level Parallelism

### Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity

- Multi-Level Micro-Tiling
- Kernel Fusion
- Mixed Precision and Tensor Cores
- Using CUTLASS for Optimal Performance
- Inline PTX and SASS Tuning

### Chapter 10: Intra-Kernel Pipelining and Cooperative Thread Block Clusters

- Intra-Kernel Pipelining Techniques
- Warp-Specialized Producer-Consumer Model
- Persistent Kernels and Megakernels
- Thread Block Clusters and Distributed Shared Memory
- Cooperative Groups

### Chapter 11: Inter-Kernel Pipelining and CUDA Streams

- Using Streams to Overlap Compute with Data Transfers
- Stream-Ordered Memory Allocator
- Fine-Grained Synchronization with Events
- Zero-Overhead Launch with CUDA Graphs

### Chapter 12: Dynamic and Device-Side Kernel Orchestration

- Dynamic Scheduling with Atomic Work Queues
- Batch Repeated Kernel Launches with CUDA Graphs
- Dynamic Parallelism
- Orchestrate Across Multiple GPUs with NVSHMEM

### Chapter 13: Profiling, Tuning, and Scaling PyTorch

- NVTX Markers and Profiling Tools
- PyTorch Compiler (torch.compile)
- Profiling and Tuning Memory in PyTorch
- Scaling with PyTorch Distributed
- Multi-GPU Profiling with HTA

### Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends

- PyTorch Compiler Deep Dive
- Writing Custom Kernels with OpenAI Triton
- PyTorch XLA Backend
- Advanced Triton Kernel Implementations

### Chapter 15: Multi-Node Inference Parallelism and Routing

- Disaggregated Prefill and Decode Architecture
- Parallelism Strategies for MoE Models
- Speculative and Parallel Decoding Techniques
- Dynamic Routing Strategies

### Chapter 16: Profiling, Debugging, and Tuning Inference at Scale

- Workflow for Profiling and Tuning Performance
- Dynamic Request Batching and Scheduling
- Systems-Level Optimizations
- Quantization Approaches for Real-Time Inference
- Application-Level Optimizations

### Chapter 17: Scaling Disaggregated Prefill and Decode

- Prefill-Decode Disaggregation Benefits
- Prefill Workers Design
- Decode Workers Design
- Disaggregated Routing and Scheduling Policies
- Scalability Considerations

### Chapter 18: Advanced Prefill-Decode and KV Cache Tuning

- Optimized Decode Kernels (FlashMLA, ThunderMLA, FlexDecoding)
- Tuning KV Cache Utilization and Management
- Heterogeneous Hardware and Parallelism Strategies
- SLO-Aware Request Management

### Chapter 19: Dynamic and Adaptive Inference Engine Optimizations

- Adaptive Parallelism Strategies
- Dynamic Precision Changes
- Kernel Auto-Tuning
- Reinforcement Learning Agents for Runtime Tuning
- Adaptive Batching and Scheduling

### Chapter 20: AI-Assisted Performance Optimizations

- AlphaTensor AI-Discovered Algorithms
- Automated GPU Kernel Optimizations
- Self-Improving AI Agents
- Scaling Toward Multi-Million GPU Clusters

## Community Resources

Monthly meetups with 100k+ members across 20+ cities:

- [YouTube Channel](https://www.youtube.com/@AIPerformanceEngineering)
- [Meetup Group](https://www.meetup.com/ai-performance-engineering)

Recent sessions:

- [Dynamic Adaptive RL Inference CUDA Kernel Tuning](resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf)
- [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)
- [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf)

### Monthly Meetup Summaries

- **September 15, 2025** – [YouTube](https://www.youtube.com/watch?v=eLnHXL1xXfM): Dynamic Adaptive RL inference kernel tuning deep dive; companion slides in `resources/Dynamic_Adaptive_RL_Inference_CUDA_Kernel_Tuning.pdf`.
- **August 18, 2025** – [YouTube](https://www.youtube.com/watch?v=SBPlOUww57I): Multi-GPU orchestration strategies and Nsight profiling case studies.
- **July 21, 2025** – [YouTube](https://youtu.be/jaiMotxv8ck): FlashMLA, ThunderMLA, and FlexDecoding kernel walkthroughs with live Nsight Compute demos.
- **June 16, 2025** – Slides: [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf) covering disaggregated inference routing.
- **May 19, 2025** – [YouTube](https://youtu.be/F8jJwI9xHTE) & [PyTorch Data Loader Optimization](resources/PyTorch_Model_Optimization_Data_Loader.pdf): Torch.compile pipelines, data loader throughput tuning, and cross-architecture CUDA/ROCm kernels.
- **April 21, 2025** – [YouTube](https://youtu.be/XoZcY_fDUKA) & [AI Performance Engineering Meetup Slides](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf): End-to-end GPU performance playbook plus the [PyTorch Model Optimization](resources/PyTorch_Model_Optimization.pdf) workshop.

## Contributing

Contributions are welcome! See `CONTRIBUTING.md` for guidelines on code,
documentation, and performance improvements.

## License

MIT License – see `LICENSE` for details.
