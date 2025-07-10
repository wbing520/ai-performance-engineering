# AI Systems Performance Engineering
## O'Reilly Book - Summer 2025

https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/

[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

# Meetup and YouTube Links
* Monthly Meetup: https://www.meetup.com/ai-performance-engineering
* YouTube Videos: https://www.youtube.com/@AIPerformanceEngineering

# O'Reilly Book - Table of Contents

Book: https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/

# Monthly Meetups (100,000 Global Members, 20+ Cities)
* https://www.meetup.com/ai-performance-engineering

## Apr 21, 2025
* [YouTube Video](https://youtu.be/XoZcY_fDUKA)
* [AI_Performance_Engineering_Meetup_Apr_21_2025](resources/AI_Performance_Engineering_Meetup_Apr_21_2025.pdf)
* [PyTorch_Model_Optimization](resources/PyTorch_Model_Optimization.pdf)

## May 19, 2025
* [YouTube Video](https://youtu.be/F8jJwI9xHTE)
* [PyTorch Optimizations: Data Loader Pipeline](resources/PyTorch_Model_Optimization_Data_Loader.pdf)
* [Cross-Architeecture CUDA and ROCm Kernel Development](resources/ai_perf_eng_meetup.pdf)

## June 16, 2025
* [YouTube Video]()
* [High Performance Agentic AI Inference Systems](resources/High_Performance_Agentic_AI_Inference_Systems.pdf)

## Chapter 1: Introduction and AI System Overview
* The AI Systems Performance Engineer
* Benchmarking and Profiling
* Scaling Distributed Training and Inference
* Managing Resources Efficiently
* Cross-Team Collaboration
* Transparency and Reproducibility
* Towards 100-Trillion-Parameter Models
* NVIDIA’s “AI Supercomputer in a Rack”
* Mechanical Sympathy: Hardware-Software Co-Design
* Measuring “Goodput” Useful Throughput
* Book Roadmap and Methodology

## Chapter 2: AI System Hardware Overview
* The CPU and GPU “Superchip”
* NVIDIA Grace CPU
* NVIDIA Blackwell GPU
* NVIDIA GPU Tensor Cores and Transformer Engine
* Streaming Multiprocessors, Threads, and Warps
* Ultra-Scale Networking Treating Many GPUs as One
* NVLink and NVSwitch
* Multi-GPU Programming
* In-Network Aggregations with NVIDIA SHARP
* Multi-Rack and Storage Communication
* Pre-Integrated Rack Appliance
* Co-Packaged Optics: Future of Networking Hardware
* Compute Density and Power Requirements
* Liquid Cooling vs. Air Cooling
* Performance Monitoring and Utilization in Practice
* Sharing and Scheduling
* ROI of Upgrading Your Hardware
* A Glimpse into the Future: NVIDIA’s Roadmap
* Blackwell Ultra and Grace-Blackwell Ultra
* Vera-Rubin Supership (2026)
* Rubin Ultra and Vera-Rubin Ultra (2027)
* Feynman GPU (2028) and Doubling Something Every Year

## Chapter 3: OS, Docker, and Kubernetes Tuning for GPU-based Environments
* Operating System
* GPU Driver and Software Stack
* GPU Driver
* CUDA Toolkit and Runtime
* C++ and Python CUDA Libraries
* PyTorch and Higher-Level AI Frameworks
* Configuring the CPUs and OS for GPU Environments
* NUMA Awareness and CPU Pinning
* NUMA-Friendly Memory Allocation and Memory Pinning
* Transparent Huge Pages
* Scheduler and Interrupt Affinity
* Virtual Memory and Swapping
* Filesystem Caching and Write-Back
* CPU Frequency and C-states
* Tune Host CPU Memory Allocator
* GPU Driver and Runtime Settings for Performance
* GPU Persistence Mode
* Multi-Process Service
* Multi-Instance GPU
* GPU Clock Speeds and Error Correcting Code
* GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling
* Container Runtime Optimizations for GPUs
* NVIDIA Container Toolkit and CUDA Compatibility
* NVIDIA Container Runtime
* Avoiding Container Overlay Filesystem Overhead
* Reduce Image Size for Faster Container Startup
* Kubernetes for Topology-Aware Container Orchestration and Networking
* Orchestrating Containers with Kubernetes Topology Manager
* Job Scheduling with Kubernetes and SLURM
* Slicing a GPU with Multi-Instance GPU
* Optimizing Network Communication for Kubernetes
* Reducing Kubernetes Orchestration Jitter
* Improving Resource Guarantees
* Memory Isolation and Avoiding the OOM Killer
* Dealing with I/O Isolation

## Chapter 4: Tuning Distributed Networking Communication
* Overlapping Communication and Computation
* Asynchronous Execution with Streams
* Reducing Communication Frequency and Volume
* Achieving Maximal Overlap in Practice
* NVIDIA Magnum IO Optimization Stack
* High-Speed, Low-Overhead Data Transfers with RDMA
* Tuning Multi-Node Connectivity
* Multi-Node Communication Pitfalls
* NCCL for Distributed Multi-GPU Communication
* Topology Awareness in NCCL
* NCCL Communication Algorithms
* Choosing the Right Parallelism: DataParallel vs DistributedDataParallel
* NCCL Communicator Lifecycle and Environment Gotchas
* Profiling and Debugging NCCL
* In-Network SHARP Aggregation
* NIXL and Key-Value Cache Offloading for Inference
* Separate Prefill and Decode Inference Stages
* Intelligent Interconnect Routing for KV Cache Transfers
* NIXL Asynchronous API with Callbacks
* KV-Cache Offloading with NIXL
* NIXL and High-Performance Inference Systems like NVIDIA Dynamo
* NCCL vs. NIXL

## Chapter 5: GPU-based Storage I/O Optimizations
* Fast Storage and Data Locality
* Sequential vs. Random Read Patterns
* Tuning NVMe and Filesystem for Throughput
* Using NVIDIA GPUDirect Storage
* Measuring GPUDirect Storage with gdsio
* DeepSeek’s Fire-Flyer File System (3FS)
* Distributed, Parallel File Systems and Object Stores
* Tuning, Replicating, and Compressing Data
* Monitoring Storage I/O
* Tuning the Data Pipeline
* Efficient Data Loading and Preprocessing
* Scaling Out Workers as you Scale Out Number of GPUs
* Multi-Modal Data Processing with NVIDIA DALI
* Creating High-Quality LLM Datasets with NVIDIA NeMo Curator
* Continuous Profiling and Tuning Workflow
* Diagnosing Communication vs. Compute Bound Workloads

## Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy
* Understanding GPU Architecture
* Threads, Warps, Blocks, and Grids
* Choosing Threads-per-Block and Blocks-per-Grid Sizes
* CUDA Programming Refresher
* Configuring Launch Parameters: Blocks Per Grid and Threads Per Block
* 2D and 3D Kernel Inputs
* Asynchronous Memory Allocation and Memory Pools
* Understanding GPU Memory Hierarchy
* Unified Memory
* Maintaining High Occupancy and GPU Utilization
* Tuning Occupancy with Launch Bounds
* Roofline Model: Compute-Bound or Memory-Bound Workloads

## Chapter 7: Profiling and Tuning GPU Memory Access Patterns
* Avoid Warp Divergence
* Coalesced vs. Uncoalesced Global Memory Access
* Vectorized Memory Access
* Avoid Shared-Memory Bank Conflicts
* Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization
* Tiling and Data Reuse Using Shared Memory
* Read-Only Data Caches
* Asynchronous Memory Prefetching and Tensor Memory Accelerator

## Chapter 8: Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism
* Profiling and Diagnosing GPU Bottlenecks
* Nsight Systems Timeline View
* Profiling and Tuning the Data Pipeline
* Nsight Compute and Roofline Analysis
* PyTorch Profiler and Visualization Tools
* Profiler-Guided Analysis
* Analyzing Warp Stall Reasons with Nsight Compute
* Memory-Related Stalls
* Execution Dependency Stalls
* Execution Unit Contention
* Other Stall Reasons
* Inspecting Achieved Occupancy and GPU Utilization
* Kernel Memory Throughput vs. Peak HBM Memory Bandwidth
* Kernel Compute Throughput vs. Peak GPU FLOPs
* Iteratively Profiling and Determining the Kernel Bottleneck
* Optimizing the Kernel
* Tuning Occupancy
* Find the Right Occupancy for your Workload
* Techniques for Occupancy Tuning
* Compiler Hints to Optimize Occupancy
* Determine Optimal Launch Configuration with the Occupancy API
* Tuning Occupancy with PyTorch
* Profiling and Improving Warp Execution Efficiency
* Detecting Warp Divergence
* Causes of Warp Divergence
* Profiling and Eliminating Warp Divergence with Predication
* Efficient Intra-Warp Communication with Warp Intrinsics
* PyTorch Considerations for Warp-Level Efficiency
* Exposing Instruction-Level Parallelism
* Warp Scheduling and Dual Issue Instructions
* ILP and Occupancy
* Loop Unrolling, Interleaving, and Compiler Hinting
* Profiling and Mitigating Register Pressure

## Chapter 9: Increasing CUDA Kernel Efficiency and Arithmetic Intensity
* Multi-Level Micro-Tiling and Software Prefetching
* Tiling with Thread Block Clusters
* Kernel Fusion
* Structured Sparsity
* Recomputation vs. Memory Trade-Off
* PyTorch and Arithmetic Intensity
* Mixed Precision and Utilizing Tensor Cores
* Feeding Tensor Cores with TMEM and TMA
* TF32 and Automatic Mixed Precision (PyTorch)
* BF16/FP16, FP8, and FP4 Reduced Precision
* INT8 Reduced Precision and DP4A Instructions for Inference
* Transformer Engine and TMEM and WMMA In-Depth
* Tensor Cores and Thread Block Cluster Pairs (CTA Pairs)
* Impact of Converting a GEMM from FP32 to Mixed Precision
* Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance
* Inline PTX and SASS Tuning for Micro-Optimizations
* DeepSeek’s Use of Inline PTX for Memory Allocation Optimization

## Chapter 10: Intra-Kernel Pipelining, Warp Specialization, and Cooperative CTA Clusters
* Intra-Kernel Pipelining Techniques
* Cooperative Tiling and Double-Buffering with the CUDA Pipeline API
* Warp-Specialized Producer-Consumer Model with CUDA Pipeline API
* Using CUDA Pipeline API for Warp Specialization
* PyTorch, CUDA Pipeline API, and Warp Specialization
* Cooperative Groups, Persistent Kernels, and Megakernels
* Cooperative Groups
* Persistent Kernels and Megakernels
* Combining Cooperative Grid Synchronization and Persistent Kernels
* Thread Block Clusters (CTA Clusters) and Distributed Shared Memory (DSM)
* Distributed Shared Memory
* Scratch Memory
* Launching a CTA Cluster
* Coordinating with Cooperative Groups API
* CTA Pairs
* Reducing Global Memory Traffic with CTAs
* Designing Efficient Algorithms with CTA Clusters
* Warp Specialization with CTA Clusters

## Chapter 11: Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations
* Using Streams to Overlap Compute with Data Transfers
* The Stream-Ordered Memory Allocator
* Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs
* Legacy Default Stream
* Modern Per-Thread Default Stream
* Default vs. Explicit (Non-Default) Streams
* Default Stream Best Practices
* Fine-Grained Synchronization with Events and Callbacks
* Using CUDA Events for Cross-Stream Synchronization
* Combining Intra-Kernel Overlap (Warp Specialization) and Inter-Kernel Overlap (CUDA Streams)
* Combining CTA Clusters with Warp Specialization and CUDA Streams
* Multi-GPU Compute and Data Transfer Overlap with CUDA Streams
* Zero-Overhead Launch with CUDA Graphs

## Chapter 12: Dynamic and Device-Side Kernel Orchestration with CUDA Graphs
* Dynamic Scheduling with Atomic Work Queues
* Atomic Counters
* Dynamic Work Distribution with Atomic Counters
* Batch Repeated Kernel Launches with CUDA Graphs
* Dynamic Graph Updates
* Device-Initiated Programmatic Device Launch (PDL) Graphs
* Dynamic Parallelism: Offload Kernel Dispatch From the CPU to the GPU
* Orchestrate Across Multiple GPUs and Cluster Nodes
* Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM
* Capturing Multi-GPU Collectives with NCCL and CUDA Graphs
* Pattern for N-GPU Scaling
* Roofline-Guided Scheduling and Orchestration Decisions

## Chapter 13: Profiling, Tuning, and Scaling PyTorch
* Profiling and NVTX Instrumentation
* Profiling PyTorch to Identify Bottlenecks
* Using PyTorch Profiler
* System Profiling with Nsight Systems and NVTX Timelines
* Kernel Roofline Analysis with Nsight Compute
* CPU and GPU Profiling with Linux perf
* PyTorch Compiler (torch.compile)
* Using PyTorch Compiler
* Compiling versus Writing Custom Kernels
* Compilation Modes and Trade-offs in Speed, Memory, and Compile Time
* Profiling and Debugging Compiler Performance Issues
* PyTorch Optimized Attention Mechanisms
* PyTorch Architecture Optimization (torch.ao), Quantization, Sparsity, and Pruning
* Concurrency with CUDA Streams
* Overlapping Communication and Computation
* Stream Synchronization with Events
* Reducing Kernel Launch Overhead with CUDA Graphs
* Capturing a CUDA Graph and Pre-Allocating Memory
* Replaying the Graph
* Best Practices for CUDA Graphs
* Profiling and Tuning Memory in PyTorch
* Tuning the CUDA Memory Allocator
* Activation/Gradient Checkpointing for Memory Savings
* Offloading Parameters to CPU and NVMe
* FSDP Automatic Checkpointing and Offloading
* Pluggable Memory Allocators and Cross-GPU Data Transfers
* Enabling Peer-to-Peer (P2P) DMA and UCX
* Optimizing the Data Input Pipeline
* Scaling with PyTorch Distributed
* DDP with torch.compile
* FSDP with torch.compile
* Tensor and Pipeline Parallelism with torch.compile
* Multi-GPU Profiling with Holistic Tracing Analysis (HTA)
* Continuous Integration and Performance Benchmarking
* PyTorch Heads-Up-Display (HUD) Performance Dashboard
* Performance Benchmarks and MLPerf Logging

## Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends
* PyTorch Compiler Deep Dive
* TorchDynamo for Bytecode Capture and Graph Extraction
* AOTAutograd Ahead-of-Time Fusion for Forward and Backward Passes
* Prims IR (PrimTorch IR) Simplified Operator Set
* TorchInductor Backend Code Generation
* Autotuning with TorchInductor
* Dynamic Shapes and Variable Sequence Lengths
* Regional Compilation
* NVFuser and Legacy PyTorch JIT Compilation
* Disabling PyTorch Compiler and Reverting Back to Eager Mode
* Performance Hints and Debugging Generated Code
* Debugging Numerical Correctness and Accuracy
* Explaining and Minimizing Graph Breaks
* Graph Breaks and torch._dynamo.explain
* Minimize Graph Recompilations
* Mark Functions and Code Blocks as Safe with allow_in_graph
* Tips for Handling Graph Breaks
* Debugging Compiler Phases, Graph Breaks, and Performance
* Writing Custom Kernels with OpenAI Triton
* Triton Programming Model
* Accessing Shared Memory in Triton
* Registering Custom Kernels with PyTorch
* Tuning Kernel Launch Parameters
* Auto-Tuning Triton Kernels
* Profiling with Triton Proton Profiler
* Advanced Triton Kernel Implementations
* Warp Specialization with Triton
* Persistent Matmul Kernel (Single-Kernel Tiling)
* Software Pipelining and Double-Buffering with Triton
* Accessing Tensor Cores with Warp-Level Matrix Multiply and Inline PTX
* PyTorch XLA Backend

## Chapter 15: Multi-Node Inference Parallelism, Decoding, and Routing Optimizations
* Disaggregated Prefill and Decode Architecture
* Parallelism Strategies for Serving Massive MoE Models
* Tensor Parallelism
* Pipeline Parallelism
* Expert Parallelism
* Data Parallelism
* Context (Sequence) Parallelism
* Hybrid Parallelism
* Speculative and Parallel Decoding Techniques
* Two-Model Speculative Decoding and EAGLE
* Single-Model Self-Speculative Decoding
* Multi-Token Decoding with Medusa’s Multiple Heads
* Interleaving Decode Steps from Multiple Requests
* Combining Decoding Techniques and Evaluating Complexity
* Constrained Decoding Performance Implications
* Dynamic Routing Strategies for MoE Inference
* Expert Communication Optimization
* Load Balancing, Capacity Factor, and Expert Replication
* Adaptive MoE Routing and Real-Time Expert Monitoring

## Chapter 16: Profiling, Debugging, and Tuning Inference at Scale
* Workflow for Profiling, Debugging, and Tuning Performance
* Monitoring System Metrics, Counters, and LModern NVIDIA GPUs support low-progs
* Profiling with Nsight Systems and Nsight Compute
* Inference Troubleshooting Recipes
* Full-Stack Inference Optimizations
* Debugging Correctness Issues
* Dynamic Request Batching, Scheduling, and Routing
* Dynamic Request Batching
* Latency-Aware Scheduling and Dynamic Routing
* Stall-Free Scheduling (Chunked Prefill)
* Continuous Batching
* Continuous Scheduling and Concurrent Model Streams
* Systems-Level Optimizations
* Overlapping Communication and Computation
* Maximizing GPU Utilization and Throughput vs. Latency Trade-offs
* Power and Thermal Constraints
* Error Handling
* Memory
* KV Cache Offloading and Memory Pool Allocation
* Quantization Approaches for Real-Time Inference
* Reducing Precision From FP16 Down to FP8/FP4
* Weight-Only Quantization (GPTQ, AWQ)
* Activation Quantization
* Post-Training Quantization Workflow
* Combining Weight and Activation Quantization
* Application-Level Optimizations
* Prompt Compression
* Prompt Cleansing
* Prefix Caching
* Model Cascading and Tiered Model Deployment
* Streaming Responses
* Debouncing and Request Coalescing
* Token Output Limits and Timeouts

## Chapter 17: Scaling Disaggregated Prefill and Decode for Inference
* Why Prefill-Decode Disaggregation?
* Advantages of Disaggregation
* Reduced Interference
* Phase-Specific Optimizations
* Disaggregated Cluster Architecture
* Prefill Workers Design
* Dynamic Batching
* Memory Management
* Optimizing for Latency versus Throughput
* Latency-Aware Scheduling and Batching
* Decode Workers Design
* Continuous Batching
* Grouping Variable-Length Sequences
* Memory Management for the KV Cache
* Disaggregated Routing and Scheduling Policies
* Routing Factors
* Example Dynamic Routing Policy in Code
* Example Dynamic Routing Policy Configuration
* Latency-Aware Routing and Multi-Path Inference
* Multi-Branch, Parallel Speculative Decoding Across Workers
* Quality-of-Service and Early Rejection Policies
* Conditional Routing in Practice
* Scalability of Disaggregated Prefill-Decode

## Chapter 18: Advanced Prefill-Decode and KV Cache Tuning
* Optimized Decode Kernels
* FlashMLA (DeepSeek)
* ThunderMLA (Stanford)
* FlexDecoding (PyTorch)
* Tuning KV Cache Utilization and Management
* Disaggregated KV Cache Pool
* KV Cache Reuse and Prefix Sharing
* Optimized KV Cache Memory Layout
* GPU Memory Bandwidth Improvements
* Fast KV Cache Transfer Between Prefill and Decode
* KV Cache Size
* Zero-Copy GPU-to-GPU Transfer
* Connector and Data Path Design
* Heterogeneous Hardware and Parallelism Strategies for Prefill and Decode
* Compute-Optimized versus Memory-Optimized Hardware
* Throughput and Cost Benefits
* Phase-Specific Model Parallelism
* Different Precision for Prefill and Decode
* Hybrid Prefill with GPU–CPU Collaboration
* SLO-Aware Request Management and Fault Tolerance
* Early Rejection (Admission Control)
* Quality of Service (QoS)
* Fault Tolerance
* Dynamic Scheduling and Load Balancing
* Adaptive Resource Scheduling and Hotspot Prevention
* Dynamic Resource Scaling

## Chapter 19: Dynamic and Adaptive Inference Engine Optimizations
* Dynamic Parallelism Switching (TP vs. PP vs. Hybrid)
* Dynamic Precision Switching (FP8 ⇆ FP4 on the Fly)
* Kernel Auto-Tuning for Transformer Self-Attention and MLP Paths
* Dynamic Shared Memory Allocation and Occupancy-Aware Kernel Selection
* Speculative KV Prefetching and Cache Routing for Faster TTFT
* Real-Time KV Cache Compression and Policy Switching
* Runtime Reinforcement Learning Agents for Tuning
* Dynamic Memory-Allocation Switching (Slab vs. Caching vs. Stream-Ordered)
* Runtime Kernel Performance Improvements and Hot-Swappable Implementations
* Continuous Prewarming of CUDA Graphs and Caches using Time-Series Prediction
* Adaptive Batching and Prefill-Decode Disaggregation Strategies
* Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs
* NVLink/NVSwitch Topology and Bandwidth Constraints
* Real-Time Link Telemetry and Monitoring
* Adaptive Pipeline Stage Remapping
* Optimizing Collective Communication with NCCL
* Multi-Node Communication and GPUDirect RDMA
* Localizing Communication using MoE Expert Shuffling
* Dynamic Congestion-Aware Scheduling
* Coordinating NVSwitch Transfers with Fine-Tuned Scheduling
* Additional Adaptive and Dynamic Optimization Techniques
* Dynamic Early Exit Networks
* Input-Aware Layer Skipping (DASH)
* Speculative MoE Expert Routing and Communication Reduction
* Dynamic Token Pruning with LazyLLM
* Edge-Oriented MoE Memory Budgeting
* Dynamic Quantization and Activation Range Adjustment

## Chapter 20: Profiling and Tuning High-Performance Agentic Systems
* Fundamental Agent Protocols
* Model Context Protocol (MCP)
* Agent-to-Agent (A2A) Protocol
* Protocols and Agent Performance
* Architecture of an Agent System
* User Interface / Frontend API
* Agent Orchestrator
* MCP Client and Tool Integrations
* LLM Inference Backend
* Self-Hosted Model
* Remote API (e.g. OpenAI API or AWS Lambda)
* Hybrid Self-Hosted and External API
* A2A Multi-Agent Coordination
* Assembling the Response
* Performance Metrics for Agent Workflows
* Time to First Token (TTFT)
* Time per Output Token (TPOT)
* End-to-End Latency
* Queries per Second (QPS) Throughput
* Goodput
* System Metrics

## Chapter 21: Agent Profiling and Observability Techniques
* Distributed Tracing with OpenTelemetry
* Aggregate Metrics and Logs
* CPU and Memory Profiling
* Python Profiler
* Node.js Profiler
* Asynchronous Profiler
* GPU Profiling
* GPU Usage
* System-Wide CPU and GPU Profiling with Nsight Systems
* Kernel-Level GPU Profiling with Nsight Compute
* End-to-End Request-Response Breakdown
* Optimizing MCP Communication and Tool Integration
* Minimize Round Trips
* Asynchronous Streaming
* Connection Reuse
* Optimize MCP Server Performance
* Security Overhead
* Limit Payload Size
* Optimizing LLM Inference
* Time-to-First-Token (TTFT) Latency Optimizations
* Avoid Overly Long Prompts
* Streaming API Usage
* Model Size and Type
* Continuous Batching Effects
* Time-Per-Output-Token (TPOT) Throughput Optimizations
* Use Hardware Acceleration
* Embrace Quantization
* Parallelism and Scalability
* Efficient Batch Scheduling
* Model-Specific Optimizations
* Limit Max Output Length
* Parallelizing Non-LLM Work
* External API Usage Optimization (e.g. OpenAI)
* Route to a Faster, Smaller, Cheaper Model
* Use Batched Endpoints
* Minimize Network Latency to External API Provider
* Track Token Usage

## Chapter 22: Orchestration and Multi-Agent Coordination (A2A) Performance
* Asynchronous Orchestration and Concurrency
* Optimizing A2A Interactions
* Reduce Discovery Overhead
* Minimize Task Handoff Cost
* Parallelize Agent Calls
* Timeouts and Fallbacks
* Concurrent Task Handling in Agents
* Lightweight Protocol Handling
* Security Context Propagation
* Deployment Optimizations with Kubernetes and Serverless Functions
* Kubernetes-based Agent, LLMs, and Tools
* Resource Allocation
* Horizontal Scaling
* Concurrency and Workers
* Networking and I/O
* Kubernetes-specific Optimizations
* Monitoring
* Cold Start Problem
* Serverless Functions for Agents, LLMs, and Tools
* Function Runtimes Lack Modern GPU Support
* Memory and CPU for each Serverless Function
* Concurrency Control
* External Calls from Serverless Functions
* Stateless Design and Caching
* Step Functions vs Orchestration
* Cold Start Problem
* Edge Inference and Hybrid Architectures
* Split Inference Between Edge and Cloud
* Edge Caching and Locality
* Prefix Cache Locality
* KV Prefetch
* Compression
* Edge Fallbacks
* Synchronization and Updates
* Quantization and Hardware Considerations
* Monitoring and Debugging at the Edge

## Chapter 23: AI System Performance Optimization Case Studies
* OpenAI’s Journey to Train GPT 4.5 with New Hardware at Ultra-Scale
* DeepSeek Scales to 671-Billion Parameter Model Despite Hardware Constraints
* MobileEye Improves GPU Performance Using FP8 Precision with PyTorch
* Boost Performance with Open Source NVIDIA Dynamo Inference Server
* Efficient Inference with vLLM: High Throughput at Lower Cost
* DeepMind’s AlphaTensor: AI-Discovered Algorithms Boosting GPU Performance
* NVIDIA’s AI-Assisted GPU Kernel Optimizations with DeepSeek-R1
* Sakana.ai’s Agent: LLMs That Write 100× Faster GPU Kernels
* Predibase’s Reinforcement Learning Approach to Generating Optimized GPU Kernels
* NVIDIA Grace-Hopper (GH200) Superchip Performance Compared to Hopper (H100)
* High-Speed Inference with the Grace-Blackwell NVL72 Rack System
* Faster Experiments and Insights for Trillion-Parameter LLMs with Grace-Blackwell Clusters
* HPE’s Grace-Blackwell Supercomputer for the Trillion-Parameter Era
* Training and Serving a 100-Trillion-Parameter Model

## Chapter 24: Future Trends in Ultra-Scale AI Systems Performance Engineering
* Convergence of AI and Scientific Computing
* Massive Data Centers Powering Self-Improving, Perpetually-Learning AI Agents
* Smart Compilers and Automated Code Optimizations
* AI-Assisted Real-Time System Optimizations and Cluster Operations
* Sparsity and Conditional Computation as First-Class Citizens
* GPUs Performing More Tasks Across the Entire AI Pipeline
* High-Throughput, Low-Latency Inference Orchestration
* Silicon Photonics and Optical Interconnects
* Globally Distributed Data Centers (AI Factories)
* Smart Networking and Distributed Compute Offload
* HBM Memory Stacked on GPUs (3D GPUs)
* Energy Efficiency and Sustainability at Scale
* Advanced Cooling and Energy Reuse Techniques
* Hybrid Classical-Quantum Computing (CUDA Quantum)
* Scaling Toward 100-Trillion-Parameter Models

## Chapter 25: 250+ Tips for AI Systems Performance Engineers
* Performance Tuning Mindset and Cost Optimization
* Reproducibility and Documentation Best Practices
* System Architecture and Hardware Planning
* Grace-Blackwell GB200 Unified Architecture
* Multi-GPU Scaling and Interconnect Optimizations
* Operating System and Driver Optimizations
* GPU Resource Management and Scheduling
* Data Pipeline and I/O Optimization
* Workload Profiling and Monitoring
* GPU Programming and CUDA Tuning Optimizations
* Data Pipeline and Storage Tips
* Precision and Arithmetic Optimizations
* Advanced Strategies and Algorithmic Tricks
* Distributed Training and Network Optimization
* Efficient Inference and Serving
* Power and Thermal Management
