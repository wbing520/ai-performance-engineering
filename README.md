# AI Systems Performance Engineering
## O'Reilly Book - Summer 2025
[![O'Reilly Book](img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/

# Meetup and YouTube Links
* Monthly Meetup: https://www.meetup.com/ai-performance-engineering
* YouTube Videos: https://www.youtube.com/@AIPerformanceEngineering

# O'Reilly Book - Table of Contents
## Chapter 1: Introduction and AI System Overview
* The AI Systems Performance Engineer
* Towards 100-Trillion-Parameter Models
* NVIDIA’s “AI Supercomputer in a Rack”
* Mechanical Sympathy: Hardware-Software Co-Design
* Measuring “Goodput” Useful Throughput
* Book Roadmap and Methodology
* Key Takeaways
* Conclusion

## Chapter 2: AI System Hardware Overview
* The CPU and GPU “Superchip”
* Ultra-Scale Networking Treating Many GPUs as One
* Compute Density and Power Requirements
* Liquid Cooling vs. Air Cooling
* Performance Monitoring and Utilization in Practice
* Sharing and Scheduling
* ROI of Upgrading Your Hardware
* A Glimpse into the Future: NVIDIA’s Roadmap
* Key Takeaways
* Conclusion

## Chapter 3: OS, Docker, and Kubernetes Tuning for GPUs
* Operating System
* GPU Driver and Software Stack
* Configuring the CPUs and OS for GPU Environments
* GPU Driver and Runtime Settings for Performance
* Container Runtime Optimizations for GPUs
* Kubernetes for Topology-Aware Container Orchestration and Networking
* Key Takeaways
* Conclusion

## Chapter 4: Distributed Communication and I/O Optimizations
* Overlapping Communication and Computation
* NVIDIA Magnum IO Optimization Stack
* High Speed, Low Overhead Data Transfers with RDMA
* NCCL for Distributed, Multi-GPU Communication
* NIXL for Accelerated Data Transfer and Inference
* Storage I/O Optimizations
* Tuning the Data Pipeline
* Key Takeaways
* Conclusion

## Chapter 5: Optimizing Memory Access and Data Movement
* Coalesced Global Memory Access
* Data Reuse and Tiling
* Shared Memory Bank Conflicts
* Aligned and Scalar Access vs. Vectorized Loads
* Read-Only Data Caches
* Tensor Memory Accelerator (TMA)
* Avoid CPU-GPU Transfer Bottlenecks
* Key Takeaways
* Conclusion

## Chapter 6: Tuning Compute Strategy and Parallelism
* Using Tensor Cores and Transformer Engines
* Instruction-Level Parallelism (Loop Unrolling)
* Warp Specialization and Producer-Consumer Parallelism
* Avoid Warp Branch Divergence
* Eliminating Redundant Computation and Memory Access
* Imbalanced Work Distribution and Persistent Kernels
* CUDA Graphs for Launch Overhead Reduction
* Asynchronous Data Transfers and Memory Prefetching
* Key Takeaways
* Conclusion

## Chapter 7: Improve Kernel Scheduling, Execution, Orchestration
* Avoid Excessive Global Synchronization
* Sequential Host-Driven Execution vs. Streams Overlap
* High Kernel Launch Overhead and Kernel Fusion
* Improve Kernel Batching (CUDA Graphs)
* Increasing GPU Utilization with Multi-Stream Kernels
* Reduce Kernel Overhead with Persistent Kernels
* Reduce Device-Launch Latency (Dynamic Parallelism)
* Balance Work Distribution and Dynamic Load Balancing
* Reduce Throttling by Insufficient CPU Submission
* Key Takeaways
* Conclusion

## Chapter 8: PyTorch and Triton Optimizations
* Profiling and Bottleneck Identification
* Compiler-Level Optimizations with torch.compile
* Mixed-Precision and Tensor-Core Techniques
* Attention Variants and High-Level PyTorch Attention APIs
* Fusion and Custom Kernels
* Advanced GPU Kernel Techniques
* Memory Management and Data Pipelines
* Distributed Training and MLPerf
* PyTorch XLA Fundamentals and Workloads
* Key Takeaways
* Conclusion

## Chapter 9: Distributed Training at Ultra-Scale
* Fundamental Challenges and the Need for Distributed Systems
* Core Parallelism Strategies
* Advanced Hardware and Infrastructure
* Overlapping Communication with Computation
* Memory Optimization and Mixed Precision Techniques
* Advanced Scheduling and Kernel‑Level Optimizations
* In‑Depth Communication and Network Optimizations
* Convergence, Stability, and Distributed Optimization Nuances
* Energy Efficiency, Economic Considerations, and Resource Management
* Enhanced Observability, Instrumentation, and Self‑Tuning
* Emerging Trends and Future Directions
* Advanced Operational Considerations
* Key Takeaways
* Conclusion

## Chapter 10: High-Performance Inference Optimizations
* Core System Architectures
* Distributed Inference Strategies for Large Models
* Low‑Level GPU Kernel and Execution Optimizations
* Memory Management and KV Cache Strategies
* Advanced Decoding Techniques
* Batching and Scheduling Strategies
* Precision, Quantization, and Sparsity Techniques
* Profiling and Roofline Analysis
* Application‑Level Best Practices
* Key Takeaways
* Conclusion

## Chapter 11: AI System Performance Optimization Case Studies
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
* Key Takeaways
* Conclusion

## Chapter 12: Future Trends in Ultra-Scale AI Systems Performance Engineering
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
* Key Takeaways
* Conclusion

## Chapter 13: 250+ Tips for AI Systems Performance Engineers
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
* Conclusion

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
