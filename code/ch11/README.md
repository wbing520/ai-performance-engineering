# Chapter 10 Code Examples

This folder contains all of the PyTorch code and scripts from Chapter 10 (Profiling, Compiling, and Tuning in PyTorch at Ultra-Scale), updated for:

- **CUDA 13**  
- **PyTorch 2.7**  
- **Blackwell B200** GPU

Examples extracted from Chapter 10: baseline training, profiling with torch.profiler and NVTX, model compilation (torch.compile), mixed-precision training, FlashAttention integration, memory optimizations (checkpointing/offloading), DDP scaling, and inference optimizations with INT8 and CUDA Graphs fileciteturn5file0

Each example has its own directory. From the repo root you can:

```bash
cd baseline && ./run.sh
cd profiling && ./run.sh
cd compile && ./run.sh
cd amp && ./run.sh
cd flash_attention && ./run.sh
cd memory_opt && ./run.sh
cd ddp && ./run.sh
cd inference && ./run.sh
```

Profiler scripts are under **profiler_scripts/** for Nsight Systems (`nsys`) and Nsight Compute (`ncu`):

```bash
bash profiler_scripts/nsys_profile.sh baseline/train_baseline.py
bash profiler_scripts/ncu_profile.sh baseline/train_baseline.py
```
