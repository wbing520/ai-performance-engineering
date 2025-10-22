#!/bin/bash
# Profiling script for Chapter 14: PyTorch Compiler, Triton, XLA
# Tests torch.compile, Triton kernels, FP8, and DeepSeek optimizations

set -e

echo "Chapter 14: PyTorch & Triton Profiling"
echo "======================================"
echo ""

# 1. Profile torch.compile performance
echo "1. Profiling torch.compile..."
echo "-----------------------------"
python3 -m torch.profiler torch_compiler_examples.py > torch_compile_profile.txt 2>&1 &
PID=$!

nsys profile -o ch14_torch_compile_timeline \
    --trace=cuda,nvtx,osrt,python \
    --python-sampling=true \
    --force-overwrite=true \
    -p $PID

echo "  Output: torch_compile_profile.txt"
echo "  Timeline: ch14_torch_compile_timeline.nsys-rep"
echo ""

# 2. Profile Triton FP8 kernels
echo "2. Profiling Triton FP8 Kernels..."
echo "----------------------------------"
python3 -c "
import torch
import sys
sys.path.append('.')
from triton_examples import benchmark_fp8_vs_fp16

# Run FP8 benchmark
print('Testing FP8 performance...')
benchmark_fp8_vs_fp16()
" > triton_fp8_results.txt 2>&1

echo "  Output: triton_fp8_results.txt"
echo ""

# 3. Profile DeepSeek L2 cache optimization
echo "3. Profiling DeepSeek Innovation..."
echo "-----------------------------------"
python3 deepseek_innovation_l2_bypass.py > deepseek_results.txt 2>&1

echo "  Output: deepseek_results.txt"
echo ""

# 4. Profile torch.compile with large model
echo "4. Profiling Large Model Compilation..."
echo "---------------------------------------"
python3 torch_compile_large_model.py > large_model_results.txt 2>&1

echo "  Output: large_model_results.txt"
echo ""

# 5. Profile training speedup
echo "5. Profiling Training Performance..."
echo "------------------------------------"
python3 training_large_model_1_5x.py > training_results.txt 2>&1

echo "  Output: training_results.txt"
echo ""

# Summary
echo "Key Metrics to Verify:"
echo "  torch.compile:"
echo "    - Speedup: Target 1.3x+ for large models (>500M params)"
echo "    - Warmup: 100+ iterations required"
echo "    - Compilation time: ~10-30s for first run"
echo ""
echo "  Triton FP8:"
echo "    - FP8 TFLOPS: Target >1200"
echo "    - Speedup vs FP16: Target 1.5-2.0x"
echo "    - Memory reduction: 50%"
echo ""
echo "  DeepSeek L2 cache:"
echo "    - Speedup: Target 1.05-1.15x (5-15% improvement)"
echo "    - Cache hit rate improvement"
echo ""
echo "  Training:"
echo "    - Speedup: Target 1.2-1.5x for large models"
echo "    - End-to-end (forward + backward + optimizer)"
echo ""

echo "Profiling complete!"
echo ""
echo "View results:"
echo "  cat torch_compile_profile.txt"
echo "  cat triton_fp8_results.txt"
echo "  cat deepseek_results.txt"
echo "  nsys-ui ch14_torch_compile_timeline.nsys-rep"

