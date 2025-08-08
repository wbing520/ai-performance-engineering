#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1

# Memory profiling for PyTorch 2.8 and CUDA 12.9
echo "Running memory profiler..."

# Set PyTorch memory profiling environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_SHOW_CPP_STACKTRACES=1

# Run with memory profiling
python3 -c "
import torch
import sys
import os

# Enable memory profiling
torch.cuda.set_per_process_memory_fraction(0.8)
torch.cuda.empty_cache()

# Import and run the script
sys.path.append('.')
exec(open('$SCRIPT').read())

# Print memory statistics
print('\\nMemory Statistics:')
print(f'Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
print(f'Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
print(f'Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')
print(f'Max cached: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB')
"

echo "Memory profiling completed."
