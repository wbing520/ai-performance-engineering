# Makefile for Chapter 7: GPU Memory Access Pattern Optimization
CUDA_VERSION = 12.9
# CUDA Compiler Settings
NVCC = nvcc
CUDA_ARCH = -arch=sm_100  # Ampere/Ada Lovelace (adjust for your GPU)
CUDA_FLAGS = -O3 -std=c++17 -I/usr/local/cuda/include
CUDA_LIBS = -lcuda -lcudart

# C++ Compiler for host code
CXX = g++
CXX_FLAGS = -O3 -std=c++17

# Python interpreter
PYTHON = python3

# Source files
CUDA_SOURCES = $(wildcard *.cu)
CUDA_TARGETS = $(CUDA_SOURCES:.cu=)

.PHONY: all clean cuda python profile help

all: cuda
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Build all CUDA examples
cuda: $(CUDA_TARGETS)

# Generic rule for CUDA compilation
%: %.cu
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) $< -o $@ $(CUDA_LIBS)

# Special rule for TMA async prefetch (requires newer CUDA features)
async_prefetch_tma: async_prefetch_tma.cu
	$(NVCC) -arch=sm_100 $(CUDA_FLAGS) $< -o $@ $(CUDA_LIBS)

# Run Python examples
python:
	@echo "Running PyTorch memory access examples..."
	$(PYTHON) memory_access_pytorch.py
	$(PYTHON) vectorized_pytorch.py
	$(PYTHON) matmul_pytorch.py
	$(PYTHON) lookup_pytorch.py

# Profiling targets
profile-memory: coalesced_copy uncoalesced_copy
	@echo "Profiling memory coalescing examples..."
	ncu --metrics smsp__gld_efficiency.avg.pct_of_peak_sustained,dram__throughput.avg.pct_of_peak_sustained ./uncoalesced_copy
	ncu --metrics smsp__gld_efficiency.avg.pct_of_peak_sustained,dram__throughput.avg.pct_of_peak_sustained ./coalesced_copy

profile-vectorized: scalar_copy vectorized_copy
	@echo "Profiling vectorized memory access..."
	ncu --metrics smsp__gld_efficiency.avg.pct_of_peak_sustained,dram__sectors.read.avg.per_request ./scalar_copy
	ncu --metrics smsp__gld_efficiency.avg.pct_of_peak_sustained,dram__sectors.read.avg.per_request ./vectorized_copy

profile-matmul: naive_matmul tiled_matmul
	@echo "Profiling matrix multiplication..."
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained,sm__sass_average_ffma_ops.pct_of_peak_sustained ./naive_matmul
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained,sm__sass_average_ffma_ops.pct_of_peak_sustained ./tiled_matmul

profile-transpose: transpose_naive transpose_padded
	@echo "Profiling transpose bank conflicts..."
	ncu --metrics smsp__shared_load_bank_conflict.sum,smsp__shared_store_bank_conflict.sum ./transpose_naive
	ncu --metrics smsp__shared_load_bank_conflict.sum,smsp__shared_store_bank_conflict.sum ./transpose_padded

profile-lookup: naive_lookup optimized_lookup
	@echo "Profiling read-only cache usage..."
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained,l2__t_read_throughput.avg.pct_of_peak_sustained ./naive_lookup
	ncu --metrics dram__throughput.avg.pct_of_peak_sustained,l2__t_read_throughput.avg.pct_of_peak_sustained ./optimized_lookup

profile-python:
	@echo "Profiling PyTorch examples..."
	nsys profile --trace=cuda,nvtx -o memory_access_pytorch $(PYTHON) memory_access_pytorch.py
	nsys profile --trace=cuda,nvtx -o vectorized_pytorch $(PYTHON) vectorized_pytorch.py
	nsys profile --trace=cuda,nvtx -o matmul_pytorch $(PYTHON) matmul_pytorch.py

# Install Python dependencies
install-deps:
	pip install -r requirements.txt

clean:
	rm -f $(CUDA_TARGETS) *.o *.nsys-rep *.ncu-rep

help:
	@echo "Available targets:"
	@echo "  all              - Build all CUDA examples"
	@echo "  cuda             - Build CUDA examples"
	@echo "  python           - Run Python examples"
	@echo "  profile-memory   - Profile memory coalescing"
	@echo "  profile-vectorized - Profile vectorized access"
	@echo "  profile-matmul   - Profile matrix multiplication"
	@echo "  profile-transpose - Profile transpose bank conflicts"
	@echo "  profile-lookup   - Profile read-only cache"
	@echo "  profile-python   - Profile Python examples"
	@echo "  install-deps     - Install Python dependencies"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"

# HTA (Holistic Tracing Analysis) profiling
profile-hta: $(TARGET)
	@echo "HTA profiling for multi-GPU analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile ./$(TARGET)

# Perf profiling for system-level analysis
profile-perf: $(TARGET)
	@echo "Perf profiling for system-level analysis..."
	perf record -g -p $$(pgrep $(TARGET)) -o perf.data ./$(TARGET)
	perf report -i perf.data

# Enhanced profiling with all tools
profile-all: $(TARGET)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline ./$(TARGET)
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel ./$(TARGET)
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory ./$(TARGET)
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta ./$(TARGET)
