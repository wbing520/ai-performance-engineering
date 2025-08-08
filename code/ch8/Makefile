# Makefile for Chapter 8: Occupancy Tuning, Warp Efficiency, and ILP
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

# Run Python examples
python:
	@echo "Running PyTorch occupancy examples..."
	$(PYTHON) occupancy_pytorch.py
	$(PYTHON) warp_divergence_pytorch.py
	$(PYTHON) ilp_pytorch.py

# Profiling targets for occupancy analysis
profile-occupancy: occupancy_tuning occupancy_api_example
	@echo "Profiling occupancy optimization..."
	ncu --metrics smsp__warps_active.avg.pct_of_peak_sustained,smsp__occupancy.avg.pct_of_peak_sustained ./occupancy_tuning
	ncu --metrics smsp__warps_active.avg.pct_of_peak_sustained,smsp__occupancy.avg.pct_of_peak_sustained ./occupancy_api_example

# Profiling targets for warp divergence
profile-divergence: threshold_naive threshold_predicated
	@echo "Profiling warp divergence..."
	ncu --metrics smsp__warp_execution_efficiency.avg.pct,smsp__thread_inst_executed_pred_off.avg.pct ./threshold_naive
	ncu --metrics smsp__warp_execution_efficiency.avg.pct,smsp__thread_inst_executed_pred_off.avg.pct ./threshold_predicated

# Profiling targets for instruction-level parallelism
profile-ilp: independent_ops loop_unrolling
	@echo "Profiling instruction-level parallelism..."
	ncu --metrics smsp__issue_efficiency.avg,smsp__inst_executed.avg.per_cycle,smsp__average_warp_latency_issue_stalled_exec_dependency ./independent_ops
	ncu --metrics smsp__issue_efficiency.avg,smsp__inst_executed.avg.per_cycle,smsp__average_warp_latency_issue_stalled_exec_dependency ./loop_unrolling

# Comprehensive occupancy analysis
profile-occupancy-detailed: occupancy_tuning
	@echo "Detailed occupancy analysis..."
	ncu --metrics smsp__warps_active.avg.pct_of_peak_sustained,smsp__occupancy.avg.pct_of_peak_sustained,launch__occupancy_limit_warps,launch__occupancy_limit_blocks ./occupancy_tuning

# Comprehensive warp efficiency analysis
profile-warp-efficiency: threshold_naive threshold_predicated
	@echo "Detailed warp efficiency analysis..."
	ncu --metrics smsp__warp_execution_efficiency.avg.pct,smsp__average_warp_latency_issue_stalled_branch_resolving,smsp__inst_executed.sum,smsp__thread_inst_executed_pred_off.avg.pct ./threshold_naive
	ncu --metrics smsp__warp_execution_efficiency.avg.pct,smsp__average_warp_latency_issue_stalled_branch_resolving,smsp__inst_executed.sum,smsp__thread_inst_executed_pred_off.avg.pct ./threshold_predicated

# Timeline profiling with Nsight Systems
profile-timeline: occupancy_tuning
	@echo "Timeline profiling..."
	nsys profile --trace=cuda,nvtx -o occupancy_timeline ./occupancy_tuning

# PyTorch profiling
profile-python:
	@echo "Profiling PyTorch examples..."
	nsys profile --trace=cuda,nvtx -o occupancy_pytorch $(PYTHON) occupancy_pytorch.py
	nsys profile --trace=cuda,nvtx -o warp_divergence_pytorch $(PYTHON) warp_divergence_pytorch.py
	nsys profile --trace=cuda,nvtx -o ilp_pytorch $(PYTHON) ilp_pytorch.py

# Compare different optimization levels
compare-optimizations: threshold_naive threshold_predicated independent_ops
	@echo "Comparing optimization techniques..."
	@echo "=== Warp Divergence Comparison ==="
	@echo "Naive threshold (with divergence):"
	@timeout 30 ncu --metrics smsp__warp_execution_efficiency.avg.pct ./threshold_naive || true
	@echo "Predicated threshold (without divergence):"
	@timeout 30 ncu --metrics smsp__warp_execution_efficiency.avg.pct ./threshold_predicated || true
	@echo "=== ILP Comparison ==="
	@echo "Independent operations (better ILP):"
	@timeout 30 ncu --metrics smsp__issue_efficiency.avg ./independent_ops || true

# Install Python dependencies
install-deps:
	pip install -r requirements.txt

# Check GPU properties
gpu-info:
	$(NVCC) --version
	nvidia-smi
	@echo "Compiling and running GPU info utility..."
	@echo '#include <cuda_runtime.h>' > gpu_info.cu
	@echo '#include <iostream>' >> gpu_info.cu
	@echo 'int main() {' >> gpu_info.cu
	@echo '  cudaDeviceProp prop;' >> gpu_info.cu
	@echo '  cudaGetDeviceProperties(&prop, 0);' >> gpu_info.cu
	@echo '  std::cout << "GPU: " << prop.name << std::endl;' >> gpu_info.cu
	@echo '  std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;' >> gpu_info.cu
	@echo '  std::cout << "Max warps per SM: " << prop.maxThreadsPerMultiProcessor/32 << std::endl;' >> gpu_info.cu
	@echo '  std::cout << "Registers per SM: " << prop.regsPerMultiprocessor << std::endl;' >> gpu_info.cu
	@echo '  std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;' >> gpu_info.cu
	@echo '  return 0;' >> gpu_info.cu
	@echo '}' >> gpu_info.cu
	$(NVCC) $(CUDA_ARCH) gpu_info.cu -o gpu_info
	./gpu_info
	rm -f gpu_info gpu_info.cu

clean:
	rm -f $(CUDA_TARGETS) *.o *.nsys-rep *.ncu-rep gpu_info

help:
	@echo "Available targets:"
	@echo "  all                      - Build all CUDA examples"
	@echo "  cuda                     - Build CUDA examples"
	@echo "  python                   - Run Python examples"
	@echo "  profile-occupancy        - Profile occupancy optimization"
	@echo "  profile-divergence       - Profile warp divergence"
	@echo "  profile-ilp              - Profile instruction-level parallelism"
	@echo "  profile-occupancy-detailed - Detailed occupancy analysis"
	@echo "  profile-warp-efficiency  - Detailed warp efficiency analysis"
	@echo "  profile-timeline         - Timeline profiling with Nsight Systems"
	@echo "  profile-python           - Profile Python examples"
	@echo "  compare-optimizations    - Compare different optimization techniques"
	@echo "  install-deps             - Install Python dependencies"
	@echo "  gpu-info                 - Show GPU properties"
	@echo "  clean                    - Remove build artifacts"
	@echo "  help                     - Show this help message"

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
