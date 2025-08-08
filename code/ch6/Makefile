# Makefile for Chapter 6 - GPU Architecture and CUDA Programming
# Updated for CUDA 12.9 and Blackwell B200/B300 (SM100)

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_100 -DCUDA_VERSION=12.9 -lnvtx3
PYTHON = python3

# CUDA examples
CUDA_TARGETS = my_first_kernel simple_kernel 2d_kernel add_sequential add_parallel \
               stream_ordered_allocator unified_memory launch_bounds_example occupancy_api

# Python examples
PYTHON_TARGETS = add_sequential.py add_parallel.py

.PHONY: all clean cuda python test profile

all: cuda
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
cuda: $(CUDA_TARGETS)

python:
	@echo "Python examples ready to run:"
	@echo "  python $(PYTHON_TARGETS)"

# CUDA compilation rules with Blackwell B200/B300 optimizations
my_first_kernel: my_first_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

simple_kernel: simple_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

2d_kernel: 2d_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

add_sequential: add_sequential.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

add_parallel: add_parallel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

stream_ordered_allocator: stream_ordered_allocator.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

unified_memory: unified_memory.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

launch_bounds_example: launch_bounds_example.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

occupancy_api: occupancy_api.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Test targets
test: cuda
	@echo "Running basic functionality tests..."
	./my_first_kernel
	./simple_kernel
	./2d_kernel
	./add_parallel
	./occupancy_api
	$(PYTHON) add_parallel.py

# Performance comparison
compare: add_sequential add_parallel
	@echo "Comparing sequential vs parallel performance..."
	./add_sequential
	./add_parallel
	$(PYTHON) add_sequential.py
	$(PYTHON) add_parallel.py

# Enhanced profiling targets with latest tools
profile: add_parallel occupancy_api
	@echo "Profiling with Nsight Systems (latest)..."
	nsys profile --force-overwrite=true --stats=true -t cuda,nvtx,osrt -o add_parallel_profile ./add_parallel
	@echo "Profiling with Nsight Compute (latest)..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed -o occupancy_profile ./occupancy_api

profile-comparison: add_sequential add_parallel
	@echo "Profiling sequential vs parallel with enhanced tools..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o sequential_profile ./add_sequential
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o parallel_profile ./add_parallel

# Blackwell B200/B300 specific profiling
profile-blackwell: add_parallel
	@echo "Profiling with Blackwell B200/B300 optimizations..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o blackwell_profile ./add_parallel
	ncu --metrics dram_read_throughput,dram_write_throughput,sm__throughput.avg.pct_of_peak_sustained_elapsed -o blackwell_memory_profile ./add_parallel

# Memory profiling
profile-memory: add_parallel
	@echo "Memory profiling with latest tools..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o memory_profile ./add_parallel

# HTA (Holistic Tracing Analysis) profiling
profile-hta: add_parallel
	@echo "HTA profiling for multi-GPU analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile ./add_parallel

# Perf profiling for system-level analysis
profile-perf: add_parallel
	@echo "Perf profiling for system-level analysis..."
	perf record -g -p $$(pgrep add_parallel) -o perf.data ./add_parallel
	perf report -i perf.data

# Enhanced profiling with all tools
profile-all: add_parallel
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline ./add_parallel
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel ./add_parallel
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory ./add_parallel
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta ./add_parallel

clean:
	rm -f $(CUDA_TARGETS)
	rm -f *.o
	rm -f *.nsys-rep *.ncu-rep
	rm -f *_profile.* *.qdrep *.sqlite
	rm -f perf.data

help:
	@echo "Available targets:"
	@echo "  all              - Build all CUDA examples"
	@echo "  cuda             - Build CUDA examples"
	@echo "  python           - List Python examples"
	@echo "  test             - Run basic functionality tests"
	@echo "  compare          - Compare sequential vs parallel performance"
	@echo "  profile          - Profile with Nsight tools"
	@echo "  profile-comparison - Profile sequential vs parallel"
	@echo "  profile-blackwell - Profile with Blackwell B200/B300 optimizations"
	@echo "  profile-memory   - Memory profiling"
	@echo "  profile-hta      - HTA profiling for multi-GPU"
	@echo "  profile-perf     - Perf profiling for system-level"
	@echo "  profile-all      - Comprehensive profiling with all tools"
	@echo "  clean            - Remove built files and profiles"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Architecture: SM100 (Blackwell B200/B300)"
	@echo "CUDA Version: 12.9"
	@echo "Optimization: -O3 with latest flags"
	@echo "Profiling: Nsight Systems, Nsight Compute, HTA, Perf"
