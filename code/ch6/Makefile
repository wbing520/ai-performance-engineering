# Makefile for Chapter 6 - GPU Architecture and CUDA Programming

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80
PYTHON = python3

# CUDA examples
CUDA_TARGETS = my_first_kernel simple_kernel 2d_kernel add_sequential add_parallel \
               stream_ordered_allocator unified_memory launch_bounds_example occupancy_api

# Python examples
PYTHON_TARGETS = add_sequential.py add_parallel.py

.PHONY: all clean cuda python test profile

all: cuda

cuda: $(CUDA_TARGETS)

python:
	@echo "Python examples ready to run:"
	@echo "  python $(PYTHON_TARGETS)"

# CUDA compilation rules
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

# Profiling targets
profile: add_parallel occupancy_api
	@echo "Profiling with Nsight Systems..."
	nsys profile --stats=true -o add_parallel_profile ./add_parallel
	@echo "Profiling with Nsight Compute..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency -o occupancy_profile ./occupancy_api

profile-comparison: add_sequential add_parallel
	@echo "Profiling sequential vs parallel..."
	nsys profile -o sequential_profile ./add_sequential
	nsys profile -o parallel_profile ./add_parallel

clean:
	rm -f $(CUDA_TARGETS)
	rm -f *.o
	rm -f *.nsys-rep *.ncu-rep
	rm -f *_profile.* *.qdrep *.sqlite

help:
	@echo "Available targets:"
	@echo "  all       - Build all CUDA examples"
	@echo "  cuda      - Build CUDA examples"
	@echo "  python    - List Python examples"
	@echo "  test      - Run basic functionality tests"
	@echo "  compare   - Compare sequential vs parallel performance"
	@echo "  profile   - Profile with Nsight tools"
	@echo "  clean     - Remove built files and profiles"
	@echo "  help      - Show this help message"
