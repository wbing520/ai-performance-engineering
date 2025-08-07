# Makefile for ILP examples
CUDA_VERSION = 13.1NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: reduce_naive reduce_unrolled
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
reduce_naive: reduce_naive.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

reduce_unrolled: reduce_unrolled.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f reduce_naive reduce_unrolled