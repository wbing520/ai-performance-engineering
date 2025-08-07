# Makefile for warp specialization
CUDA_VERSION = 13.1NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: add_naive add_specialized
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
add_naive: add_naive.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

add_specialized: add_specialized.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f add_naive add_specialized