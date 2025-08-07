# Makefile for streams overlap
CUDA_VERSION = 13.1NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: stream_overlap
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
stream_overlap: stream_overlap.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f stream_overlap