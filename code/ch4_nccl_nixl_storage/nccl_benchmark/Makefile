CUDA_PATH := /usr/local/cuda-13.0
NVCC      := $(CUDA_PATH)/bin/nvcc
CXXFLAGS  := -std=c++17 -O2 -arch=sm_90
LDFLAGS   := 

all: p2p_bandwidth_bench

p2p_bandwidth_bench: p2p_bandwidth_bench.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f p2p_bandwidth_bench
