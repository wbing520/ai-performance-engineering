# Makefile for TensorCore examples
NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: matmul_naive_fp32 matmul_tensorcore_fp16

matmul_naive_fp32: matmul_naive_fp32.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

matmul_tensorcore_fp16: matmul_tensorcore_fp16.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f matmul_naive_fp32 matmul_tensorcore_fp16