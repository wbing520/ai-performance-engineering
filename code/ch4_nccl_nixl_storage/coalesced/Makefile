# Makefile for coalesced_copy.cu
# Targets: all, clean

CUDA_PATH   := /usr/local/cuda-13.0
NVCC        := $(CUDA_PATH)/bin/nvcc
CXXFLAGS    := -std=c++17 -O2 -arch=sm_90   # sm_90 for Grace-Blackwell; use sm_80 for H100
LDFLAGS     :=

all: coalesced_copy

coalesced_copy: coalesced_copy.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f coalesced_copy
