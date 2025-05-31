# Makefile for warp specialization
NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: add_naive add_specialized

add_naive: add_naive.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

add_specialized: add_specialized.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f add_naive add_specialized