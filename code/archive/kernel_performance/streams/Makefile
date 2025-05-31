# Makefile for streams overlap
NVCC := /usr/local/cuda-13.0/bin/nvcc
CXXFLAGS := -O3 -std=c++17

all: stream_overlap

stream_overlap: stream_overlap.cu
	$(NVCC) $(CXXFLAGS) -o $@ $<

clean:
	rm -f stream_overlap