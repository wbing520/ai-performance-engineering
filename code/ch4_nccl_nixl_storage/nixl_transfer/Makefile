CUDA_PATH := /usr/local/cuda-13.0
NVCC      := $(CUDA_PATH)/bin/nvcc
CXXFLAGS  := -std=c++17 -O2 -arch=sm_90
LDFLAGS   := -lnixl -lcudart

all: nixl_transfer

nixl_transfer: nixl_transfer.cpp
	$(NVCC) $(CXXFLAGS) nixl_transfer.cpp -o nixl_transfer $(LDFLAGS)

clean:
	rm -f nixl_transfer
