TARGET = cutlass_gemm
NVCC   = nvcc
ARCH   = -arch=sm_100
OPT    = -O3
CUDA_VERSION = 12.9

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): CUTLASS_GEMM.cpp
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -I/path/to/cutlass/include -DCUDA_VERSION=$(CUDA_VERSION) $< -o $@

clean:
	rm -f $(TARGET)
