TARGET = matmul_tensorcore
NVCC   = nvcc
ARCH   = -arch=sm_100
OPT    = -O3
CUDA_VERSION = 12.9

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): matmul_tensorcore.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -ltarget_device_arch -o $@ $<

clean:
	rm -f $(TARGET)
