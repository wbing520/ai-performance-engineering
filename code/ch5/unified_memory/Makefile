TARGET = unified_mem
CUDA_VERSION = 13.1NVCC   = nvcc
ARCH   = -arch=sm_100a
OPT    = -O3

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): unified_mem.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)
