TARGET = async_alloc
CUDA_VERSION = 13.1NVCC   = nvcc
ARCH   = -arch=sm_100a
OPT    = -O3

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): async_alloc.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)
