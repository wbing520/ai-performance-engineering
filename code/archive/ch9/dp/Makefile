TARGETS = dp_host_launched dp_device_launched
CUDA_VERSION = 13.1NVCC    = nvcc
ARCH    = -arch=sm_100a
OPT     = -O3

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
dp_host_launched: dp_host_launched.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

dp_device_launched: dp_device_launched.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGETS)
