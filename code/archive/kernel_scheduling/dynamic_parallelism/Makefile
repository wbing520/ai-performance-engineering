NVCC = nvcc
CUDA_VERSION = 13.1CXXFLAGS = -std=c++17
TARGETS = dp_host_launched dp_device_launched

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
dp_host_launched: dp_host_launched.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

dp_device_launched: dp_device_launched.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
