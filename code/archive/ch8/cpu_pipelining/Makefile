NVCC = nvcc
CUDA_VERSION = 13.1CXXFLAGS = -std=c++17
TARGETS = cpu_bottleneck_naive cpu_bottleneck_pipelined

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
cpu_bottleneck_naive: cpu_bottleneck_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cpu_bottleneck_pipelined: cpu_bottleneck_pipelined.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
