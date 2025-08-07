NVCC = nvcc
CUDA_VERSION = 13.1CXXFLAGS = -std=c++17
TARGETS = sequential_small_kernels concurrent_stream_kernels

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
sequential_small_kernels: sequential_small_kernels.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

concurrent_stream_kernels: concurrent_stream_kernels.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
