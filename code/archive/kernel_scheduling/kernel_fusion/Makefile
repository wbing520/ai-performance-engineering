NVCC = nvcc
CUDA_VERSION = 13.1CXXFLAGS = -std=c++17
TARGETS = add_mul_naive fused_add_mul

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
add_mul_naive: add_mul_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

fused_add_mul: fused_add_mul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
