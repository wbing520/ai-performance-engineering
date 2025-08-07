TARGETS = threshold_naive threshold_predicated
CUDA_VERSION = 13.1NVCC    = nvcc
ARCH    = -arch=sm_100a
OPT     = -O3

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
threshold_naive: threshold_naive.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

threshold_predicated: threshold_predicated.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGETS)
