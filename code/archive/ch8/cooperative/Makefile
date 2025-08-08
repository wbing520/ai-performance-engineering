TARGETS = compute_naive_loop compute_persistent
CUDA_VERSION = 13.1NVCC     = nvcc
ARCH     = -arch=sm_100a
OPT      = -O3

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
compute_naive_loop: compute_naive_loop.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

compute_persistent: compute_persistent.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGETS)
