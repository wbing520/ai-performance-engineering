TARGETS = uneven_static uneven_dynamic
NVCC    = nvcc
ARCH    = -arch=sm_100
OPT     = -O3
CUDA_VERSION = 12.9

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"
uneven_static: uneven_static.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) $< -o $@

uneven_dynamic: uneven_dynamic.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) $< -o $@

clean:
	rm -f $(TARGETS)
