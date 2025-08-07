NVCC = nvcc
CUDA_VERSION = 13.1CXXFLAGS = -std=c++17
TARGETS = uneven_static uneven_dynamic

all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
uneven_static: uneven_static.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

uneven_dynamic: uneven_dynamic.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
