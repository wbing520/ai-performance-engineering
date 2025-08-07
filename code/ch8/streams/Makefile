TARGET = streams_overlap
CUDA_VERSION = 13.1NVCC   = nvcc
ARCH   = -arch=sm_100a
OPT    = -O3

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): streams_overlap.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)
