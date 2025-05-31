TARGET = unified_mem
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): unified_mem.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)
