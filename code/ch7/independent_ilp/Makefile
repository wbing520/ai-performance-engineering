TARGET = independent_ops
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): independent_ops.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGET)
