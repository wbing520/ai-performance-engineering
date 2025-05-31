TARGET = matmul_tensorcore
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): matmul_tensorcore.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -ltarget_device_arch -o $@ $<

clean:
	rm -f $(TARGET)
