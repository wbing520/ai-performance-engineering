TARGET = cutlass_gemm
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): CUTLASS_GEMM.cpp
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -I/path/to/cutlass/include $< -o $@

clean:
	rm -f $(TARGET)
