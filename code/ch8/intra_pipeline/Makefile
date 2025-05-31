TARGET=gemm_tiled_pipeline
NVCC=nvcc
ARCH=-arch=sm_90
OPT=-O3

all: $(TARGET)

$(TARGET): gemm_tiled_pipeline.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)
