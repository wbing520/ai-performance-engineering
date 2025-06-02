TARGET = warp_specialized_pipeline_kernel
NVCC    = nvcc
ARCH    = -arch=sm_90
OPT     = -O3

all: $(TARGET)

$(TARGET): warp_specialized_pipeline.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)
