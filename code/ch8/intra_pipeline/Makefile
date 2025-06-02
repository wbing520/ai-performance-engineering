TARGET = double_buffered_pipeline
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): double_buffered_pipeline.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)
