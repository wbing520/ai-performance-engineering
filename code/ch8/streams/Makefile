TARGET = streams_overlap
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: $(TARGET)

$(TARGET): streams_overlap.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGET)
