NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17
TARGET = warp_specialized_pipeline

all: $(TARGET)

$(TARGET): warp_specialized_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
