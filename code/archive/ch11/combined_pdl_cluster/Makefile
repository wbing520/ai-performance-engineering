NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17
TARGET = combined_pdl_cluster

all: $(TARGET)

$(TARGET): combined_pdl_cluster.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
