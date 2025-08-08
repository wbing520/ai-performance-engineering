NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17
TARGET = cuda_graphs

all: $(TARGET)

$(TARGET): cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

run_pytorch:
	python cuda_graphs_pytorch.py

.PHONY: all clean run run_pytorch
