NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17
TARGET = atomic_work_queue

all: $(TARGET)

$(TARGET): atomic_work_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
