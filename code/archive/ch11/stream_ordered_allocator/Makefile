NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a
TARGET = stream_ordered_allocator

all: $(TARGET)

$(TARGET): stream_ordered_allocator.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
