NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a
TARGET = basic_streams

all: $(TARGET)

$(TARGET): basic_streams.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
