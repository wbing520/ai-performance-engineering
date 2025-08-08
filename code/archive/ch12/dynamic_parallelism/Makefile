NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17

HOST_TARGET = dp_host_launched
DEVICE_TARGET = dp_device_launched

all: $(HOST_TARGET) $(DEVICE_TARGET)

$(HOST_TARGET): dp_host_launched.cu
	$(NVCC) $(NVCC_FLAGS) -o $(HOST_TARGET) $<

$(DEVICE_TARGET): dp_device_launched.cu
	$(NVCC) $(NVCC_FLAGS) -o $(DEVICE_TARGET) $<

clean:
	rm -f $(HOST_TARGET) $(DEVICE_TARGET)

run_host: $(HOST_TARGET)
	./$(HOST_TARGET)

run_device: $(DEVICE_TARGET)
	./$(DEVICE_TARGET)

.PHONY: all clean run_host run_device
