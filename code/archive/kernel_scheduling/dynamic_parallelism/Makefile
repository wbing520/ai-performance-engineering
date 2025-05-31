NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = dp_host_launched dp_device_launched

all: $(TARGETS)

dp_host_launched: dp_host_launched.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

dp_device_launched: dp_device_launched.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
