TARGETS = dp_host_launched dp_device_launched
NVCC    = nvcc
ARCH    = -arch=sm_90
OPT     = -O3

all: $(TARGETS)

dp_host_launched: dp_host_launched.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

dp_device_launched: dp_device_launched.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGETS)
