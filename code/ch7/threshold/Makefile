TARGETS = threshold_naive threshold_predicated
NVCC    = nvcc
ARCH    = -arch=sm_90
OPT     = -O3

all: $(TARGETS)

threshold_naive: threshold_naive.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

threshold_predicated: threshold_predicated.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGETS)
