TARGETS = uneven_static uneven_dynamic
NVCC    = nvcc
ARCH    = -arch=sm_90
OPT     = -O3

all: $(TARGETS)

uneven_static: uneven_static.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

uneven_dynamic: uneven_dynamic.cu
	$(NVCC) $(OPT) $(ARCH) $< -o $@

clean:
	rm -f $(TARGETS)
