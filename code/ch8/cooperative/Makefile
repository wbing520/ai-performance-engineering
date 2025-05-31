TARGETS=compute_naive_loop compute_persistent
NVCC=nvcc
ARCH=-arch=sm_90
OPT=-O3

all: $(TARGETS)

compute_naive_loop: compute_naive_loop.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

compute_persistent: compute_persistent.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $<

clean:
	rm -f $(TARGETS)
