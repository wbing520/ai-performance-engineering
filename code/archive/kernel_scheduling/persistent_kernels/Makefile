NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = compute_naive_loop compute_persistent

all: $(TARGETS)

compute_naive_loop: compute_naive_loop.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

compute_persistent: compute_persistent.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
