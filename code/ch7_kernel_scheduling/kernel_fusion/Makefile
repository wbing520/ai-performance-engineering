NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = add_mul_naive fused_add_mul

all: $(TARGETS)

add_mul_naive: add_mul_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

fused_add_mul: fused_add_mul.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
