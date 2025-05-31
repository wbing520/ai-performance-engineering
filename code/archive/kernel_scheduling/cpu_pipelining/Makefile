NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = cpu_bottleneck_naive cpu_bottleneck_pipelined

all: $(TARGETS)

cpu_bottleneck_naive: cpu_bottleneck_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

cpu_bottleneck_pipelined: cpu_bottleneck_pipelined.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
