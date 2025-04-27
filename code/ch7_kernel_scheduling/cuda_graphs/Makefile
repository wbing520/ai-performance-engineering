NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = multi_op_naive multi_op_graph

all: $(TARGETS)

multi_op_naive: multi_op_naive.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

multi_op_graph: multi_op_graph.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
