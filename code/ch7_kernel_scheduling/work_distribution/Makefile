NVCC = nvcc
CXXFLAGS = -std=c++17
TARGETS = uneven_static uneven_dynamic

all: $(TARGETS)

uneven_static: uneven_static.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

uneven_dynamic: uneven_dynamic.cu
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
