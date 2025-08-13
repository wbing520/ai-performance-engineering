# Architecture switching support
# Set ARCH=sm_90 for Hopper H100/H200 or ARCH=sm_100 for Blackwell B200/B300
ARCH ?= sm_90

# CUDA configuration
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

# Find all CUDA source files
CUDA_SOURCES = $(wildcard *.cu)
CUDA_TARGETS = $(CUDA_SOURCES:.cu=)

# Default target - build all CUDA examples
all: $(CUDA_TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@if [ "$(ARCH)" = "sm_90" ]; then \
		echo "✓ Targeting Hopper H100/H200 (Compute Capability 9.0)"; \
	elif [ "$(ARCH)" = "sm_100" ]; then \
		echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"; \
	fi
	@echo "✓ Built $(words $(CUDA_TARGETS)) CUDA examples"

# Build individual targets
$(CUDA_TARGETS): %: %.cu
	@echo "Compiling $@..."
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Architecture-specific targets
hopper: ARCH=sm_90
hopper: all
	@echo "Built for Hopper H100/H200"

blackwell: ARCH=sm_100
blackwell: all
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: all
	@echo "Nsight Systems timeline profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o nsys_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-ncu: all
	@echo "Nsight Compute kernel profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o ncu_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o hta_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || echo "Profiling $$exe failed"; \
	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(CUDA_TARGETS); do \
		echo "Profiling $$exe with perf..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || echo "perf failed for $$exe"; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(CUDA_TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency -o comprehensive_kernel_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; \
		echo "3. Perf system-level analysis..."; \
		perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe || true; \
		perf report -i perf.data_$(ARCH)_$$exe || true; \
	done

clean:
	rm -f $(CUDA_TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all hopper blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(CUDA_TARGETS)
