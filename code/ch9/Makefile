# Architecture switching support
# Set ARCH=sm_90 for Hopper H100/H200 or ARCH=sm_100 for Blackwell B200/B300
ARCH ?= sm_90

# CUDA configuration
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

# Find all CUDA source files
CUDA_SOURCES = $(wildcard *.cu)
TARGETS = $(CUDA_SOURCES:.cu=)

# Default target
all: $(TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@if [ "$(ARCH)" = "sm_90" ]; then \
		echo "✓ Targeting Hopper H100/H200 (Compute Capability 9.0)"; \
	elif [ "$(ARCH)" = "sm_100" ]; then \
		echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"; \
	fi
	@echo "Built targets: $(TARGETS)"

# Build each target
$(TARGETS): %: %.cu
	@echo "Building $@..."
	@if [ "$@" = "cutlass_gemm_example" ]; then \
		$(NVCC) $(NVCC_FLAGS) -lcublas -o $@ $<; \
	else \
		$(NVCC) $(NVCC_FLAGS) -o $@ $<; \
	fi

# Architecture-specific targets
hopper: ARCH=sm_90
hopper: $(TARGETS)
	@echo "Built for Hopper H100/H200"

blackwell: ARCH=sm_100
blackwell: $(TARGETS)
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH) ./$(firstword $(TARGETS))

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	perf record -g -p $$(pgrep $(firstword $(TARGETS))) -o perf.data_$(ARCH) ./$(firstword $(TARGETS))
	perf report -i perf.data_$(ARCH)

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH) ./$(firstword $(TARGETS))
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH) ./$(firstword $(TARGETS))
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH) ./$(firstword $(TARGETS))
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH) ./$(firstword $(TARGETS))

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all hopper blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(TARGETS)
