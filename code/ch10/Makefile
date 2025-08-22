# Architecture switching support
# Set ARCH=sm_90 for Hopper H100/H200 or ARCH=sm_100 for Blackwell B200/B300
ARCH ?= sm_90

# CUDA configuration
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

# Source files
SOURCES = double_buffered_pipeline.cu warp_specialized_pipeline.cu cooperative_persistent_kernel.cu
TARGETS = $(SOURCES:.cu=)

# Blackwell-only examples are built only when ARCH=sm_100
ifeq ($(ARCH),sm_100)
SOURCES += tma_2d_pipeline_blackwell.cu cluster_group_blackwell.cu
TARGETS += tma_2d_pipeline_blackwell cluster_group_blackwell
endif

# Default target
all: $(TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@if [ "$(ARCH)" = "sm_90" ]; then \
		echo "✓ Targeting Hopper H100/H200 (Compute Capability 9.0)"; \
	elif [ "$(ARCH)" = "sm_100" ]; then \
		echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"; \
	fi

# Build individual targets
double_buffered_pipeline: double_buffered_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

warp_specialized_pipeline: warp_specialized_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cooperative_persistent_kernel: cooperative_persistent_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Blackwell-only builds
tma_2d_pipeline_blackwell: tma_2d_pipeline_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cluster_group_blackwell: cluster_group_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

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
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH)_$$target ./$$target; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH)_$$target ./$$target; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH)_$$target ./$$target; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		perf record -g -o perf.data_$(ARCH)_$$target ./$$target; \
		perf report -i perf.data_$(ARCH)_$$target; \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for target in $(TARGETS); do \
		echo "Comprehensive profiling of $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH)_$$target ./$$target; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH)_$$target ./$$target; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH)_$$target ./$$target; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH)_$$target ./$$target; \
	done

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all hopper blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(TARGETS)
