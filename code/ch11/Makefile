# Architecture switching support
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

# Targets
TARGETS = basic_streams stream_ordered_allocator multi_stream_pipeline

# Default target
all: $(TARGETS)
	@echo "Building all examples with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"

# Build individual targets
basic_streams: basic_streams.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

stream_ordered_allocator: stream_ordered_allocator.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

multi_stream_pipeline: multi_stream_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: $(TARGETS)
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$$target_$(ARCH) ./$$target; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$$target_$(ARCH) ./$$target; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$$target_$(ARCH) ./$$target; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for target in $(TARGETS); do \
		echo "Profiling $$target..."; \
		perf record -g -o perf.data_$$target_$(ARCH) ./$$target; \
		perf report -i perf.data_$$target_$(ARCH); \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for target in $(TARGETS); do \
		echo "Comprehensive profiling of $$target..."; \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$$target_$(ARCH) ./$$target; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$$target_$(ARCH) ./$$target; \
		echo "3. Memory profiling..."; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$$target_$(ARCH) ./$$target; \
		echo "4. HTA analysis..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$$target_$(ARCH) ./$$target; \
	done

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all $(TARGETS) blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean
