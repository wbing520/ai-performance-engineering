# Architecture switching support
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 12.8
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

# Build all CUDA sources in this chapter
SOURCES = atomic_work_queue.cu cuda_graphs.cu dynamic_parallelism.cu
TARGETS = $(SOURCES:.cu=)

# Default target
all: $(TARGETS)
	@echo "Building with $(ARCH) architecture (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"

$(TARGETS): %: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Dynamic parallelism requires relocatable device code and device runtime library
dynamic_parallelism: dynamic_parallelism.cu
	$(NVCC) $(NVCC_FLAGS) -rdc=true -o $@ $< -lcudadevrt

# Architecture-specific targets
blackwell: ARCH=sm_100
blackwell: $(TARGETS)
	@echo "Built for Blackwell B200/B300"

# Enhanced profiling targets
profile-nsys: $(TARGETS)
	@echo "Nsight Systems timeline profiling..."
	@for t in $(TARGETS); do \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas -o nsys_profile_$(ARCH)_$$t ./$$t; \
	done

profile-ncu: $(TARGETS)
	@echo "Nsight Compute kernel profiling..."
	@for t in $(TARGETS); do \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o ncu_profile_$(ARCH)_$$t ./$$t; \
	done

profile-hta: $(TARGETS)
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for t in $(TARGETS); do \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile_$(ARCH)_$$t ./$$t; \
	done

profile-perf: $(TARGETS)
	@echo "Perf system-level profiling..."
	@for t in $(TARGETS); do \
		perf record -g -o perf.data_$(ARCH)_$$t ./$$t; \
		perf report -i perf.data_$(ARCH)_$$t; \
	done

profile-all: $(TARGETS)
	@echo "Comprehensive profiling with all tools..."
	@for t in $(TARGETS); do \
		echo "1. Nsight Systems timeline..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline_$(ARCH)_$$t ./$$t; \
		echo "2. Nsight Compute kernel analysis..."; \
		ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel_$(ARCH)_$$t ./$$t; \
		echo "3. Memory profiling..."; \
		nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory_$(ARCH)_$$t ./$$t; \
		echo "4. HTA analysis..."; \
		nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta_$(ARCH)_$$t ./$$t; \
	done

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* ncu_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all blackwell profile-nsys profile-ncu profile-hta profile-perf profile-all clean $(TARGETS)
