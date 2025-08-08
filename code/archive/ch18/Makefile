# Makefile for Chapter 18: Advanced Prefill-Decode and KV Cache Tuning
CUDA_VERSION = 12.9
# CUDA compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_100 -O3 -std=c++17 -lcudart
LIBS = -lcudart

# Target executables
TARGETS = flashmla_kernel thundermla_kernel

# Source files
FLASHMLA_SOURCES = flashmla_kernel.cu
THUNDERMLA_SOURCES = flashmla_kernel.cu  # Same file contains both kernels

# Default target
all: $(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Compile FlashMLA kernel
flashmla_kernel: $(FLASHMLA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o flashmla_kernel $(FLASHMLA_SOURCES) $(LIBS)

# Compile ThunderMLA kernel
thundermla_kernel: $(THUNDERMLA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o thundermla_kernel $(THUNDERMLA_SOURCES) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o

# Run FlashMLA kernel
run_flashmla: flashmla_kernel
	./flashmla_kernel

# Run ThunderMLA kernel
run_thundermla: thundermla_kernel
	./thundermla_kernel

# Profile with Nsight Systems
profile_flashmla: flashmla_kernel
	nsys profile -t cuda,osrt -o flashmla_profile ./flashmla_kernel

profile_thundermla: thundermla_kernel
	nsys profile -t cuda,osrt -o thundermla_profile ./thundermla_kernel

# Profile with Nsight Compute
compute_flashmla: flashmla_kernel
	ncu --metrics achieved_occupancy,warp_execution_efficiency -o flashmla_compute_profile ./flashmla_kernel

compute_thundermla: thundermla_kernel
	ncu --metrics achieved_occupancy,warp_execution_efficiency -o thundermla_compute_profile ./thundermla_kernel

# Profile with PC sampling
pc_sampling_flashmla: flashmla_kernel
	ncu --sampling-period=5 -o flashmla_pc_profile ./flashmla_kernel

pc_sampling_thundermla: thundermla_kernel
	ncu --sampling-period=5 -o thundermla_pc_profile ./thundermla_kernel

# Run Python examples
run_flexdecoding:
	python flexdecoding_example.py

# Run all examples
run_all: run_flashmla run_thundermla run_flexdecoding

# Profile all examples
profile_all: profile_flashmla profile_thundermla

compute_all: compute_flashmla compute_thundermla

# Memory profiling
memory_profile_flashmla: flashmla_kernel
	ncu --metrics dram_read_throughput,dram_write_throughput -o flashmla_memory_profile ./flashmla_kernel

memory_profile_thundermla: thundermla_kernel
	ncu --metrics dram_read_throughput,dram_write_throughput -o thundermla_memory_profile ./thundermla_kernel

# Kernel analysis
kernel_analysis_flashmla: flashmla_kernel
	ncu --kernel-regex "flashmla_decode_kernel" -o flashmla_kernel_analysis ./flashmla_kernel

kernel_analysis_thundermla: thundermla_kernel
	ncu --kernel-regex "thundermla_mega_kernel" -o thundermla_kernel_analysis ./thundermla_kernel

# Help target
help:
	@echo "Available targets:"
	@echo "  all                    - Build all executables"
	@echo "  flashmla_kernel        - Build FlashMLA kernel"
	@echo "  thundermla_kernel      - Build ThunderMLA kernel"
	@echo "  clean                  - Clean build artifacts"
	@echo "  run_flashmla           - Run FlashMLA kernel"
	@echo "  run_thundermla         - Run ThunderMLA kernel"
	@echo "  run_flexdecoding       - Run FlexDecoding Python example"
	@echo "  run_all                - Run all examples"
	@echo "  profile_flashmla       - Profile FlashMLA with Nsight Systems"
	@echo "  profile_thundermla     - Profile ThunderMLA with Nsight Systems"
	@echo "  compute_flashmla       - Profile FlashMLA with Nsight Compute"
	@echo "  compute_thundermla     - Profile ThunderMLA with Nsight Compute"
	@echo "  pc_sampling_flashmla   - PC sampling for FlashMLA"
	@echo "  pc_sampling_thundermla - PC sampling for ThunderMLA"
	@echo "  memory_profile_flashmla - Memory profiling for FlashMLA"
	@echo "  memory_profile_thundermla - Memory profiling for ThunderMLA"
	@echo "  kernel_analysis_flashmla - Kernel analysis for FlashMLA"
	@echo "  kernel_analysis_thundermla - Kernel analysis for ThunderMLA"
	@echo "  profile_all            - Profile all CUDA examples"
	@echo "  compute_all            - Compute profile all CUDA examples"
	@echo "  help                   - Show this help message"

.PHONY: all clean run_flashmla run_thundermla run_flexdecoding run_all
.PHONY: profile_flashmla profile_thundermla compute_flashmla compute_thundermla
.PHONY: pc_sampling_flashmla pc_sampling_thundermla memory_profile_flashmla memory_profile_thundermla
.PHONY: kernel_analysis_flashmla kernel_analysis_thundermla profile_all compute_all help

# HTA (Holistic Tracing Analysis) profiling
profile-hta: $(TARGET)
	@echo "HTA profiling for multi-GPU analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile ./$(TARGET)

# Perf profiling for system-level analysis
profile-perf: $(TARGET)
	@echo "Perf profiling for system-level analysis..."
	perf record -g -p $$(pgrep $(TARGET)) -o perf.data ./$(TARGET)
	perf report -i perf.data

# Enhanced profiling with all tools
profile-all: $(TARGET)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline ./$(TARGET)
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel ./$(TARGET)
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory ./$(TARGET)
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta ./$(TARGET)
