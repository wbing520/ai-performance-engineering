# Makefile for Chapter 16: Profiling, Debugging, and Tuning Inference at Scale
CUDA_VERSION = 12.9
# CUDA compiler and flags
NVCC = nvcc
NVCC_FLAGS = -arch=sm_100 -O3 -std=c++17
LIBS = -lnvtx3

# Target executable
TARGET = nvtx_profiling

# Source files
SOURCES = nvtx_profiling.cu

# Default target
all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Compile the CUDA program
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run the program
run: $(TARGET)
	./$(TARGET)

# Profile with Nsight Systems
profile: $(TARGET)
	nsys profile -t cuda,osrt -o nvtx_profile ./$(TARGET)

# Profile with Nsight Compute
compute: $(TARGET)
	ncu --metrics achieved_occupancy,warp_execution_efficiency -o nvtx_compute_profile ./$(TARGET)

# Profile with PC sampling
pc_sampling: $(TARGET)
	ncu --sampling-period=1000 -o pc_sampling_profile ./$(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build the NVTX profiling example"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Run the program"
	@echo "  profile    - Profile with Nsight Systems"
	@echo "  compute    - Profile with Nsight Compute"
	@echo "  pc_sampling - Profile with PC sampling"
	@echo "  help       - Show this help message"

.PHONY: all clean run profile compute pc_sampling help

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
