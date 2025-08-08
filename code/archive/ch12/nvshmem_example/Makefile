NVCC = nvcc
CUDA_VERSION = 12.9NVCC_FLAGS = -O3 -arch=sm_100a -std=c++17
TARGET = nvshmem_example

# NVSHMEM library flags (adjust paths as needed)
NVSHMEM_INC = -I/usr/local/nvshmem/include
NVSHMEM_LIB = -L/usr/local/nvshmem/lib -lnvshmem

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
$(TARGET): nvshmem_example.cu
	$(NVCC) $(NVCC_FLAGS) $(NVSHMEM_INC) -o $(TARGET) $< $(NVSHMEM_LIB)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	# Run with 2 PEs (GPUs)
	mpirun -np 2 ./$(TARGET)

.PHONY: all clean run

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
