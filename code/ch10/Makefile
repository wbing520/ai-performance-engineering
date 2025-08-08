# Makefile for Chapter 10 - Intra-Kernel Pipelining and Warp Specialization
CUDA_VERSION = 12.9
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_100 --expt-relaxed-constexpr
PYTHON = python3

# CUDA targets
CUDA_TARGETS = double_buffered_pipeline warp_specialized_pipeline cooperative_persistent_kernel

.PHONY: all clean test profile benchmark analyze help

all: $(CUDA_TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Compilation rules
double_buffered_pipeline: double_buffered_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

warp_specialized_pipeline: warp_specialized_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cooperative_persistent_kernel: cooperative_persistent_kernel.cu
	$(NVCC) $(NVCC_FLAGS) -lcuda -o $@ $<

# Test targets
test: $(CUDA_TARGETS)
	@echo "=== Testing Pipeline Examples ==="
	@echo "Testing double-buffered pipeline..."
	./double_buffered_pipeline 512 512 512
	@echo
	@echo "Testing warp-specialized pipeline..."
	./warp_specialized_pipeline 100 5
	@echo
	@echo "Testing cooperative persistent kernel..."
	./cooperative_persistent_kernel
	@echo "All tests completed successfully!"

# Benchmark targets
benchmark: $(CUDA_TARGETS)
	@echo "=== Pipeline Optimization Benchmarks ==="
	@echo "Small GEMM (512x512x512):"
	./double_buffered_pipeline 512 512 512
	@echo
	@echo "Medium GEMM (1024x1024x1024):"
	./double_buffered_pipeline 1024 1024 1024
	@echo
	@echo "Large GEMM (2048x2048x2048):"
	./double_buffered_pipeline 2048 2048 2048
	@echo
	@echo "Warp specialization with varying tile counts:"
	@echo "100 tiles:"
	./warp_specialized_pipeline 100 10
	@echo "500 tiles:"
	./warp_specialized_pipeline 500 10
	@echo "1000 tiles:"
	./warp_specialized_pipeline 1000 10
	@echo
	@echo "Persistent vs traditional kernel comparison:"
	./cooperative_persistent_kernel

# Profiling targets
profile-ncu: $(CUDA_TARGETS)
	@echo "=== Nsight Compute Profiling ==="
	@echo "Profiling double-buffered pipeline..."
	ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
		-o double_buffered_profile ./double_buffered_pipeline 1024 1024 1024
	@echo
	@echo "Profiling warp-specialized pipeline..."
	ncu --section WarpStateStats --section InstructionStats \
		-o warp_specialized_profile ./warp_specialized_pipeline 1000 5
	@echo
	@echo "Profiling cooperative kernel..."
	ncu --section LaunchStats --section WarpStateStats \
		-o cooperative_profile ./cooperative_persistent_kernel

profile-nsys: $(CUDA_TARGETS)
	@echo "=== Nsight Systems Profiling ==="
	nsys profile --force-overwrite=true -o pipeline_timeline \
		./double_buffered_pipeline 1024 1024 1024
	nsys profile --force-overwrite=true -o warp_specialized_timeline \
		./warp_specialized_pipeline 1000 10
	nsys profile --force-overwrite=true -o cooperative_timeline \
		./cooperative_persistent_kernel

# Detailed analysis
analyze: profile-ncu
	@echo "=== Performance Analysis Summary ==="
	@echo "Generated profiling reports:"
	@echo "  - double_buffered_profile.ncu-rep"
	@echo "  - warp_specialized_profile.ncu-rep" 
	@echo "  - cooperative_profile.ncu-rep"
	@echo
	@echo "Key metrics to examine in Nsight Compute:"
	@echo "1. SM Utilization (target: >90%)"
	@echo "2. Warp Execution Efficiency"
	@echo "3. Memory Bandwidth Utilization"
	@echo "4. Pipeline Stall Cycles"
	@echo "5. Instruction Count Reduction"
	@echo
	@echo "Expected improvements with pipelining:"
	@echo "  - 2x speedup with double-buffering"
	@echo "  - Additional 10-15% with warp specialization"
	@echo "  - Reduced shared memory stalls"
	@echo "  - Higher L2 cache throughput"

# Performance comparison
compare: $(CUDA_TARGETS)
	@echo "=== Performance Comparison ==="
	@echo "Running comprehensive comparison..."
	@echo "Matrix size: 1024x1024x1024"
	@echo
	@echo "1. Double-buffered pipeline vs naive:"
	./double_buffered_pipeline 1024 1024 1024
	@echo
	@echo "2. Warp specialization comparison:"
	./warp_specialized_pipeline 1000 10
	@echo
	@echo "3. Persistent vs traditional kernels:"
	./cooperative_persistent_kernel
	@echo
	@echo "Expected performance hierarchy:"
	@echo "  Naive < Double-buffered < Warp-specialized"
	@echo "  Traditional kernels < Persistent kernels"

# Educational targets
demo: $(CUDA_TARGETS)
	@echo "=== Pipeline Optimization Demo ==="
	@echo "This demo shows the progression of optimizations:"
	@echo
	@echo "1. Starting with double-buffered pipeline (2-stage)..."
	./double_buffered_pipeline 512 512 512
	@echo
	@echo "2. Moving to warp-specialized pipeline (3-stage)..."
	./warp_specialized_pipeline 500 5
	@echo
	@echo "3. Demonstrating persistent kernels..."
	./cooperative_persistent_kernel
	@echo
	@echo "Key takeaways:"
	@echo "  - Pipeline API eliminates __syncthreads() overhead"
	@echo "  - Warp specialization improves instruction-level parallelism"
	@echo "  - Persistent kernels reduce launch overhead"
	@echo "  - Cooperative groups enable fine-grained synchronization"

# Architecture-specific builds
sm_75: NVCC_FLAGS += -arch=sm_100
sm_75: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 7.5 (Turing)"

sm_80: NVCC_FLAGS += -arch=sm_100  
sm_80: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 8.0 (Ampere)"

sm_90: NVCC_FLAGS += -arch=sm_100
sm_90: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 9.0 (Blackwell)"

# Debug builds
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(CUDA_TARGETS)
	@echo "Debug builds created with device debugging enabled"

# Clean targets
clean:
	rm -f $(CUDA_TARGETS)
	rm -f *.ncu-rep *.nsys-rep *.qdrep
	rm -f *.log *.ptx

clean-profiles:
	rm -f *.ncu-rep *.nsys-rep *.qdrep

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all examples"
	@echo "  test         - Run basic functionality tests"
	@echo "  benchmark    - Run comprehensive benchmarks"
	@echo "  profile-ncu  - Profile with Nsight Compute"
	@echo "  profile-nsys - Profile with Nsight Systems"
	@echo "  analyze      - Generate detailed performance analysis"
	@echo "  compare      - Run performance comparisons"
	@echo "  demo         - Educational demonstration"
	@echo "  sm_75/80/90  - Build for specific compute capabilities"
	@echo "  debug        - Build with debugging symbols"
	@echo "  clean        - Remove built files"
	@echo "  help         - Show this help message"
	@echo
	@echo "Example usage:"
	@echo "  make benchmark                           # Run all benchmarks"
	@echo "  make profile-ncu                         # Profile with NCU"
	@echo "  ./double_buffered_pipeline 2048 2048 2048  # Custom GEMM size"
	@echo "  ./warp_specialized_pipeline 2000 20     # Custom tile count"

# Dependency checks
check-deps:
	@echo "Checking dependencies..."
	@nvcc --version || echo "ERROR: NVCC not found"
	@which ncu > /dev/null && echo "Nsight Compute available" || echo "WARNING: NCU not found"
	@which nsys > /dev/null && echo "Nsight Systems available" || echo "WARNING: nsys not found"
	@echo "Checking CUDA Pipeline API support..."
	@echo "#include <cuda/pipeline>" | $(NVCC) -x cu - -c -o /dev/null 2>/dev/null && \
		echo "CUDA Pipeline API supported" || echo "ERROR: Pipeline API not supported"

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
