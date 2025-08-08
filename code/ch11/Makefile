# Makefile for Chapter 11 - Inter-Kernel Pipelining and Stream-Ordered Allocation
CUDA_VERSION = 12.9
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_100 --expt-relaxed-constexpr
PYTHON = python3

# CUDA targets
CUDA_TARGETS = basic_streams stream_ordered_allocator multi_stream_pipeline

.PHONY: all clean test profile benchmark analyze help check-deps

all: $(CUDA_TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Compilation rules
basic_streams: basic_streams.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

stream_ordered_allocator: stream_ordered_allocator.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

multi_stream_pipeline: multi_stream_pipeline.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Test targets
test: $(CUDA_TARGETS)
	@echo "=== Testing CUDA Streams Examples ==="
	@echo "Testing basic streams..."
	./basic_streams
	@echo
	@echo "Testing stream-ordered allocator..."
	./stream_ordered_allocator
	@echo
	@echo "Testing multi-stream pipeline..."
	./multi_stream_pipeline
	@echo "All tests completed successfully!"

# Benchmark targets
benchmark: $(CUDA_TARGETS)
	@echo "=== Stream Performance Benchmarks ==="
	@echo "1. Basic streams with kernel overlap:"
	./basic_streams
	@echo
	@echo "2. Stream-ordered vs traditional allocation:"
	./stream_ordered_allocator
	@echo
	@echo "3. Multi-stream pipeline with warp specialization:"
	./multi_stream_pipeline

# Profiling targets
profile-nsys: $(CUDA_TARGETS)
	@echo "=== Nsight Systems Profiling ==="
	@echo "Profiling basic streams..."
	nsys profile --force-overwrite=true -o basic_streams_timeline ./basic_streams
	@echo
	@echo "Profiling stream-ordered allocator..."
	nsys profile --force-overwrite=true -o stream_alloc_timeline ./stream_ordered_allocator
	@echo
	@echo "Profiling multi-stream pipeline..."
	nsys profile --force-overwrite=true -o multi_stream_timeline ./multi_stream_pipeline

profile-ncu: $(CUDA_TARGETS)
	@echo "=== Nsight Compute Profiling ==="
	@echo "Profiling basic streams..."
	ncu --section LaunchStats --section MemoryWorkloadAnalysis \
		-o basic_streams_profile ./basic_streams
	@echo
	@echo "Profiling stream-ordered allocator..."
	ncu --section MemoryWorkloadAnalysis \
		-o stream_alloc_profile ./stream_ordered_allocator
	@echo
	@echo "Profiling multi-stream pipeline..."
	ncu --section WarpStateStats --section LaunchStats \
		-o multi_stream_profile ./multi_stream_pipeline

# Analysis targets
analyze: profile-nsys
	@echo "=== Performance Analysis Summary ==="
	@echo "Generated timeline reports:"
	@echo "  - basic_streams_timeline.nsys-rep"
	@echo "  - stream_alloc_timeline.nsys-rep"
	@echo "  - multi_stream_timeline.nsys-rep"
	@echo
	@echo "Key patterns to look for in Nsight Systems:"
	@echo "1. Overlapping kernel execution across streams"
	@echo "2. Simultaneous H2D, compute, and D2H operations"
	@echo "3. Non-blocking memory allocation patterns"
	@echo "4. Event-based synchronization efficiency"
	@echo "5. Absence of global synchronization barriers"
	@echo
	@echo "Performance expectations:"
	@echo "  - 2-3x throughput with proper stream overlap"
	@echo "  - 50-80% reduction in allocation overhead"
	@echo "  - Near-linear scaling with concurrent streams"

# Stream-specific benchmarks
benchmark-overlap: basic_streams
	@echo "=== Stream Overlap Analysis ==="
	@echo "Running stream overlap benchmark..."
	./basic_streams
	@echo
	@echo "Expected results:"
	@echo "  - Explicit streams should show 1.5-2x speedup vs default stream"
	@echo "  - Mixed usage should show performance degradation"
	@echo "  - Timeline should show concurrent kernel execution"

benchmark-allocation: stream_ordered_allocator
	@echo "=== Memory Allocation Benchmark ==="
	@echo "Comparing traditional vs stream-ordered allocation..."
	./stream_ordered_allocator
	@echo
	@echo "Expected results:"
	@echo "  - Stream-ordered allocation should be faster"
	@echo "  - Better overlap in LLM-style variable-length workloads"
	@echo "  - Reduced global synchronization overhead"

benchmark-pipeline: multi_stream_pipeline
	@echo "=== Multi-Stream Pipeline Benchmark ==="
	@echo "Testing combined intra-kernel and inter-kernel pipelining..."
	./multi_stream_pipeline
	@echo
	@echo "Expected results:"
	@echo "  - Multiple batches processed concurrently"
	@echo "  - Warp specialization within each kernel"
	@echo "  - Event-based synchronization working correctly"

# Educational demos
demo-default-stream: basic_streams
	@echo "=== Default Stream Pitfalls Demo ==="
	@echo "This demo shows why default stream usage hurts performance:"
	./basic_streams
	@echo
	@echo "Key lessons:"
	@echo "  1. Default stream creates global barriers"
	@echo "  2. Explicit streams enable concurrency"
	@echo "  3. Mixed usage is worst of both worlds"
	@echo "  4. Always use explicit streams for performance"

demo-events: multi_stream_pipeline
	@echo "=== Event-Based Synchronization Demo ==="
	@echo "This demo shows fine-grained stream coordination:"
	./multi_stream_pipeline
	@echo
	@echo "Key concepts:"
	@echo "  1. Events provide precise synchronization"
	@echo "  2. No blocking the CPU or entire device"
	@echo "  3. Producer-consumer patterns"
	@echo "  4. Host callbacks for async coordination"

demo-allocation: stream_ordered_allocator
	@echo "=== Stream-Ordered Allocation Demo ==="
	@echo "This demo shows modern memory management:"
	./stream_ordered_allocator
	@echo
	@echo "Key benefits:"
	@echo "  1. No global device synchronization"
	@echo "  2. Perfect for variable-length sequences"
	@echo "  3. Essential for LLM workloads"
	@echo "  4. Memory pool reduces OS overhead"

# PyTorch integration help
pytorch-setup:
	@echo "=== PyTorch Stream-Ordered Allocation Setup ==="
	@echo "To enable stream-ordered allocation in PyTorch:"
	@echo
	@echo "export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync"
	@echo
	@echo "Add this to your .bashrc or run before Python scripts:"
	@echo
	@echo "import torch"
	@echo "# PyTorch will now use stream-ordered allocation automatically"
	@echo "# DataLoader with pin_memory=True enables async transfers"
	@echo
	@echo "Benefits:"
	@echo "  - Faster tensor allocation/deallocation"
	@echo "  - Better overlap with computation"
	@echo "  - Reduced global synchronization"

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

# Performance tuning
tune-streams:
	@echo "=== Stream Performance Tuning Guide ==="
	@echo "1. Optimal number of streams:"
	@echo "   - Start with 2-4 streams"
	@echo "   - Increase until no performance gain"
	@echo "   - Consider memory constraints"
	@echo
	@echo "2. Memory allocation strategy:"
	@echo "   - Use stream-ordered allocation"
	@echo "   - Tune memory pool release threshold"
	@echo "   - Pre-allocate for predictable workloads"
	@echo
	@echo "3. Synchronization optimization:"
	@echo "   - Use events instead of stream sync"
	@echo "   - Minimize synchronization points"
	@echo "   - Avoid default stream usage"
	@echo
	@echo "4. Batch size considerations:"
	@echo "   - Balance parallelism vs memory usage"
	@echo "   - Consider kernel launch overhead"
	@echo "   - Profile different configurations"

# Clean targets
clean:
	rm -f $(CUDA_TARGETS)
	rm -f *.nsys-rep *.ncu-rep *.qdrep
	rm -f *.log *.ptx

clean-profiles:
	rm -f *.nsys-rep *.ncu-rep *.qdrep

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all examples"
	@echo "  test             - Run basic functionality tests"
	@echo "  benchmark        - Run comprehensive benchmarks"
	@echo "  profile-nsys     - Profile with Nsight Systems"
	@echo "  profile-ncu      - Profile with Nsight Compute"
	@echo "  analyze          - Generate detailed performance analysis"
	@echo
	@echo "Benchmark targets:"
	@echo "  benchmark-overlap     - Test stream overlap patterns"
	@echo "  benchmark-allocation  - Compare allocation strategies"
	@echo "  benchmark-pipeline    - Test multi-stream pipeline"
	@echo
	@echo "Demo targets:"
	@echo "  demo-default-stream   - Show default stream pitfalls"
	@echo "  demo-events          - Show event-based synchronization"
	@echo "  demo-allocation      - Show stream-ordered allocation"
	@echo
	@echo "Other targets:"
	@echo "  pytorch-setup    - PyTorch integration guide"
	@echo "  tune-streams     - Performance tuning guide"
	@echo "  sm_75/80/90      - Build for specific compute capabilities"
	@echo "  debug            - Build with debugging symbols"
	@echo "  clean            - Remove built files"
	@echo "  help             - Show this help message"
	@echo
	@echo "Example usage:"
	@echo "  make benchmark                    # Run all benchmarks"
	@echo "  make profile-nsys                 # Profile with timeline view"
	@echo "  make demo-default-stream          # See default stream issues"

# Dependencies check
check-deps:
	@echo "Checking dependencies..."
	@nvcc --version || echo "ERROR: NVCC not found"
	@which nsys > /dev/null && echo "Nsight Systems available" || echo "WARNING: nsys not found"
	@which ncu > /dev/null && echo "Nsight Compute available" || echo "WARNING: ncu not found"
	@echo "Checking CUDA features..."
	@echo "#include <cuda/pipeline>" | $(NVCC) -x cu - -c -o /dev/null 2>/dev/null && \
		echo "CUDA Pipeline API supported" || echo "ERROR: Pipeline API not supported"
	@echo "Checking device capabilities..."
	@echo "Run 'deviceQuery' or check device properties for:"
	@echo "  - Concurrent kernels support"
	@echo "  - Async engine count (copy engines)"
	@echo "  - Memory pools support"

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
