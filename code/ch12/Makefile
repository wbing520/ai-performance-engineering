# Makefile for Chapter 12 - Dynamic Scheduling and CUDA Graphs
CUDA_VERSION = 12.9
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_100 --expt-relaxed-constexpr -rdc=true
PYTHON = python3

# CUDA targets
CUDA_TARGETS = atomic_work_queue cuda_graphs dynamic_parallelism

.PHONY: all clean test profile benchmark analyze help

all: $(CUDA_TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"
# Compilation rules
atomic_work_queue: atomic_work_queue.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cuda_graphs: cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

dynamic_parallelism: dynamic_parallelism.cu
	$(NVCC) $(NVCC_FLAGS) -lcudadevrt -o $@ $<

# Test targets
test: $(CUDA_TARGETS)
	@echo "=== Testing Dynamic Scheduling Examples ==="
	@echo "Testing atomic work queue..."
	./atomic_work_queue
	@echo
	@echo "Testing CUDA graphs..."
	./cuda_graphs
	@echo
	@echo "Testing dynamic parallelism..."
	./dynamic_parallelism
	@echo "All tests completed!"

# Benchmark targets
benchmark: $(CUDA_TARGETS)
	@echo "=== Dynamic Scheduling Benchmarks ==="
	@echo "1. Atomic work queue performance:"
	./atomic_work_queue
	@echo
	@echo "2. CUDA graphs overhead reduction:"
	./cuda_graphs
	@echo
	@echo "3. Dynamic parallelism flexibility:"
	./dynamic_parallelism

# Profiling targets
profile-ncu: $(CUDA_TARGETS)
	@echo "=== Nsight Compute Profiling ==="
	@echo "Profiling atomic work queue..."
	ncu --section MemoryWorkloadAnalysis --section WarpStateStats \
		-o atomic_queue_profile ./atomic_work_queue
	@echo
	@echo "Profiling CUDA graphs..."
	ncu --section LaunchStats \
		-o cuda_graphs_profile ./cuda_graphs
	@echo
	@echo "Profiling dynamic parallelism..."
	ncu --section LaunchStats --section WarpStateStats \
		-o dynamic_parallelism_profile ./dynamic_parallelism

profile-nsys: $(CUDA_TARGETS)
	@echo "=== Nsight Systems Profiling ==="
	nsys profile --force-overwrite=true -o atomic_queue_timeline ./atomic_work_queue
	nsys profile --force-overwrite=true -o graphs_timeline ./cuda_graphs
	nsys profile --force-overwrite=true -o dynamic_timeline ./dynamic_parallelism

# Detailed analysis
analyze: profile-ncu
	@echo "=== Performance Analysis Summary ==="
	@echo "Generated profiling reports:"
	@echo "  - atomic_queue_profile.ncu-rep"
	@echo "  - cuda_graphs_profile.ncu-rep"
	@echo "  - dynamic_parallelism_profile.ncu-rep"
	@echo
	@echo "Key metrics to examine:"
	@echo "1. Atomic Work Queue:"
	@echo "   - atomic_transactions_per_request (target: ~1.0)"
	@echo "   - SM utilization and load balance"
	@echo "   - Warp execution efficiency"
	@echo
	@echo "2. CUDA Graphs:"
	@echo "   - Launch overhead reduction"
	@echo "   - Kernel execution continuity"
	@echo "   - CPU-GPU handshake elimination"
	@echo
	@echo "3. Dynamic Parallelism:"
	@echo "   - Child kernel launch patterns"
	@echo "   - Resource utilization across generations"
	@echo "   - Kernel nesting depth and efficiency"

# Specific benchmarks
benchmark-atomic: atomic_work_queue
	@echo "=== Atomic Work Queue Detailed Benchmark ==="
	@echo "Testing load balancing effectiveness..."
	./atomic_work_queue
	@echo
	@echo "Key insights:"
	@echo "- Dynamic distribution eliminates idle SMs"
	@echo "- Batching reduces atomic contention"
	@echo "- Hierarchical distribution handles extreme imbalance"

benchmark-graphs: cuda_graphs
	@echo "=== CUDA Graphs Detailed Benchmark ==="
	@echo "Measuring launch overhead reduction..."
	./cuda_graphs
	@echo
	@echo "Key insights:"
	@echo "- Graphs reduce CPU involvement in GPU scheduling"
	@echo "- Parameter updates are microsecond-fast"
	@echo "- Conditional nodes enable device-side branching"

benchmark-dynamic: dynamic_parallelism
	@echo "=== Dynamic Parallelism Detailed Benchmark ==="
	@echo "Testing adaptive scheduling capabilities..."
	./dynamic_parallelism
	@echo
	@echo "Key insights:"
	@echo "- Device-side decisions eliminate CPU round-trips"
	@echo "- Recursive patterns map naturally to GPU"
	@echo "- Adaptive algorithms optimize per-element processing"

# Educational demonstrations
demo-load-balance: atomic_work_queue
	@echo "=== Load Balancing Demonstration ==="
	@echo "This demo shows how dynamic work distribution solves load imbalance:"
	./atomic_work_queue
	@echo
	@echo "Key concepts:"
	@echo "1. Static assignment leaves SMs idle with irregular workloads"
	@echo "2. Atomic queues enable dynamic work stealing"
	@echo "3. Batching amortizes atomic operation overhead"
	@echo "4. Hierarchical distribution scales to extreme imbalance"

demo-graph-efficiency: cuda_graphs
	@echo "=== Graph Efficiency Demonstration ==="
	@echo "This demo shows CUDA graphs eliminating launch overhead:"
	./cuda_graphs
	@echo
	@echo "Key benefits:"
	@echo "1. Single graph launch replaces multiple kernel launches"
	@echo "2. GPU scheduling optimization with known dependencies"
	@echo "3. Parameter updates avoid full recapture"
	@echo "4. Conditional nodes enable device-side control flow"

demo-device-orchestration: dynamic_parallelism
	@echo "=== Device Orchestration Demonstration ==="
	@echo "This demo shows GPU-driven scheduling without CPU:"
	./dynamic_parallelism
	@echo
	@echo "Key capabilities:"
	@echo "1. Kernels launch children based on runtime conditions"
	@echo "2. Recursive algorithms with natural GPU mapping"
	@echo "3. Adaptive processing per data characteristics"
	@echo "4. Device-resident schedulers eliminate CPU latency"

# Optimization guides
tune-atomics:
	@echo "=== Atomic Optimization Guide ==="
	@echo "1. Batch Size Selection:"
	@echo "   - Start with warp size (32) for moderate imbalance"
	@echo "   - Increase to 128-256 for extreme imbalance"
	@echo "   - Monitor atomic_transactions_per_request metric"
	@echo
	@echo "2. Distribution Strategy:"
	@echo "   - Warp-level: Good for most workloads"
	@echo "   - Block-level: Better for very irregular patterns"
	@echo "   - Hierarchical: Optimal for extreme variance"
	@echo
	@echo "3. Contention Reduction:"
	@echo "   - Use fewer, larger atomic operations"
	@echo "   - Consider per-SM counters for high contention"
	@echo "   - Leverage L2 cache locality for atomic variables"

tune-graphs:
	@echo "=== Graph Optimization Guide ==="
	@echo "1. Capture Strategy:"
	@echo "   - Pre-allocate all memory before capture"
	@echo "   - Warm up operations to initialize libraries"
	@echo "   - Capture with maximum expected problem size"
	@echo
	@echo "2. Memory Management:"
	@echo "   - Use static memory pools with graphs"
	@echo "   - Avoid malloc/free inside captured operations"
	@echo "   - Update parameters rather than recapturing"
	@echo
	@echo "3. Performance Optimization:"
	@echo "   - Group related operations in single graph"
	@echo "   - Use conditional nodes for device-side branching"
	@echo "   - Monitor launch overhead reduction with profiling"

tune-dynamic:
	@echo "=== Dynamic Parallelism Optimization Guide ==="
	@echo "1. Launch Decisions:"
	@echo "   - Profile overhead vs benefit of child launches"
	@echo "   - Use for data-dependent branching scenarios"
	@echo "   - Consider conditional graphs as alternative"
	@echo
	@echo "2. Resource Management:"
	@echo "   - Limit kernel nesting depth (typically 2-3 levels)"
	@echo "   - Monitor concurrent kernel slots usage"
	@echo "   - Balance parent and child resource requirements"
	@echo
	@echo "3. Performance Considerations:"
	@echo "   - Ensure sufficient work per child kernel"
	@echo "   - Minimize divergence in launch patterns"
	@echo "   - Use persistent kernels with device graphs when possible"

# Architecture-specific builds
sm_75: NVCC_FLAGS += -arch=sm_100
sm_75: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 7.5 (Turing)"
	@echo "Features: CUDA Graphs, Limited Dynamic Parallelism"

sm_80: NVCC_FLAGS += -arch=sm_100
sm_80: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 8.0 (Ampere)"
	@echo "Features: CUDA Graphs, Device Graph Launch, Full Dynamic Parallelism"

sm_90: NVCC_FLAGS += -arch=sm_100
sm_90: $(CUDA_TARGETS)
	@echo "Built for Compute Capability 9.0 (Blackwell)"
	@echo "Features: All optimizations, Enhanced Graph Capabilities"

# Debug builds
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(CUDA_TARGETS)
	@echo "Debug builds created with device debugging enabled"
	@echo "Use cuda-gdb for debugging device-side launches"

# Performance comparison
compare: $(CUDA_TARGETS)
	@echo "=== Performance Comparison ==="
	@echo "Running comparative analysis of scheduling techniques..."
	@echo
	@echo "1. Work Distribution Comparison:"
	./atomic_work_queue | grep -E "(Static|Dynamic|Speedup)"
	@echo
	@echo "2. Launch Overhead Comparison:"
	./cuda_graphs | grep -E "(Traditional|Graph|Speedup)"
	@echo
	@echo "3. Orchestration Flexibility:"
	./dynamic_parallelism | grep -E "(time|launched|iterations)"

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
	@echo "  all              - Build all examples"
	@echo "  test             - Run basic functionality tests"
	@echo "  benchmark        - Run comprehensive benchmarks"
	@echo "  profile-ncu      - Profile with Nsight Compute"
	@echo "  profile-nsys     - Profile with Nsight Systems"
	@echo "  analyze          - Generate detailed performance analysis"
	@echo
	@echo "Benchmark targets:"
	@echo "  benchmark-atomic - Test atomic work queue performance"
	@echo "  benchmark-graphs - Test CUDA graphs efficiency"
	@echo "  benchmark-dynamic - Test dynamic parallelism flexibility"
	@echo
	@echo "Demo targets:"
	@echo "  demo-load-balance      - Show load balancing benefits"
	@echo "  demo-graph-efficiency  - Show graph overhead reduction"
	@echo "  demo-device-orchestration - Show device-side scheduling"
	@echo
	@echo "Optimization guides:"
	@echo "  tune-atomics     - Atomic operation optimization guide"
	@echo "  tune-graphs      - CUDA graphs optimization guide"
	@echo "  tune-dynamic     - Dynamic parallelism optimization guide"
	@echo
	@echo "Other targets:"
	@echo "  compare          - Performance comparison summary"
	@echo "  sm_75/80/90      - Build for specific compute capabilities"
	@echo "  debug            - Build with debugging symbols"
	@echo "  clean            - Remove built files"
	@echo "  help             - Show this help message"
	@echo
	@echo "Example usage:"
	@echo "  make benchmark                    # Run all benchmarks"
	@echo "  make demo-load-balance           # See load balancing in action"
	@echo "  make profile-nsys                # Timeline analysis"
	@echo "  make tune-atomics                # Optimization guidelines"

# Dependencies check
check-deps:
	@echo "Checking dependencies..."
	@nvcc --version || echo "ERROR: NVCC not found"
	@which ncu > /dev/null && echo "Nsight Compute available" || echo "WARNING: NCU not found"
	@which nsys > /dev/null && echo "Nsight Systems available" || echo "WARNING: nsys not found"
	@echo "Checking device capabilities..."
	@echo "Run './atomic_work_queue' to verify device support for scheduling features"
	@echo "Run './cuda_graphs' to verify CUDA graph support"
	@echo "Run './dynamic_parallelism' to verify dynamic parallelism support"

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
