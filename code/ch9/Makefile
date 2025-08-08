# Makefile for Chapter 9 - Kernel Fusion and Arithmetic Intensity

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80
PYTHON = python3

# CUDA targets
CUDA_TARGETS = fusedL2Norm inline_ptx_example cutlass_gemm_example

# Python targets  
PYTHON_TARGETS = fusion_pytorch.py

.PHONY: all clean cuda python test profile

all: cuda

cuda: $(CUDA_TARGETS)

python:
	@echo "Python examples ready to run:"
	@echo "  python $(PYTHON_TARGETS)"

# CUDA compilation rules
fusedL2Norm: fusedL2Norm.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

inline_ptx_example: inline_ptx_example.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

cutlass_gemm_example: cutlass_gemm_example.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_PATH)/include -o $@ $<

# Test targets
test: cuda
	@echo "=== Testing Kernel Fusion Examples ==="
	@echo "Running fused L2 normalization..."
	./fusedL2Norm 32 512 10
	@echo
	@echo "Running inline PTX examples..."
	./inline_ptx_example
	@echo
	@echo "Running CUTLASS GEMM (if available)..."
	-./cutlass_gemm_example
	@echo
	@echo "Running PyTorch fusion examples..."
	$(PYTHON) fusion_pytorch.py

# Profiling targets
profile-ncu: cuda
	@echo "=== Nsight Compute Profiling ==="
	ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis -o fusedL2Norm_profile ./fusedL2Norm 128 4096 10
	ncu --section InstructionStats --section LaunchStats -o ptx_profile ./inline_ptx_example

profile-nsys: cuda
	@echo "=== Nsight Systems Profiling ==="
	nsys profile --force-overwrite=true -o fusion_analysis ./fusedL2Norm 128 4096 100
	nsys profile --force-overwrite=true -o ptx_analysis ./inline_ptx_example

profile-pytorch:
	@echo "=== PyTorch Profiling ==="
	$(PYTHON) -c "import fusion_pytorch; fusion_pytorch.benchmark_fusion()"

# Benchmark targets
benchmark: cuda
	@echo "=== Kernel Fusion Benchmarks ==="
	@echo "Small problem (batch=32, hidden=512):"
	./fusedL2Norm 32 512 100
	@echo
	@echo "Medium problem (batch=128, hidden=2048):"
	./fusedL2Norm 128 2048 100
	@echo
	@echo "Large problem (batch=256, hidden=4096):"
	./fusedL2Norm 256 4096 100
	@echo
	@echo "PyTorch fusion benchmark:"
	$(PYTHON) fusion_pytorch.py

# Analysis targets
analyze: profile-ncu
	@echo "=== Performance Analysis ==="
	@echo "Open the following files in Nsight Compute:"
	@echo "  - fusedL2Norm_profile.ncu-rep"
	@echo "  - ptx_profile.ncu-rep"
	@echo
	@echo "Key metrics to examine:"
	@echo "  - Memory bandwidth utilization"
	@echo "  - Arithmetic intensity (FLOP/byte)"
	@echo "  - Cache hit rates"
	@echo "  - Instruction mix"

# Clean targets
clean:
	rm -f $(CUDA_TARGETS)
	rm -f *.ncu-rep *.nsys-rep *.qdrep
	rm -f *.json *.log

# Documentation
help:
	@echo "Available targets:"
	@echo "  all       - Build all CUDA examples"
	@echo "  cuda      - Build CUDA examples"
	@echo "  python    - Show Python example info"
	@echo "  test      - Run all examples with small inputs"
	@echo "  benchmark - Run comprehensive benchmarks"
	@echo "  profile-ncu   - Profile with Nsight Compute"
	@echo "  profile-nsys  - Profile with Nsight Systems" 
	@echo "  profile-pytorch - Profile PyTorch examples"
	@echo "  analyze   - Generate and explain profiling reports"
	@echo "  clean     - Remove generated files"
	@echo "  help      - Show this help message"
	@echo
	@echo "Example usage:"
	@echo "  make benchmark                    # Run all benchmarks"
	@echo "  make profile-ncu                  # Profile with NCU"
	@echo "  ./fusedL2Norm 128 4096 100       # Custom L2 norm benchmark"
	@echo "  python fusion_pytorch.py         # PyTorch fusion examples"

# Dependencies check
check-deps:
	@echo "Checking dependencies..."
	@nvcc --version || echo "ERROR: NVCC not found"
	@$(PYTHON) -c "import torch; print(f'PyTorch {torch.__version__}')" || echo "ERROR: PyTorch not found"
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@which ncu > /dev/null && echo "Nsight Compute available" || echo "WARNING: Nsight Compute not found"
	@which nsys > /dev/null && echo "Nsight Systems available" || echo "WARNING: Nsight Systems not found"
