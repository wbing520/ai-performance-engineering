# Makefile for Chapter 16: Profiling, Debugging, and Tuning Inference at Scale

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
