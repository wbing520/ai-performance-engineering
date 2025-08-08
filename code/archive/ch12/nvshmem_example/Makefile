NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90a -std=c++17
TARGET = nvshmem_example

# NVSHMEM library flags (adjust paths as needed)
NVSHMEM_INC = -I/usr/local/nvshmem/include
NVSHMEM_LIB = -L/usr/local/nvshmem/lib -lnvshmem

all: $(TARGET)

$(TARGET): nvshmem_example.cu
	$(NVCC) $(NVCC_FLAGS) $(NVSHMEM_INC) -o $(TARGET) $< $(NVSHMEM_LIB)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	# Run with 2 PEs (GPUs)
	mpirun -np 2 ./$(TARGET)

.PHONY: all clean run
