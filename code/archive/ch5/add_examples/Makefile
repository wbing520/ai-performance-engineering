all: addSequential addParallel
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"CUDA_VERSION = 13.1
addSequential: addSequential.cu
	nvcc -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -arch=sm_100a -O3 $< -o addSequential

addParallel: addParallel.cu
	nvcc -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -DCUDA_VERSION=$(CUDA_VERSION) -arch=sm_100a -O3 $< -o addParallel

clean:
	rm -f addSequential addParallel
