all: addSequential addParallel

addSequential: addSequential.cu
	nvcc -std=c++17 -arch=sm_90 -O3 $< -o addSequential

addParallel: addParallel.cu
	nvcc -std=c++17 -arch=sm_90 -O3 $< -o addParallel

clean:
	rm -f addSequential addParallel
