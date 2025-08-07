TARGET = launch_batched_pipeline
CUDA_VERSION = 13.1NVCC   = nvcc
ARCH   = -arch=sm_100a
OPT    = -O3

all: warp_specialized_stream_pipeline launch_batched_pipeline
	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"	@echo "Building with Blackwell B200/B300 optimizations"
warp_specialized_stream_pipeline: warp_specialized_stream_pipeline.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o pipelineKernel.o -c $<

launch_batched_pipeline: launch_batched_pipeline.cpp pipelineKernel.o
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $^

clean:
	rm -f *.o launch_batched_pipeline
