TARGET = launch_batched_pipeline
NVCC   = nvcc
ARCH   = -arch=sm_90
OPT    = -O3

all: warp_specialized_stream_pipeline launch_batched_pipeline

warp_specialized_stream_pipeline: warp_specialized_stream_pipeline.cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o pipelineKernel.o -c $<

launch_batched_pipeline: launch_batched_pipeline.cpp pipelineKernel.o
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -o $@ $^

clean:
	rm -f *.o launch_batched_pipeline
