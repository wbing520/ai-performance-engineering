#include <cuda_runtime.h>
#include <iostream>
#define NUM_STREAMS 2
#define TILE_ELEMS 1024

void launch_batched_pipeline(const float* hA, const float* hB, float* hC, int batches){
    cudaStream_t streams[NUM_STREAMS];
    for(int i=0;i<NUM_STREAMS;++i) cudaStreamCreate(&streams[i]);
    for(int b=0;b<batches;++b){
        int sid=b%NUM_STREAMS;
        auto s=streams[sid];
        float *dA,*dB,*dC;
        size_t bytes=TILE_ELEMS*sizeof(float);
        cudaMallocAsync(&dA,bytes,s);
        cudaMallocAsync(&dB,bytes,s);
        cudaMallocAsync(&dC,bytes,s);
        cudaMemcpyAsync(dA,hA+b*TILE_ELEMS,bytes,cudaMemcpyHostToDevice,s);
        cudaMemcpyAsync(dB,hB+b*TILE_ELEMS,bytes,cudaMemcpyHostToDevice,s);
        // launch kernel in separate file
        extern void pipelineKernel(float*,float*,float*,int);
        pipelineKernel<<<1,96,3*TILE_ELEMS*sizeof(float),s>>>(dA,dB,dC,1);
        cudaMemcpyAsync(hC+b*TILE_ELEMS,dC,bytes,cudaMemcpyDeviceToHost,s);
        cudaFreeAsync(dA,s); cudaFreeAsync(dB,s); cudaFreeAsync(dC,s);
    }
    for(int i=0;i<NUM_STREAMS;++i) cudaStreamDestroy(streams[i]);
}

int main(){ return 0; }