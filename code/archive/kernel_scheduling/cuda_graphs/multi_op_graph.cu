#include <cuda_runtime.h>
#include <iostream>

int main() {
    const int N = 1<<20;
    float *d; cudaMalloc(&d, N*sizeof(float));
    dim3 b(256), g((N+255)/256);

    cudaGraph_t graph;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraphCreate(&graph, 0);

    for(int i=0; i<10; ++i) {
        cudaGraphNode_t node;
        cudaKernelNodeParams params = {0};
        params.func = (void*)addKernel;
        params.gridDim = g; params.blockDim = b;
        params.kernelParams = new void*[2]{&d, &N};
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);
    }

    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d);
    return 0;
}

// Kernel definition reused
__global__ void addKernel(float* data, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1.0f;
}
