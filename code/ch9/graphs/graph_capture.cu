// graph_capture.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelA(float* X, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) X[idx] *= 1.1f;
}
__global__ void kernelB(float* X, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) X[idx] += 2.0f;
}
__global__ void kernelC(float* X, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) X[idx] = sqrtf(X[idx]);
}

int main() {
    const int N = 1<<20;
    float *d_X;
    cudaMalloc(&d_X, N*sizeof(float));
    cudaMemset(d_X, 1, N*sizeof(float));

    dim3 block(256), grid((N+255)/256);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernelA<<<grid,block,0,stream>>>(d_X,N);
    kernelB<<<grid,block,0,stream>>>(d_X,N);
    kernelC<<<grid,block,0,stream>>>(d_X,N);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    for(int i=0;i<100;++i) {
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaFree(d_X);
    std::cout<<"Graph capture replay complete"<<std::endl;
    return 0;
}
