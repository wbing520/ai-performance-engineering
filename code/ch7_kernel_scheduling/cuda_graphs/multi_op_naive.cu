#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(float* data, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1.0f;
}

int main() {
    const int N = 1<<20;
    float *d; cudaMalloc(&d, N*sizeof(float));
    dim3 b(256), g((N+255)/256);
    for (int i=0; i<10; ++i) {
        addKernel<<<g,b>>>(d, N);
        cudaDeviceSynchronize();
    }
    cudaFree(d);
    return 0;
}
