#include <cuda_runtime.h>
#include <iostream>

// Launch many small kernels sequentially
__global__ void smallKernel(int* data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    data[idx] += 1;
}

int main() {
    const int N = 1<<20;
    int *d; cudaMalloc(&d, N*sizeof(int));
    for (int i = 0; i < 100; ++i) {
        smallKernel<<<N/256,256>>>(d);
        cudaDeviceSynchronize();
    }
    cudaFree(d);
    return 0;
}
