#include <cuda_runtime.h>

// Child kernel
__global__ void childKernel(int *data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    data[idx] += 1;
}

// Parent launches child kernels via device
__global__ void parentKernel(int *data, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        childKernel<<<gridDim, blockDim>>>(data);
    }
}

int main() {
    const int N = 1<<20;
    int *d; cudaMalloc(&d, N*sizeof(int));
    parentKernel<<<N/256,256>>>(d, N);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
