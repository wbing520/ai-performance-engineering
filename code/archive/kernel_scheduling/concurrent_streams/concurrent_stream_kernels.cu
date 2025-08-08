// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <iostream>

// Launch small kernels concurrently on two streams
__global__ void smallKernel(int* data) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    data[idx] += 1;
}

int main() {
    const int N = 1<<20;
    int *d1, *d2;
    cudaMalloc(&d1, N*sizeof(int));
    cudaMalloc(&d2, N*sizeof(int));
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    for (int i = 0; i < 100; ++i) {
        smallKernel<<<N/256,256,0,s1>>>(d1);
        smallKernel<<<N/256,256,0,s2>>>(d2);
    }
    cudaDeviceSynchronize();
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d1); cudaFree(d2);
    return 0;
}

// CUDA 12.9 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 12.9 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
