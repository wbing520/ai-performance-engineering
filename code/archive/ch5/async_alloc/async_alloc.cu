// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(float* buf, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) buf[idx] = buf[idx] * 2.0f;
}

int main() {
    cudaStream_t s;
    cudaStreamCreate(&s);

    const int N = 1<<20;
    float *d_buf;
    cudaMallocAsync(&d_buf, N * sizeof(float), s);

    int threads = 256, blocks = (N + threads-1)/threads;
    myKernel<<<blocks, threads, 0, s>>>(d_buf, N);

    cudaFreeAsync(d_buf, s);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    return 0;
}
