// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <iostream>

__global__ void uncoalescedCopy(const float* __restrict__ in, float* __restrict__ out, int N, int stride) {
    // n = 1048576, stride = 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Loads from in[] with a stride, causing
        // multiple memory segments to be fetched
        out[idx] = in[idx * stride];
    }
}

int main() {
    const int N = 1 << 20;
    const int stride = 2;
    
    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * stride * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));
    
    for (int i = 0; i < N * stride; ++i) {
        h_in[i] = static_cast<float>(i);
    }
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * stride * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    cudaMemcpy(d_in, h_in, N * stride * sizeof(float), cudaMemcpyHostToDevice);
    
    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;
    // Number of blocks per grid
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    uncoalescedCopy<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N, stride);
    cudaDeviceSynchronize();
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    
    return 0;
}
