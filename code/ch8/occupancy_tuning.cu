#include <cuda_runtime.h>
#include <iostream>

// Example kernel with __launch_bounds__ to guide occupancy
__global__ __launch_bounds__(256, 8)
void optimizedKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Some computation work
        float val = input[idx];
        val = val * val + val * 2.0f + 1.0f;
        output[idx] = sqrtf(val);
    }
}

// Standard kernel without launch bounds for comparison
__global__ void standardKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Same computation work
        float val = input[idx];
        val = val * val + val * 2.0f + 1.0f;
        output[idx] = sqrtf(val);
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float* h_input = nullptr;
    float* h_output = nullptr;
    cudaMallocHost(&h_input, bytes);
    cudaMallocHost(&h_output, bytes);
    
    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 1000);
    }
    
    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Copy to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch configuration guided by __launch_bounds__
    // We promised max 256 threads per block and min 8 blocks per SM
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    // Time the optimized kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    optimizedKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float optimizedTime;
    cudaEventElapsedTime(&optimizedTime, start, stop);
    
    // Time the standard kernel for comparison
    cudaEventRecord(start);
    standardKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float standardTime;
    cudaEventElapsedTime(&standardTime, start, stop);
    
    printf("Optimized kernel time: %.3f ms\n", optimizedTime);
    printf("Standard kernel time: %.3f ms\n", standardTime);
    printf("Speedup: %.2fx\n", standardTime / optimizedTime);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    
    return 0;
}
