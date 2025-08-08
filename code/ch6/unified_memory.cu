// unified_memory.cu
// Example demonstrating CUDA Managed Memory (Unified Memory)

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void unifiedMemoryKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * data[idx] + 1.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    float *data;
    
    // Allocate unified memory accessible from both CPU and GPU
    cudaMallocManaged(&data, N * sizeof(float));
    
    // Initialize data on CPU
    for (int i = 0; i < N; ++i) {
        data[i] = float(i);
    }
    
    // Get GPU device ID for prefetching
    int device;
    cudaGetDevice(&device);
    
    // Prefetch data to GPU before kernel launch
    cudaMemPrefetchAsync(data, N * sizeof(float), device);
    
    // Give memory advice
    cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetReadMostly, device);
    
    // Launch kernel
    int blocks = (N + 255) / 256;
    int threads = 256;
    unifiedMemoryKernel<<<blocks, threads>>>(data, N);
    
    // Wait for kernel completion
    cudaDeviceSynchronize();
    
    // Prefetch data back to CPU for reading
    cudaMemPrefetchAsync(data, N * sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();
    
    // Access data on CPU
    printf("First 5 results: %.1f %.1f %.1f %.1f %.1f\n", 
           data[0], data[1], data[2], data[3], data[4]);
    
    // Free unified memory
    cudaFree(data);
    
    return 0;
}
