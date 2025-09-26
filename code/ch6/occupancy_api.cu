// Architecture-specific optimizations for CUDA 12.8
// Targets Blackwell B200/B300 (sm_100)
// occupancy_api.cu
// Example demonstrating CUDA Occupancy API

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sampleKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

int main() {
    const int N = 1024 * 1024;
    
    float *h_data, *d_data;
    h_data = new float[N];
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = float(i % 1000) / 1000.0f;
    }
    
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Use occupancy API to find optimal block size
    int minGridSize = 0, bestBlockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize,
        sampleKernel,
        /* dynamicSmemBytes = */ 0,
        /* blockSizeLimit = */ 0
    );
    
    printf("Optimal block size: %d\n", bestBlockSize);
    printf("Minimum grid size: %d\n", minGridSize);
    
    // Launch with optimal block size
    int gridSize = (N + bestBlockSize - 1) / bestBlockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    sampleKernel<<<gridSize, bestBlockSize>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Copy result back
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Execution time: %.2f ms\n", ms);
    printf("First result: %.3f\n", h_data[0]);
    
    // Test with different block sizes for comparison
    printf("\nTesting different block sizes:\n");
    int testSizes[] = {64, 128, 256, 512, 1024};
    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);
    
    for (int i = 0; i < numTests; ++i) {
        int blockSize = testSizes[i];
        int gridSizeTest = (N + blockSize - 1) / blockSize;
        
        cudaEventRecord(start);
        sampleKernel<<<gridSizeTest, blockSize>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float msTest;
        cudaEventElapsedTime(&msTest, start, stop);
        
        printf("Block size %d: %.2f ms\n", blockSize, msTest);
    }
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// CUDA 12.8 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 12.8 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
