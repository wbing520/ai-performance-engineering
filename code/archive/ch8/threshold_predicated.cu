// threshold_predicated.cu
// Chapter 8: Example demonstrating warp divergence reduction using predication

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void threshold_predicated(const float* X, float* Y, float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float x = X[i];
        // Use a conditional move or multiplication by boolean
        float val = (x > threshold) ? x : 0.0f;
        Y[i] = val;
    }
}

int main() {
    const int N = 1024 * 1024;
    const float threshold = 0.5f;
    
    // Allocate host memory
    std::vector<float> h_X(N), h_Y(N);
    
    // Initialize input data (half positive, half negative for maximum divergence test)
    for (int i = 0; i < N; i++) {
        h_X[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    
    // Allocate device memory
    float *d_X, *d_Y;
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_Y, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_X, h_X.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Warm up
    threshold_predicated<<<gridSize, blockSize>>>(d_X, d_Y, threshold, N);
    cudaDeviceSynchronize();
    
    // Profile with events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    threshold_predicated<<<gridSize, blockSize>>>(d_X, d_Y, threshold, N);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Predicated threshold kernel time: " << milliseconds << " ms" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_Y.data(), d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = (h_X[i] > threshold) ? h_X[i] : 0.0f;
        if (abs(h_Y[i] - expected) > 1e-6) {
            correct = false;
        }
    }
    
    std::cout << "Results: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
