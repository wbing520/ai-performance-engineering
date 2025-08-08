// fusedL2Norm.cu
// Chapter 9: Example demonstrating kernel fusion for improved arithmetic intensity

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void fusedL2Norm(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float ai = a[i];
        float bi = b[i];
        
        // Perform multiple arithmetic ops on ai and bi before storing:
        float sumsq = ai * ai + bi * bi;
        out[i] = sqrtf(sumsq);
    }
}

// Naive implementation for comparison (4 separate kernels)
__global__ void square_a(const float *a, float *a_sq, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a_sq[i] = a[i] * a[i];
}

__global__ void square_b(const float *b, float *b_sq, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) b_sq[i] = b[i] * b[i];
}

__global__ void add_squares(const float *a_sq, const float *b_sq, float *sum_sq, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) sum_sq[i] = a_sq[i] + b_sq[i];
}

__global__ void sqrt_result(const float *sum_sq, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = sqrtf(sum_sq[i]);
}

int main() {
    const int N = 1024 * 1024;
    
    // Allocate host memory
    std::vector<float> h_a(N), h_b(N), h_out_fused(N), h_out_naive(N);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i) / N;
        h_b[i] = static_cast<float>(N - i) / N;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_out_fused, *d_out_naive;
    float *d_a_sq, *d_b_sq, *d_sum_sq; // For naive implementation
    
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out_fused, N * sizeof(float));
    cudaMalloc(&d_out_naive, N * sizeof(float));
    cudaMalloc(&d_a_sq, N * sizeof(float));
    cudaMalloc(&d_b_sq, N * sizeof(float));
    cudaMalloc(&d_sum_sq, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    fusedL2Norm<<<gridSize, blockSize>>>(d_a, d_b, d_out_fused, N);
    cudaDeviceSynchronize();
    
    // Time fused kernel
    cudaEventRecord(start);
    fusedL2Norm<<<gridSize, blockSize>>>(d_a, d_b, d_out_fused, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float fused_time = 0;
    cudaEventElapsedTime(&fused_time, start, stop);
    
    // Time naive implementation (4 separate kernels)
    cudaEventRecord(start);
    square_a<<<gridSize, blockSize>>>(d_a, d_a_sq, N);
    square_b<<<gridSize, blockSize>>>(d_b, d_b_sq, N);
    add_squares<<<gridSize, blockSize>>>(d_a_sq, d_b_sq, d_sum_sq, N);
    sqrt_result<<<gridSize, blockSize>>>(d_sum_sq, d_out_naive, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    
    std::cout << "Fused kernel time: " << fused_time << " ms" << std::endl;
    std::cout << "Naive (4 kernels) time: " << naive_time << " ms" << std::endl;
    std::cout << "Speedup: " << naive_time / fused_time << "x" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(h_out_fused.data(), d_out_fused, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_naive.data(), d_out_naive, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results match
    bool correct = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N && correct; i++) {
        float expected = std::sqrt(h_a[i] * h_a[i] + h_b[i] * h_b[i]);
        float diff_fused = std::abs(h_out_fused[i] - expected);
        float diff_naive = std::abs(h_out_naive[i] - expected);
        max_diff = std::max({max_diff, diff_fused, diff_naive});
        
        if (diff_fused > 1e-5 || diff_naive > 1e-5) {
            correct = false;
        }
    }
    
    std::cout << "Results: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Maximum difference: " << max_diff << std::endl;
    
    // Calculate arithmetic intensities
    float bytes_fused = 3 * N * sizeof(float); // Read a, b; Write out
    float bytes_naive = 6 * N * sizeof(float); // Multiple intermediate reads/writes
    float flops = 4 * N; // 2 multiplies, 1 add, 1 sqrt per element
    
    std::cout << "\nArithmetic Intensity Analysis:" << std::endl;
    std::cout << "Fused kernel: " << flops / bytes_fused << " FLOPs/byte" << std::endl;
    std::cout << "Naive kernels: " << flops / bytes_naive << " FLOPs/byte" << std::endl;
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out_fused);
    cudaFree(d_out_naive);
    cudaFree(d_a_sq);
    cudaFree(d_b_sq);
    cudaFree(d_sum_sq);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
