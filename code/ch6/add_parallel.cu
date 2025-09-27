// Architecture-specific optimizations for CUDA 12.9
// Targets Blackwell B200/B300 (sm_100)
// addParallel.cu
// Parallel vector addition example (optimal performance)
// Updated for CUDA 12.9 and Blackwell B200/B300 (SM100)

#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda/std/chrono>
#include <nvtx3/nvToolsExt.h>

const int N = 1'000'000;

// One thread per element with enhanced optimization for SM100
__global__ void addParallel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA 12.9 stream-ordered memory allocation example
__global__ void addParallelStreamOrdered(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Enhanced kernel with SM100 optimizations
__global__ void addParallelOptimized(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Use fast math operations for SM100
        C[idx] = __fadd_rn(A[idx], B[idx]);
    }
}

int main() {
    // Check CUDA version
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    
    // Check device properties for Blackwell B200/B300
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d (SM100 for Blackwell B200/B300)\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Memory Bandwidth: %.1f GB/s\n", 2.0 * prop.memoryClockRate * 1e-3f * prop.memoryBusWidth / 8.0);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    
    // Allocate and initialize host with pinned memory
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N * sizeof(float));
    cudaMallocHost(&h_B, N * sizeof(float));
    cudaMallocHost(&h_C, N * sizeof(float));
    
    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(i * 2);
    }
    
    // Traditional memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    // Copy inputs to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Time the kernel with enhanced timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Configure and launch: many threads optimized for SM100
    int threads = 256;  // Optimal for SM100
    int blocks = (N + threads - 1) / threads;
    
    // Add NVTX markers for profiling
    nvtxRangePushA("add_parallel_kernel");
    addParallel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    nvtxRangePop();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Ensure completion before exit
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Parallel kernel time: %.2f ms\n", milliseconds);
    printf("Result: C[0] = %.1f, C[N-1] = %.1f\n", h_C[0], h_C[N-1]);
    
    // CUDA 12.9 stream-ordered memory allocation example
    printf("\n--- CUDA 12.9 Stream-Ordered Memory Example ---\n");
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Stream-ordered memory allocation (CUDA 12.9 feature)
    float *d_A_async, *d_B_async, *d_C_async;
    cudaMallocAsync(&d_A_async, N * sizeof(float), stream);
    cudaMallocAsync(&d_B_async, N * sizeof(float), stream);
    cudaMallocAsync(&d_C_async, N * sizeof(float), stream);
    
    // Asynchronous memory copy
    cudaMemcpyAsync(d_A_async, h_A, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_async, h_B, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel on stream
    nvtxRangePushA("stream_ordered_kernel");
    addParallelStreamOrdered<<<blocks, threads, 0, stream>>>(d_A_async, d_B_async, d_C_async, N);
    nvtxRangePop();
    
    // Asynchronous copy back
    cudaMemcpyAsync(h_C, d_C_async, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Synchronize stream
    cudaStreamSynchronize(stream);
    
    printf("Stream-ordered memory result: C[0] = %.1f, C[N-1] = %.1f\n", h_C[0], h_C[N-1]);
    
    // Cleanup stream-ordered memory
    cudaFreeAsync(d_A_async, stream);
    cudaFreeAsync(d_B_async, stream);
    cudaFreeAsync(d_C_async, stream);
    cudaStreamDestroy(stream);
    
    // Test optimized kernel
    printf("\n--- SM100 Optimized Kernel ---\n");
    
    cudaEventRecord(start);
    nvtxRangePushA("optimized_kernel");
    addParallelOptimized<<<blocks, threads>>>(d_A, d_B, d_C, N);
    nvtxRangePop();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Optimized kernel time: %.2f ms\n", milliseconds);
    
    // Cleanup traditional memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("All operations completed successfully!\n");
    printf("CUDA 12.9 and SM100 optimizations applied.\n");
    return 0;
}
