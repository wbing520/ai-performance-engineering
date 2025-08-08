// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// cooperative_persistent_kernel.cu
// Chapter 10: Example demonstrating cooperative persistent kernels

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

// Simple computation functions for demonstration
__device__ float someComputationA(float x) {
    return x * x + 2.0f * x + 1.0f;
}

__device__ float someComputationB(float mid, float x) {
    return mid * x + 0.5f;
}

__global__ void combinedKernel(float* dataA, float* dataB, int N, int iterations) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int it = 0; it < iterations; ++it) {
        // Stage 1: e.g., process dataA and produce intermediate results
        if (idx < N) {
            dataA[idx] = someComputationA(dataA[idx]);
        }
        
        // Global sync across all blocks before proceeding
        grid.sync();
        
        // Stage 2: e.g., read intermediate results and process dataB
        if (idx < N) {
            float mid = dataA[idx]; // uses completed dataA from stage 1
            dataB[idx] = someComputationB(mid, dataB[idx]);
        }
        
        // Global sync before next iteration
        grid.sync();
    }
}

// Naive implementation for comparison (launches separate kernels)
__global__ void stageA_kernel(float* dataA, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dataA[idx] = someComputationA(dataA[idx]);
    }
}

__global__ void stageB_kernel(float* dataA, float* dataB, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mid = dataA[idx];
        dataB[idx] = someComputationB(mid, dataB[idx]);
    }
}

int main() {
    const int N = 1024 * 1024;
    const int iterations = 100;
    
    std::cout << "Cooperative Persistent Kernel Example (Chapter 10)" << std::endl;
    std::cout << "Array size: " << N << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    
    // Allocate host memory
    std::vector<float> h_dataA(N), h_dataB(N);
    std::vector<float> h_dataA_naive(N), h_dataB_naive(N);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_dataA[i] = static_cast<float>(i % 1000) / 1000.0f;
        h_dataB[i] = static_cast<float>((i + 500) % 1000) / 1000.0f;
        h_dataA_naive[i] = h_dataA[i];
        h_dataB_naive[i] = h_dataB[i];
    }
    
    // Allocate device memory
    float *d_dataA, *d_dataB, *d_dataA_naive, *d_dataB_naive;
    cudaMalloc(&d_dataA, N * sizeof(float));
    cudaMalloc(&d_dataB, N * sizeof(float));
    cudaMalloc(&d_dataA_naive, N * sizeof(float));
    cudaMalloc(&d_dataB_naive, N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_dataA, h_dataA.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, h_dataB.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataA_naive, h_dataA_naive.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB_naive, h_dataB_naive.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Check if cooperative kernels are supported
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (!prop.cooperativeLaunch) {
        std::cout << "Cooperative launch not supported on this device!" << std::endl;
        return 1;
    }
    
    std::cout << "Cooperative launch supported: " << prop.cooperativeLaunch << std::endl;
    std::cout << "Max threads per multi-processor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Multi-processor count: " << prop.multiProcessorCount << std::endl;
    
    // Check if the grid size fits in the device
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, combinedKernel, threads, 0);
    int max_grid_size = max_blocks * prop.multiProcessorCount;
    
    if (blocks > max_grid_size) {
        std::cout << "Grid size too large for cooperative launch!" << std::endl;
        std::cout << "Requested blocks: " << blocks << ", Max blocks: " << max_grid_size << std::endl;
        blocks = max_grid_size;
        std::cout << "Reducing to " << blocks << " blocks" << std::endl;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::cout << "\n" + std::string(50, '=') << std::endl;
    std::cout << "1. Cooperative Persistent Kernel" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Prepare arguments for cooperative kernel
    void* args[] = { &d_dataA, &d_dataB, &N, &iterations };
    
    // Warm up
    cudaLaunchCooperativeKernel(
        (void*)combinedKernel,
        blocks, threads,
        args
    );
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error in cooperative kernel: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Reset data for fair comparison
    cudaMemcpy(d_dataA, h_dataA.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, h_dataB.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Time cooperative kernel
    cudaEventRecord(start);
    cudaLaunchCooperativeKernel(
        (void*)combinedKernel,
        blocks, threads,
        args
    );
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float cooperative_time = 0;
    cudaEventElapsedTime(&cooperative_time, start, stop);
    
    std::cout << "Cooperative persistent kernel time: " << cooperative_time << " ms" << std::endl;
    std::cout << "Total kernel launches: 1" << std::endl;
    
    std::cout << "\n" + std::string(50, '=') << std::endl;
    std::cout << "2. Naive Implementation (Separate Kernels)" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Time naive implementation
    cudaEventRecord(start);
    for (int it = 0; it < iterations; ++it) {
        stageA_kernel<<<blocks, threads>>>(d_dataA_naive, N);
        cudaDeviceSynchronize(); // Explicit sync between stages
        
        stageB_kernel<<<blocks, threads>>>(d_dataA_naive, d_dataB_naive, N);
        cudaDeviceSynchronize(); // Explicit sync before next iteration
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    
    std::cout << "Naive implementation time: " << naive_time << " ms" << std::endl;
    std::cout << "Total kernel launches: " << (2 * iterations) << std::endl;
    std::cout << "Speedup: " << naive_time / cooperative_time << "x" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(h_dataA.data(), d_dataA, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataB.data(), d_dataB, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataA_naive.data(), d_dataA_naive, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dataB_naive.data(), d_dataB_naive, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results match
    bool results_match = true;
    float max_diff = 0.0f;
    
    for (int i = 0; i < N && results_match; i++) {
        float diff_A = std::abs(h_dataA[i] - h_dataA_naive[i]);
        float diff_B = std::abs(h_dataB[i] - h_dataB_naive[i]);
        max_diff = std::max({max_diff, diff_A, diff_B});
        
        if (diff_A > 1e-5 || diff_B > 1e-5) {
            results_match = false;
        }
    }
    
    std::cout << "\n" + std::string(50, '=') << std::endl;
    std::cout << "Verification" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "Results match: " << (results_match ? "YES" : "NO") << std::endl;
    std::cout << "Maximum difference: " << max_diff << std::endl;
    
    std::cout << "\n" + std::string(50, '=') << std::endl;
    std::cout << "Benefits of Cooperative Persistent Kernels" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "Advantages:" << std::endl;
    std::cout << "- Eliminates " << (2 * iterations - 1) << " kernel launch overheads" << std::endl;
    std::cout << "- Data stays in shared memory/registers throughout computation" << std::endl;
    std::cout << "- Uses grid.sync() for cross-block synchronization" << std::endl;
    std::cout << "- Achieves performance closer to peak hardware limits" << std::endl;
    std::cout << "- Reduces host-device synchronization points" << std::endl;
    
    std::cout << "\nTrade-offs:" << std::endl;
    std::cout << "- Reserves entire GPU for this kernel (no concurrent work)" << std::endl;
    std::cout << "- Requires cooperative launch support" << std::endl;
    std::cout << "- Grid size limited by GPU resources" << std::endl;
    
    // Cleanup
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataA_naive);
    cudaFree(d_dataB_naive);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
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
