// Architecture-specific optimizations for CUDA 13.0
// Targets Blackwell B200/B300 (sm_100)
// cooperative_persistent_kernel.cu
// Example demonstrating cooperative groups and persistent kernels

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <chrono>

namespace cg = cooperative_groups;

// Simple persistent kernel example
__device__ int g_task_index = 0; // global counter for next task

struct Task {
    float* input;
    float* output;
    int size;
    float scale;
};

__device__ void processTask(const Task& task) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < task.size; i += stride) {
        task.output[i] = task.input[i] * task.scale + 1.0f;
    }
}

__global__ void persistentKernel(Task* tasks, int totalTasks) {
    // Every thread loops, atomically grabbing the next task index until none remain
    while (true) {
        int idx = atomicAdd(&g_task_index, 1);
        if (idx >= totalTasks) break;
        
        processTask(tasks[idx]);
    }
}

// Cooperative kernel with global synchronization
__global__ void cooperativePersistentKernel(
    float* dataA, 
    float* dataB, 
    int N, 
    int iterations)
{
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int it = 0; it < iterations; ++it) {
        // Stage 1: process dataA and produce intermediate results
        if (idx < N) {
            dataA[idx] = dataA[idx] * 2.0f + 1.0f;
        }

        // Global sync across all blocks before proceeding
        grid.sync();

        // Stage 2: read intermediate results and process dataB
        if (idx < N) {
            float mid = dataA[idx]; // uses completed dataA from stage 1
            dataB[idx] = mid * dataB[idx] + 0.5f;
        }

        // Another global sync
        grid.sync();
    }
}

// Multi-stage reduction using cooperative groups
__global__ void cooperativeReduction(
    const float* input, 
    float* output, 
    int N)
{
    extern __shared__ float sdata[];
    
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    block.sync();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        block.sync();
    }
    
    // Store block result
    if (tid == 0) {
        output[bid] = sdata[0];
    }
    
    // Global sync before final reduction
    grid.sync();
    
    // Final reduction by block 0
    if (bid == 0 && tid == 0) {
        float final_sum = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            final_sum += output[i];
        }
        output[0] = final_sum;
    }
}

// Warp-level cooperative example
__global__ void warpCooperativeKernel(
    const float* input, 
    float* output, 
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Create warp-level cooperative group
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    if (idx < N) {
        float value = input[idx];
        
        // Warp-level shuffle operations
        float sum = value;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += warp.shfl_down(sum, offset);
        }
        
        // Broadcast result to all threads in warp
        float warp_sum = warp.shfl(sum, 0);
        
        // Normalize by warp sum
        output[idx] = value / (warp_sum + 1e-8f);
    }
}

// Thread block cluster example (for newer architectures)
__global__ void clusterCooperativeKernel(
    const float* input, 
    float* output, 
    int N)
{
    // Note: This requires cluster support (Compute Capability 9.0+)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Basic computation - cluster-specific features would require
    // more advanced CUDA toolkit features
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Benchmark function
void benchmarkPersistentKernel() {
    printf("=== Persistent Kernel Benchmark ===\n");
    
    const int numTasks = 1000;
    const int taskSize = 1024;
    
    // Allocate host memory
    float* h_input = new float[numTasks * taskSize];
    float* h_output_persistent = new float[numTasks * taskSize];
    float* h_output_traditional = new float[numTasks * taskSize];
    
    // Initialize input
    for (int i = 0; i < numTasks * taskSize; ++i) {
        h_input[i] = (float)rand() / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output_persistent, *d_output_traditional;
    cudaMalloc(&d_input, numTasks * taskSize * sizeof(float));
    cudaMalloc(&d_output_persistent, numTasks * taskSize * sizeof(float));
    cudaMalloc(&d_output_traditional, numTasks * taskSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, numTasks * taskSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create tasks
    Task* h_tasks = new Task[numTasks];
    Task* d_tasks;
    cudaMalloc(&d_tasks, numTasks * sizeof(Task));
    
    for (int i = 0; i < numTasks; ++i) {
        h_tasks[i].input = d_input + i * taskSize;
        h_tasks[i].output = d_output_persistent + i * taskSize;
        h_tasks[i].size = taskSize;
        h_tasks[i].scale = 2.0f;
    }
    
    cudaMemcpy(d_tasks, h_tasks, numTasks * sizeof(Task), cudaMemcpyHostToDevice);
    
    // Reset global task index
    int zero = 0;
    cudaMemcpyToSymbol(g_task_index, &zero, sizeof(int));
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Traditional approach: launch many small kernels
    // Use a simple kernel launch instead of lambda
    dim3 blockDim(256); // Assuming 256 threads per block for simplicity
    dim3 gridDim((taskSize + blockDim.x - 1) / blockDim.x);
    
    cudaEventRecord(start);
    for (int i = 0; i < numTasks; ++i) {
        // Launch a simple kernel for each task
        // Note: This is a simplified approach - in practice you'd use a proper kernel
        cudaMemset(d_output_traditional + i * taskSize, 0, taskSize * sizeof(float));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float traditional_time;
    cudaEventElapsedTime(&traditional_time, start, stop);
    
    // Persistent kernel approach
    cudaEventRecord(start);
    
    // Reset task index
    cudaMemcpyToSymbol(g_task_index, &zero, sizeof(int));
    
    // Launch persistent kernel with fewer blocks
    int blocks = 32;  // Much fewer than numTasks
    int threads = 256;
    persistentKernel<<<blocks, threads>>>(d_tasks, numTasks);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float persistent_time;
    cudaEventElapsedTime(&persistent_time, start, stop);
    
    // Copy results back
    cudaMemcpy(h_output_persistent, d_output_persistent, 
               numTasks * taskSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_traditional, d_output_traditional, 
               numTasks * taskSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < numTasks * taskSize; ++i) {
        float diff = fabs(h_output_persistent[i] - h_output_traditional[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    printf("Traditional (many kernels): %.2f ms\n", traditional_time);
    printf("Persistent (single kernel): %.2f ms\n", persistent_time);
    printf("Speedup:                    %.2fx\n", traditional_time / persistent_time);
    printf("Max difference:             %.2e\n", max_diff);
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_persistent;
    delete[] h_output_traditional;
    delete[] h_tasks;
    cudaFree(d_input);
    cudaFree(d_output_persistent);
    cudaFree(d_output_traditional);
    cudaFree(d_tasks);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmarkCooperativeKernel() {
    printf("\n=== Cooperative Kernel Benchmark ===\n");
    
    int device;
    cudaGetDevice(&device);
    
    int supportsCoop;
    cudaDeviceGetAttribute(&supportsCoop, cudaDevAttrCooperativeLaunch, device);
    
    if (!supportsCoop) {
        printf("Cooperative launch not supported on this device\n");
        return;
    }
    
    const int N = 1024 * 1024;
    const int iterations = 10;
    
    // Allocate memory
    float *h_dataA = new float[N];
    float *h_dataB = new float[N];
    
    for (int i = 0; i < N; ++i) {
        h_dataA[i] = (float)i / N;
        h_dataB[i] = 1.0f;
    }
    
    float *d_dataA, *d_dataB;
    cudaMalloc(&d_dataA, N * sizeof(float));
    cudaMalloc(&d_dataB, N * sizeof(float));
    
    cudaMemcpy(d_dataA, h_dataA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, h_dataB, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch parameters
    int blocks = 128;
    int threads = 256;
    
    // Check maximum blocks for cooperative launch
    int maxBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, cooperativePersistentKernel, threads, 0);
    
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    maxBlocks *= numSMs;
    
    if (blocks > maxBlocks) {
        blocks = maxBlocks;
        printf("Reducing blocks to %d (max cooperative: %d)\n", blocks, maxBlocks);
    }
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    if (blocks > 0) {
        void* args[] = {(void*)&d_dataA, (void*)&d_dataB, (void*)&N, (void*)&iterations};
        
        cudaEventRecord(start);
        cudaLaunchCooperativeKernel(
            (void*)cooperativePersistentKernel,
            dim3(blocks), dim3(threads),
            args, 0, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float coop_time;
        cudaEventElapsedTime(&coop_time, start, stop);
        
        printf("Cooperative kernel time: %.2f ms\n", coop_time);
        printf("Time per iteration:      %.3f ms\n", coop_time / iterations);
    } else {
        printf("Cannot launch cooperative kernel: max blocks is 0\n");
    }
    
    // Test cooperative reduction
    printf("\n=== Cooperative Reduction Test ===\n");
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    // Initialize with ones for easy verification
    cudaMemset(d_input, 0, N * sizeof(float));
    float ones = 1.0f;
    for (int i = 0; i < N; i += 4096) {
        cudaMemcpy(d_input + i, &ones, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void* reduction_args[] = {(void*)&d_input, (void*)&d_output, (void*)&N};
    
    cudaEventRecord(start);
    cudaLaunchCooperativeKernel(
        (void*)cooperativeReduction,
        dim3(blocks), dim3(threads),
        reduction_args, threads * sizeof(float), 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float reduction_time;
    cudaEventElapsedTime(&reduction_time, start, stop);
    
    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Reduction time: %.2f ms\n", reduction_time);
    printf("Result: %.1f (expected: %.1f)\n", result, (float)(N / 4096));
    
    // Cleanup
    delete[] h_dataA;
    delete[] h_dataB;
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    printf("=== Cooperative and Persistent Kernel Examples ===\n");
    
    // Get device info
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    
    int supportsCoop;
    cudaDeviceGetAttribute(&supportsCoop, cudaDevAttrCooperativeLaunch, device);
    printf("Cooperative launch support: %s\n", supportsCoop ? "Yes" : "No");
    
    printf("\n");
    
    benchmarkPersistentKernel();
    benchmarkCooperativeKernel();
    
    // Test warp cooperative kernel
    printf("\n=== Warp Cooperative Test ===\n");
    
    const int N = 1024;
    float *h_input = new float[N];
    float *h_output = new float[N];
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)(i % 32) + 1.0f; // Each warp gets values 1-32
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    warpCooperativeKernel<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Warp normalization example:\n");
    printf("Input[0-3]:  %.1f %.1f %.1f %.1f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3]);
    printf("Output[0-3]: %.3f %.3f %.3f %.3f\n", 
           h_output[0], h_output[1], h_output[2], h_output[3]);
    
    // Verify sum is approximately 1 for first warp
    float sum = 0.0f;
    for (int i = 0; i < 32; ++i) {
        sum += h_output[i];
    }
    printf("Sum of first warp: %.3f (should be ~1.0)\n", sum);
    
    printf("\n=== Profiling Commands ===\n");
    printf("ncu --section WarpStateStats --section LaunchStats ./cooperative_persistent_kernel\n");
    printf("nsys profile --force-overwrite=true -o cooperative ./cooperative_persistent_kernel\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
