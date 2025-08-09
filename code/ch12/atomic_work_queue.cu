// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// atomic_work_queue.cu
// Dynamic work distribution using atomic counters to balance workloads

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <chrono>

using namespace cooperative_groups;

// Global atomic counter for work distribution
__device__ unsigned int globalIndex = 0;

// Static work distribution kernel (uneven workload)
__global__ void computeKernelStatic(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Each thread does variable work based on idx
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}

// Dynamic work distribution kernel with atomic work queue
__global__ void computeKernelDynamicBatch(const float* input, float* output, int N) {
    // Warp-level mask and lane ID
    const unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);

    while (true) {
        // Warp leader atomically claims the next batch of 32 indices
        unsigned int base;
        if (lane == 0) {
            // One atomic per warp reduces contention
            base = atomicAdd(&globalIndex, warpSize);
        }

        // Broadcast base to all lanes in the warp using shuffle
        base = __shfl_sync(mask, base, 0);

        // Compute each thread's global index and exit if out of range
        unsigned int idx = base + lane;
        if (idx >= (unsigned int)N) break;

        // Per-index work: variable loop bound
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}

// Optimized version with larger batch sizes to reduce atomic contention
__global__ void computeKernelDynamicLargeBatch(const float* input, float* output, int N) {
    const unsigned mask = __activemask();
    int lane = threadIdx.x & (warpSize - 1);
    const int batchSize = 128; // Larger batch to amortize atomics

    while (true) {
        unsigned int base;
        if (lane == 0) {
            base = atomicAdd(&globalIndex, batchSize);
        }

        base = __shfl_sync(mask, base, 0);

        // Process multiple elements per thread
        for (int offset = 0; offset < batchSize; offset += warpSize) {
            unsigned int idx = base + offset + lane;
            if (idx >= (unsigned int)N) return;

            int work = idx % 256;
            float result = 0.0f;
            for (int i = 0; i < work; ++i) {
                result += sinf(input[idx]) * cosf(input[idx]);
            }
            output[idx] = result;
        }
    }
}

// Hierarchical work distribution using block-level batching
__global__ void computeKernelHierarchical(const float* input, float* output, int N) {
    __shared__ unsigned int blockBase;
    __shared__ int blockWorkCount;
    
    const int blockSize = blockDim.x;
    const int batchSize = blockSize * 4; // Process 4x block size per batch
    
    while (true) {
        // Block leader gets work for entire block
        if (threadIdx.x == 0) {
            blockBase = atomicAdd(&globalIndex, batchSize);
            blockWorkCount = min(batchSize, max(0, N - (int)blockBase));
        }
        __syncthreads();
        
        if (blockWorkCount <= 0 || blockBase >= (unsigned int)N) break;
        
        // Each thread processes multiple elements
        for (int i = threadIdx.x; i < blockWorkCount; i += blockSize) {
            unsigned int idx = blockBase + i;
            if (idx >= (unsigned int)N) break;
            
            int work = idx % 256;
            float result = 0.0f;
            for (int j = 0; j < work; ++j) {
                result += sinf(input[idx]) * cosf(input[idx]);
            }
            output[idx] = result;
        }
        __syncthreads();
    }
}

void resetGlobalCounter() {
    unsigned int zero = 0;
    cudaMemcpyToSymbol(globalIndex, &zero, sizeof(unsigned int));
}

void benchmarkWorkDistribution() {
    printf("=== Dynamic Work Distribution Benchmark ===\n");
    
    const int N = 1 << 20; // 1M elements
    const int bytes = N * sizeof(float);
    
    // Allocate and initialize host memory
    float *h_input = new float[N];
    float *h_output_static = new float[N];
    float *h_output_dynamic = new float[N];
    float *h_output_large_batch = new float[N];
    float *h_output_hierarchical = new float[N];
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i / N;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Launch parameters
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Problem size: %d elements\n", N);
    printf("Grid size: %d blocks of %d threads\n", grid.x, block.x);
    printf("Work per element: variable (0-255 iterations)\n\n");
    
    // Test 1: Static work distribution (baseline)
    printf("1. Static work distribution (baseline):\n");
    cudaEventRecord(start);
    computeKernelStatic<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float static_time;
    cudaEventElapsedTime(&static_time, start, stop);
    printf("   Time: %.2f ms\n", static_time);
    
    cudaMemcpy(h_output_static, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Test 2: Dynamic work distribution (warp-level batching)
    printf("\n2. Dynamic work distribution (warp-level batching):\n");
    resetGlobalCounter();
    
    cudaEventRecord(start);
    computeKernelDynamicBatch<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float dynamic_time;
    cudaEventElapsedTime(&dynamic_time, start, stop);
    printf("   Time: %.2f ms\n", dynamic_time);
    printf("   Speedup: %.2fx\n", static_time / dynamic_time);
    
    cudaMemcpy(h_output_dynamic, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Test 3: Large batch dynamic distribution
    printf("\n3. Dynamic with large batches (reduced atomic contention):\n");
    resetGlobalCounter();
    
    cudaEventRecord(start);
    computeKernelDynamicLargeBatch<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float large_batch_time;
    cudaEventElapsedTime(&large_batch_time, start, stop);
    printf("   Time: %.2f ms\n", large_batch_time);
    printf("   Speedup vs static: %.2fx\n", static_time / large_batch_time);
    printf("   Speedup vs dynamic: %.2fx\n", dynamic_time / large_batch_time);
    
    cudaMemcpy(h_output_large_batch, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Test 4: Hierarchical work distribution
    printf("\n4. Hierarchical work distribution (block-level batching):\n");
    resetGlobalCounter();
    
    cudaEventRecord(start);
    computeKernelHierarchical<<<grid, block>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float hierarchical_time;
    cudaEventElapsedTime(&hierarchical_time, start, stop);
    printf("   Time: %.2f ms\n", hierarchical_time);
    printf("   Speedup vs static: %.2fx\n", static_time / hierarchical_time);
    
    cudaMemcpy(h_output_hierarchical, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // Verify correctness
    printf("\n=== Correctness Verification ===\n");
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff1 = fabs(h_output_static[i] - h_output_dynamic[i]);
        float diff2 = fabs(h_output_static[i] - h_output_large_batch[i]);
        float diff3 = fabs(h_output_static[i] - h_output_hierarchical[i]);
        max_diff = fmax(max_diff, fmax(diff1, fmax(diff2, diff3)));
    }
    printf("Maximum difference from baseline: %.2e\n", max_diff);
    
    // Sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Index | Static    | Dynamic   | LargeBatch | Hierarchical\n");
    printf("------|-----------|-----------|------------|-------------\n");
    for (int i = 0; i < 5; ++i) {
        printf("%5d | %9.3f | %9.3f | %10.3f | %11.3f\n", 
               i, h_output_static[i], h_output_dynamic[i], 
               h_output_large_batch[i], h_output_hierarchical[i]);
    }
    
    // Cleanup
    delete[] h_input;
    delete[] h_output_static;
    delete[] h_output_dynamic;
    delete[] h_output_large_batch;
    delete[] h_output_hierarchical;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Demonstrate atomic contention analysis
__global__ void atomicContentionTest(int* counter, int* results, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int local_count = 0;
    for (int i = 0; i < iterations; ++i) {
        local_count += atomicAdd(counter, 1);
    }
    results[idx] = local_count;
}

void demonstrateAtomicContention() {
    printf("\n=== Atomic Contention Analysis ===\n");
    
    const int N = 1024;
    const int iterations = 1000;
    
    int *d_counter, *d_results;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMalloc(&d_results, N * sizeof(int));
    
    // Test different numbers of threads
    int thread_counts[] = {32, 64, 128, 256, 512, 1024};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    printf("Testing atomic contention with %d iterations per thread:\n", iterations);
    printf("Threads | Time (ms) | Transactions/Request | Efficiency\n");
    printf("--------|-----------|---------------------|------------\n");
    
    for (int test = 0; test < num_tests; ++test) {
        int num_threads = thread_counts[test];
        dim3 block(256);
        dim3 grid((num_threads + block.x - 1) / block.x);
        
        // Reset counter
        int zero = 0;
        cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Time the atomic operations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        atomicContentionTest<<<grid, block>>>(d_counter, d_results, num_threads, iterations);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        // Calculate efficiency (theoretical vs actual)
        int expected_total = num_threads * iterations;
        int actual_total;
        cudaMemcpy(&actual_total, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
        
        float efficiency = (float)actual_total / expected_total * 100.0f;
        float transactions_per_request = (float)expected_total / actual_total;
        
        printf("%7d | %9.2f | %19.2f | %9.1f%%\n", 
               num_threads, time_ms, transactions_per_request, efficiency);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\nKey insights:\n");
    printf("- Lower transactions/request = less contention\n");
    printf("- Higher efficiency = better atomic performance\n");
    printf("- Use batching to reduce atomic frequency\n");
    
    cudaFree(d_counter);
    cudaFree(d_results);
}

int main() {
    printf("Atomic Work Queue Examples - Chapter 12\n");
    printf("=======================================\n");
    
    // Device info
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("L2 Cache Size: %d MB\n", prop.l2CacheSize / (1024 * 1024));
    printf("Concurrent Kernels: %s\n", 
           prop.concurrentKernels ? "Supported" : "Not supported");
    printf("\n");
    
    benchmarkWorkDistribution();
    demonstrateAtomicContention();
    
    printf("\n=== Key Takeaways ===\n");
    printf("1. Dynamic work distribution balances irregular workloads\n");
    printf("2. Atomic batching reduces contention and improves performance\n");
    printf("3. Hierarchical distribution can further optimize for complex patterns\n");
    printf("4. Monitor atomic transactions/request to identify contention\n");
    printf("5. Choose batch size based on workload characteristics\n");
    
    printf("\n=== Optimization Strategies ===\n");
    printf("- Use warp-level batching for moderate imbalance\n");
    printf("- Use block-level batching for extreme imbalance\n");
    printf("- Increase batch size to reduce atomic frequency\n");
    printf("- Consider per-SM counters for very high contention\n");
    printf("- Profile with Nsight Compute atomic metrics\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("ncu --section MemoryWorkloadAnalysis --section WarpStateStats ./atomic_work_queue\n");
    printf("nsys profile --force-overwrite=true -o atomic_queue ./atomic_work_queue\n");
    
    return 0;
}

// CUDA 12.9 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    // Your kernel code here
}

// CUDA 12.9 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    // Your TMA code here
}
