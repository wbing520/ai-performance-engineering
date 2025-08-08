// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// basic_streams.cu
// Basic CUDA streams example showing kernel and copy overlap

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Simple kernels for demonstration
__global__ void ker_A(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 1000; ++i) {
            data[idx] = data[idx] * 1.001f + 0.001f;
        }
    }
}

__global__ void ker_B(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 800; ++i) {
            data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
        }
    }
}

__global__ void ker_1(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 600; ++i) {
            data[idx] = data[idx] + sinf(data[idx]);
        }
    }
}

__global__ void ker_2(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 400; ++i) {
            data[idx] = cosf(data[idx]) * 0.99f;
        }
    }
}

__global__ void ker_3(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 500; ++i) {
            data[idx] = expf(-data[idx] * 0.01f);
        }
    }
}

void cpu_code_1() {
    // Simulate some CPU work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
}

void cpu_code_2() {
    // Simulate more CPU work
    volatile float result = 1.0f;
    for (int i = 0; i < 500000; ++i) {
        result *= 1.0001f;
    }
}

void demonstrateBasicStreams() {
    printf("=== Basic CUDA Streams Example ===\n");
    
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate host memory (pinned for async transfers)
    float *h_data1, *h_data2;
    cudaMallocHost(&h_data1, bytes);
    cudaMallocHost(&h_data2, bytes);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data1[i] = (float)i / N;
        h_data2[i] = (float)(i % 100) / 100.0f;
    }
    
    // Allocate device memory
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);
    
    // 1) Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // 2) Define grid/block sizes
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Launching kernels on multiple streams...\n");
    
    cudaEventRecord(start);
    
    // 3) Launch ker_1 on stream1
    ker_1<<<grid, block, 0, stream1>>>(d_data1, N);
    
    // 4) CPU code 1 runs immediately (asynchronously wrt GPU)
    printf("CPU code 1 executing\n");
    cpu_code_1();
    
    // 5) Launch ker_A on stream2
    ker_A<<<grid, block, 0, stream2>>>(d_data2, N);
    
    // 6) Launch ker_B on stream1
    ker_B<<<grid, block, 0, stream1>>>(d_data1, N);
    
    // 7) Launch ker_2 on stream2
    ker_2<<<grid, block, 0, stream2>>>(d_data2, N);
    
    // 8) CPU code 2 runs immediately
    printf("CPU code 2 executing\n");
    cpu_code_2();
    
    // 9) Launch ker_3 on stream1
    ker_3<<<grid, block, 0, stream1>>>(d_data1, N);
    
    // 10) Wait for work on each stream to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    printf("Total execution time: %.2f ms\n", elapsed_ms);
    
    // Copy results back to verify
    cudaMemcpy(h_data1, d_data1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data2, d_data2, bytes, cudaMemcpyDeviceToHost);
    
    printf("Stream 1 result[0]: %.6f\n", h_data1[0]);
    printf("Stream 2 result[0]: %.6f\n", h_data2[0]);
    
    // 11) Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data1);
    cudaFreeHost(h_data2);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrateComputeDataOverlap() {
    printf("\n=== Compute and Data Transfer Overlap ===\n");
    
    const int N = 2 * 1024 * 1024;
    const int bytes = N * sizeof(float);
    const int numBatches = 4;
    const int batchSize = N / numBatches;
    const int batchBytes = batchSize * sizeof(float);
    
    // Allocate pinned host memory for async transfers
    float *h_input, *h_output;
    cudaMallocHost(&h_input, bytes);
    cudaMallocHost(&h_output, bytes);
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sinf((float)i * 0.01f);
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // Create streams for overlapping operations
    cudaStream_t computeStream, copyH2DStream, copyD2HStream;
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&copyH2DStream);
    cudaStreamCreate(&copyD2HStream);
    
    // Launch parameters
    dim3 grid((batchSize + 255) / 256);
    dim3 block(256);
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    printf("Processing %d batches with 3-way overlap...\n", numBatches);
    
    for (int batch = 0; batch < numBatches; ++batch) {
        int offset = batch * batchSize;
        
        // H2D copy for current batch
        cudaMemcpyAsync(d_input + offset, 
                       h_input + offset, 
                       batchBytes,
                       cudaMemcpyHostToDevice, 
                       copyH2DStream);
        
        // Compute on previous batch (if exists)
        if (batch > 0) {
            int prevOffset = (batch - 1) * batchSize;
            ker_A<<<grid, block, 0, computeStream>>>(d_input + prevOffset, batchSize);
        }
        
        // D2H copy for batch before previous (if exists)
        if (batch > 1) {
            int prevPrevOffset = (batch - 2) * batchSize;
            cudaMemcpyAsync(h_output + prevPrevOffset,
                           d_output + prevPrevOffset,
                           batchBytes,
                           cudaMemcpyDeviceToHost,
                           copyD2HStream);
        }
    }
    
    // Process remaining batches
    // Compute on last batch
    int lastOffset = (numBatches - 1) * batchSize;
    ker_A<<<grid, block, 0, computeStream>>>(d_input + lastOffset, batchSize);
    
    // Copy out last two batches
    int secondLastOffset = (numBatches - 2) * batchSize;
    cudaMemcpyAsync(h_output + secondLastOffset,
                   d_output + secondLastOffset,
                   batchBytes,
                   cudaMemcpyDeviceToHost,
                   copyD2HStream);
    
    cudaMemcpyAsync(h_output + lastOffset,
                   d_output + lastOffset,
                   batchBytes,
                   cudaMemcpyDeviceToHost,
                   copyD2HStream);
    
    // Wait for all operations to complete
    cudaStreamSynchronize(computeStream);
    cudaStreamSynchronize(copyH2DStream);
    cudaStreamSynchronize(copyD2HStream);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("3-way overlap completed in %ld ms\n", duration.count());
    printf("Output sample: %.6f\n", h_output[batchSize]);
    
    // Cleanup
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(copyH2DStream);
    cudaStreamDestroy(copyD2HStream);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

void demonstrateDefaultStreamPitfalls() {
    printf("\n=== Default Stream Synchronization Issues ===\n");
    
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);
    
    // Create explicit streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Testing default stream vs explicit streams...\n");
    
    // Test 1: Using default stream (bad for concurrency)
    printf("\n1. Using default stream (serialized):\n");
    cudaEventRecord(start);
    
    ker_A<<<grid, block>>>(d_data1, N);  // Default stream
    ker_B<<<grid, block>>>(d_data2, N);  // Default stream - waits for ker_A
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float default_time;
    cudaEventElapsedTime(&default_time, start, stop);
    printf("   Time: %.2f ms (serialized execution)\n", default_time);
    
    // Test 2: Using explicit streams (good for concurrency)
    printf("\n2. Using explicit streams (parallel):\n");
    cudaEventRecord(start);
    
    ker_A<<<grid, block, 0, stream1>>>(d_data1, N);  // stream1
    ker_B<<<grid, block, 0, stream2>>>(d_data2, N);  // stream2 - can overlap
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float explicit_time;
    cudaEventElapsedTime(&explicit_time, start, stop);
    printf("   Time: %.2f ms (parallel execution)\n", explicit_time);
    printf("   Speedup: %.2fx\n", default_time / explicit_time);
    
    // Test 3: Mixed default and explicit (bad - creates barriers)
    printf("\n3. Mixed default and explicit streams (bad):\n");
    cudaEventRecord(start);
    
    ker_A<<<grid, block, 0, stream1>>>(d_data1, N);  // stream1
    ker_B<<<grid, block>>>(d_data2, N);              // default stream - forces sync
    ker_1<<<grid, block, 0, stream2>>>(d_data1, N);  // stream2 - waits for default
    
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float mixed_time;
    cudaEventElapsedTime(&mixed_time, start, stop);
    printf("   Time: %.2f ms (forced synchronization)\n", mixed_time);
    
    printf("\nKey takeaway: Avoid default stream for performance-critical code!\n");
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Streams Examples - Chapter 11\n");
    printf("====================================\n");
    
    // Get device info
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "Supported" : "Not supported");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("\n");
    
    // Run demonstrations
    demonstrateBasicStreams();
    demonstrateComputeDataOverlap();
    demonstrateDefaultStreamPitfalls();
    
    printf("\n=== Best Practices Summary ===\n");
    printf("1. Use explicit streams for performance-critical code\n");
    printf("2. Enable per-thread default streams (PTDS) when using multiple CPU threads\n");
    printf("3. Use pinned memory for async transfers\n");
    printf("4. Avoid mixing default and explicit streams\n");
    printf("5. Use stream-ordered memory allocator (cudaMallocAsync)\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile --force-overwrite=true -o streams_analysis ./basic_streams\n");
    printf("ncu --section LaunchStats --section MemoryWorkloadAnalysis ./basic_streams\n");
    
    return 0;
}
