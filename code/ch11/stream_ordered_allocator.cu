// Architecture-specific optimizations for CUDA 12.9
// Targets Blackwell B200/B300 (sm_100)
// stream_ordered_allocator.cu
// Example demonstrating CUDA stream-ordered memory allocation

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <vector>

__global__ void computeKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        // Simulate some computation
        for (int i = 0; i < 100; ++i) {
            x = x * 0.99f + sinf(x * 0.1f);
        }
        output[idx] = x;
    }
}

__global__ void processKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

void setupStreamOrderedAllocator() {
    printf("=== Setting up Stream-Ordered Memory Allocator ===\n");
    
    // Initialize the async memory allocator
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    
    // Desired number of bytes to keep in pool before
    // releasing back to the OS (tune as needed)
    uint64_t threshold = 64 * 1024 * 1024; // 64 MB
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    
    printf("Memory pool configured with %lu MB threshold\n", threshold / (1024 * 1024));
    
    // Query and display memory pool attributes
    uint64_t reserved, used;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    
    printf("Initial pool state: Reserved=%lu MB, Used=%lu MB\n", 
           reserved / (1024 * 1024), used / (1024 * 1024));
}

void demonstrateStreamOrderedAllocation() {
    printf("\n=== Stream-Ordered Allocation Example ===\n");
    
    const int N = 2 * 1024 * 1024;
    const size_t dataSizeBytes = N * sizeof(float);
    
    // Allocate pinned host memory
    float *h_data1, *h_data2, *h_result1, *h_result2;
    cudaMallocHost(&h_data1, dataSizeBytes);
    cudaMallocHost(&h_data2, dataSizeBytes);
    cudaMallocHost(&h_result1, dataSizeBytes);
    cudaMallocHost(&h_result2, dataSizeBytes);
    
    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_data1[i] = sinf((float)i * 0.001f);
        h_data2[i] = cosf((float)i * 0.001f);
    }
    
    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Allocate memory using stream-ordered async allocation
    void *d_data1, *d_result1;
    void *d_data2, *d_result2;
    
    // Use cudaMallocAsync on specific streams (best practice)
    cudaMallocAsync(&d_data1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_result1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_data2, dataSizeBytes, stream2);
    cudaMallocAsync(&d_result2, dataSizeBytes, stream2);
    
    // Asynchronously copy first chunk and launch its kernel in stream1
    cudaMemcpyAsync(d_data1, h_data1, dataSizeBytes,
                    cudaMemcpyHostToDevice, stream1);
    
    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);
    
    computeKernel<<<gridDim, blockDim, 0, stream1>>>(
        (float*)d_data1, (float*)d_result1, N);
    
    cudaMemcpyAsync(h_result1, d_result1, dataSizeBytes,
                    cudaMemcpyDeviceToHost, stream1);
    
    // In parallel, do the same on stream2
    cudaMemcpyAsync(d_data2, h_data2, dataSizeBytes, 
                    cudaMemcpyHostToDevice, stream2);
    
    computeKernel<<<gridDim, blockDim, 0, stream2>>>(
        (float*)d_data2, (float*)d_result2, N);
    
    cudaMemcpyAsync(h_result2, d_result2, dataSizeBytes,
                    cudaMemcpyDeviceToHost, stream2);
    
    // Wait for both streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    
    printf("Stream-ordered allocation completed in %.2f ms\n", elapsed_ms);
    printf("Stream 1 result[0]: %.6f\n", h_result1[0]);
    printf("Stream 2 result[0]: %.6f\n", h_result2[0]);
    
    // Cleanup using stream-ordered deallocation
    cudaFreeAsync(d_data1, stream1);
    cudaFreeAsync(d_result1, stream1);
    cudaFreeAsync(d_data2, stream2);
    cudaFreeAsync(d_result2, stream2);
    
    // Sync streams before destroying them
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    // Cleanup host memory
    cudaFreeHost(h_data1);
    cudaFreeHost(h_data2);
    cudaFreeHost(h_result1);
    cudaFreeHost(h_result2);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void compareTraditionalVsStreamOrdered() {
    printf("\n=== Traditional vs Stream-Ordered Allocation Comparison ===\n");
    
    const int numIterations = 10;
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    
    float *h_data;
    cudaMallocHost(&h_data, bytes);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i / N;
    }
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    dim3 grid((N + 255) / 256);
    dim3 block(256);
    
    // Test 1: Traditional allocation (synchronous)
    printf("\n1. Traditional synchronous allocation:\n");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < numIterations; ++iter) {
        float *d_data1, *d_data2;
        
        // Synchronous allocation (blocks all streams)
        cudaMalloc(&d_data1, bytes);
        cudaMalloc(&d_data2, bytes);
        
        // Copy and compute
        cudaMemcpyAsync(d_data1, h_data, bytes, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_data2, h_data, bytes, cudaMemcpyHostToDevice, stream2);
        
        processKernel<<<grid, block, 0, stream1>>>(d_data1, N);
        processKernel<<<grid, block, 0, stream2>>>(d_data2, N);
        
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        
        // Synchronous free (blocks all streams)
        cudaFree(d_data1);
        cudaFree(d_data2);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto traditional_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("   Time: %ld ms\n", traditional_duration.count());
    
    // Test 2: Stream-ordered allocation (asynchronous)
    printf("\n2. Stream-ordered asynchronous allocation:\n");
    
    start_time = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < numIterations; ++iter) {
        float *d_data1, *d_data2;
        
        // Asynchronous allocation (stream-ordered)
        cudaMallocAsync((void**)&d_data1, bytes, stream1);
        cudaMallocAsync((void**)&d_data2, bytes, stream2);
        
        // Copy and compute
        cudaMemcpyAsync(d_data1, h_data, bytes, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_data2, h_data, bytes, cudaMemcpyHostToDevice, stream2);
        
        processKernel<<<grid, block, 0, stream1>>>(d_data1, N);
        processKernel<<<grid, block, 0, stream2>>>(d_data2, N);
        
        // Asynchronous free (stream-ordered)
        cudaFreeAsync(d_data1, stream1);
        cudaFreeAsync(d_data2, stream2);
    }
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    end_time = std::chrono::high_resolution_clock::now();
    auto stream_ordered_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("   Time: %ld ms\n", stream_ordered_duration.count());
    printf("   Speedup: %.2fx\n", 
           (float)traditional_duration.count() / stream_ordered_duration.count());
    
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data);
}

void demonstrateLLMWorkflow() {
    printf("\n=== LLM-Style Variable-Length Batch Workflow ===\n");
    
    // Simulate variable-length sequences
    std::vector<int> sequence_lengths = {128, 256, 512, 1024, 256, 128, 768, 384};
    const int num_batches = sequence_lengths.size();
    
    // Create streams for different operations
    cudaStream_t attention_stream, feedforward_stream, copy_stream;
    cudaStreamCreate(&attention_stream);
    cudaStreamCreate(&feedforward_stream);
    cudaStreamCreate(&copy_stream);
    
    printf("Processing %d batches with variable sequence lengths:\n", num_batches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int batch = 0; batch < num_batches; ++batch) {
        int seq_len = sequence_lengths[batch];
        size_t attention_scratch_bytes = seq_len * seq_len * sizeof(float); // O(N^2) attention
        size_t feedforward_scratch_bytes = seq_len * 4096 * sizeof(float);  // FFN intermediate
        
        printf("  Batch %d: seq_len=%d, attention_scratch=%.1f MB, ffn_scratch=%.1f MB\n",
               batch, seq_len, 
               attention_scratch_bytes / (1024.0f * 1024.0f),
               feedforward_scratch_bytes / (1024.0f * 1024.0f));
        
        // Allocate variable-sized scratch buffers using stream-ordered allocation
        float *attention_scratch, *ffn_scratch;
        cudaMallocAsync((void**)&attention_scratch, attention_scratch_bytes, attention_stream);
        cudaMallocAsync((void**)&ffn_scratch, feedforward_scratch_bytes, feedforward_stream);
        
        // Simulate attention computation
        dim3 attention_grid((seq_len * seq_len + 255) / 256);
        dim3 block(256);
        
        if (seq_len * seq_len > 0) {
            processKernel<<<attention_grid, block, 0, attention_stream>>>(
                attention_scratch, seq_len * seq_len);
        }
        
        // Simulate feedforward computation (can overlap with attention)
        dim3 ffn_grid((seq_len * 4096 + 255) / 256);
        if (seq_len * 4096 > 0) {
            processKernel<<<ffn_grid, block, 0, feedforward_stream>>>(
                ffn_scratch, seq_len * 4096);
        }
        
        // Free scratch memory when done (stream-ordered)
        cudaFreeAsync(attention_scratch, attention_stream);
        cudaFreeAsync(ffn_scratch, feedforward_stream);
        
        // Simulate periodic host-device communication
        if (batch % 3 == 0) {
            float *host_buffer;
            cudaMallocHost(&host_buffer, seq_len * sizeof(float));
            
            float *device_buffer;
            cudaMallocAsync((void**)&device_buffer, seq_len * sizeof(float), copy_stream);
            
            // Simulate copying some results back to host
            cudaMemcpyAsync(host_buffer, device_buffer, seq_len * sizeof(float),
                           cudaMemcpyDeviceToHost, copy_stream);
            
            // Schedule cleanup
            cudaFreeAsync(device_buffer, copy_stream);
            
            // Note: In real code, you'd need to ensure host_buffer is freed after the copy completes
            cudaStreamSynchronize(copy_stream);
            cudaFreeHost(host_buffer);
        }
    }
    
    // Wait for all operations to complete
    cudaStreamSynchronize(attention_stream);
    cudaStreamSynchronize(feedforward_stream);
    cudaStreamSynchronize(copy_stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("LLM workflow completed in %ld ms\n", duration.count());
    
    // Query final memory pool state
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    
    uint64_t reserved, used;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    
    printf("Final pool state: Reserved=%lu MB, Used=%lu MB\n", 
           reserved / (1024 * 1024), used / (1024 * 1024));
    
    // Cleanup
    cudaStreamDestroy(attention_stream);
    cudaStreamDestroy(feedforward_stream);
    cudaStreamDestroy(copy_stream);
}

int main() {
    printf("Stream-Ordered Memory Allocator Examples - Chapter 11\n");
    printf("====================================================\n");
    
    // Check CUDA version and device capabilities
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Runtime Version: %d.%d\n", 
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Memory Pool Support: %s\n", 
           prop.memoryPoolsSupported ? "Yes" : "No");
    
    if (!prop.memoryPoolsSupported) {
        printf("Warning: Memory pools not supported on this device\n");
        printf("Stream-ordered allocation will fall back to synchronous allocation\n");
    }
    
    printf("\n");
    
    // Run examples
    setupStreamOrderedAllocator();
    demonstrateStreamOrderedAllocation();
    compareTraditionalVsStreamOrdered();
    demonstrateLLMWorkflow();
    
    printf("\n=== Key Benefits of Stream-Ordered Allocation ===\n");
    printf("1. No global device synchronization on allocation/free\n");
    printf("2. Perfect for variable-length sequences (LLMs)\n");
    printf("3. Enables true overlap of memory management and compute\n");
    printf("4. Memory pool reduces OS allocation overhead\n");
    printf("5. Essential for multi-stream pipelines\n");
    
    printf("\n=== Usage in PyTorch ===\n");
    printf("export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync\n");
    printf("This enables stream-ordered allocation in PyTorch automatically\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile --force-overwrite=true -o stream_alloc ./stream_ordered_allocator\n");
    printf("ncu --section MemoryWorkloadAnalysis ./stream_ordered_allocator\n");
    
    return 0;
}
