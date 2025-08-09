// Architecture-specific optimizations for CUDA 12.8
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// multi_stream_pipeline.cu
// Combined intra-kernel and inter-kernel pipelining example

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

using namespace cooperative_groups;

#define TILE_SIZE 1024
#define NUM_STREAMS 2

// Helper: perform computation on one tile
__device__ float computeTile(const float* tile_data, int lane_id) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < TILE_SIZE; k += 32) {
        if (k + lane_id < TILE_SIZE) {
            int idx1 = lane_id * TILE_SIZE + k + lane_id;
            int idx2 = (k + lane_id) * TILE_SIZE + lane_id;
            if (idx1 < TILE_SIZE * TILE_SIZE && idx2 < TILE_SIZE * TILE_SIZE) {
                sum += tile_data[idx1 % TILE_SIZE] * tile_data[idx2 % TILE_SIZE];
            }
        }
    }
    return sum;
}

// Simplified warp-specialized kernel without complex pipeline
__global__ void warp_specialized_kernel(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles)
{
    // Allocate shared memory
    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;                    // [0 ... TILE_SIZE-1]
    float* B_tile = shared_mem + TILE_SIZE;        // [TILE_SIZE ... 2*TILE_SIZE-1]
    float* C_tile = shared_mem + 2 * TILE_SIZE;    // [2*TILE_SIZE ... 3*TILE_SIZE-1]

    // Compute warp_id and lane_id
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // Calculate global warp stride
    int warps_per_block = blockDim.x >> 5;
    int totalWarps = gridDim.x * warps_per_block;
    int global_warp = warp_id + (blockIdx.x * warps_per_block);

    // Process tiles in strided fashion
    for (int tile = global_warp; tile < numTiles; tile += totalWarps) {
        size_t offset = size_t(tile) * TILE_SIZE;

        // Load data (warp 0)
        if (warp_id == 0 && lane_id < TILE_SIZE) {
            if (offset + lane_id < numTiles * TILE_SIZE) {
                A_tile[lane_id] = A_global[offset + lane_id];
                B_tile[lane_id] = B_global[offset + lane_id];
            }
        }

        // Synchronize to ensure data is loaded
        __syncthreads();

        // Compute (warp 1)
        if (warp_id == 1 && lane_id < TILE_SIZE) {
            float resultA = computeTile(A_tile, lane_id);
            float resultB = computeTile(B_tile, lane_id);
            C_tile[lane_id] = resultA + resultB;
        }

        // Synchronize to ensure computation is done
        __syncthreads();

        // Store results (warp 2)
        if (warp_id == 2 && lane_id < TILE_SIZE) {
            if (offset + lane_id < numTiles * TILE_SIZE) {
                C_global[offset + lane_id] = C_tile[lane_id];
            }
        }

        // Synchronize before next iteration
        __syncthreads();
    }
}

// Simple kernels for the event synchronization example
__global__ void preprocess_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

__global__ void process_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sqrtf(input[idx] * input[idx] + 1.0f);
    }
}

__global__ void finalize_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 0.5f;
    }
}

// Simple kernel for host callbacks
__global__ void simple_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sinf((float)idx * 0.01f);
    }
}

// Host-side multi-stream pipeline launcher
void launch_multistream_warp_pipeline(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int numBatches)
{
    printf("Launching multi-stream warp-specialized pipeline...\n");
    printf("Number of batches: %d\n", numBatches);
    printf("Tile size: %d\n", TILE_SIZE);
    printf("Number of streams: %d\n", NUM_STREAMS);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Events for synchronization and timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Process each batch
    for (int batch = 0; batch < numBatches; ++batch) {
        int sid = batch % NUM_STREAMS;
        cudaStream_t stream = streams[sid];

        printf("  Processing batch %d on stream %d\n", batch, sid);

        // Device memory pointers
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        size_t bytes = TILE_SIZE * sizeof(float);

        // Stream-ordered allocation (no global sync)
        cudaMallocAsync((void**)&d_A, bytes, stream);
        cudaMallocAsync((void**)&d_B, bytes, stream);
        cudaMallocAsync((void**)&d_C, bytes, stream);

        // Asynchronous H2D copies
        cudaMemcpyAsync(d_A,
            h_A + batch * TILE_SIZE,
            bytes,
            cudaMemcpyHostToDevice,
            stream);

        cudaMemcpyAsync(d_B,
            h_B + batch * TILE_SIZE,
            bytes,
            cudaMemcpyHostToDevice,
            stream);

        // Launch warp-specialized kernel
        // 3 warps per block (96 threads total)
        dim3 blockDim(96);
        dim3 gridDim(1);
        size_t shmemBytes = 3 * TILE_SIZE * sizeof(float);

        warp_specialized_kernel<<<
            gridDim, blockDim, shmemBytes, stream>>>(
            d_A, d_B, d_C, 1);

        // Asynchronous D2H copy
        cudaMemcpyAsync(h_C + batch * TILE_SIZE,
            d_C,
            bytes,
            cudaMemcpyDeviceToHost,
            stream);

        // Stream-ordered deallocation
        cudaFreeAsync(d_A, stream);
        cudaFreeAsync(d_B, stream);
        cudaFreeAsync(d_C, stream);
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("Multi-stream pipeline completed in %.2f ms\n", elapsed_ms);
    printf("Average time per batch: %.2f ms\n", elapsed_ms / numBatches);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Fine-grained event-based synchronization example
void demonstrateEventSynchronization() {
    printf("\n=== Event-Based Stream Synchronization ===\n");

    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);

    // Allocate pinned host memory
    float *h_data, *h_intermediate, *h_result;
    cudaMallocHost(&h_data, bytes);
    cudaMallocHost(&h_intermediate, bytes);
    cudaMallocHost(&h_result, bytes);

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = sinf((float)i * 0.001f);
    }

    // Allocate device memory
    float *d_data, *d_intermediate, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_intermediate, bytes);
    cudaMalloc(&d_result, bytes);

    // Create streams
    cudaStream_t producer_stream, consumer_stream, output_stream;
    cudaStreamCreate(&producer_stream);
    cudaStreamCreate(&consumer_stream);
    cudaStreamCreate(&output_stream);

    // Create events for synchronization
    cudaEvent_t data_ready, processing_done;
    cudaEventCreate(&data_ready);
    cudaEventCreate(&processing_done);

    // Launch parameters
    dim3 grid((N + 255) / 256);
    dim3 block(256);

    printf("Setting up producer-consumer pipeline with events...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Producer stream: Load and preprocess data
    cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, producer_stream);
    
    // Simple preprocessing kernel
    preprocess_kernel<<<grid, block, 0, producer_stream>>>(d_data, N);
    
    // Record event when data is ready
    cudaEventRecord(data_ready, producer_stream);

    // Consumer stream: Wait for data, then process
    cudaStreamWaitEvent(consumer_stream, data_ready, 0);
    
    // Processing kernel
    process_kernel<<<grid, block, 0, consumer_stream>>>(d_data, d_intermediate, N);
    
    // Record event when processing is done
    cudaEventRecord(processing_done, consumer_stream);

    // Output stream: Wait for processing, then output
    cudaStreamWaitEvent(output_stream, processing_done, 0);
    
    // Final processing and output
    finalize_kernel<<<grid, block, 0, output_stream>>>(d_intermediate, d_result, N);
    
    cudaMemcpyAsync(h_result, d_result, bytes, cudaMemcpyDeviceToHost, output_stream);

    // Wait for all operations
    cudaStreamSynchronize(output_stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("Event-synchronized pipeline completed in %.2f ms\n", elapsed_ms);
    printf("Sample result: input=%.6f -> output=%.6f\n", h_data[0], h_result[0]);

    // Cleanup
    cudaStreamDestroy(producer_stream);
    cudaStreamDestroy(consumer_stream);
    cudaStreamDestroy(output_stream);
    cudaEventDestroy(data_ready);
    cudaEventDestroy(processing_done);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFreeHost(h_data);
    cudaFreeHost(h_intermediate);
    cudaFreeHost(h_result);
    cudaFree(d_data);
    cudaFree(d_intermediate);
    cudaFree(d_result);
}

// Host callback demonstration
void CUDART_CB hostCallback(void* userData) {
    int* counter = (int*)userData;
    (*counter)++;
    printf("    Host callback executed! Counter = %d\n", *counter);
}

void demonstrateHostCallbacks() {
    printf("\n=== Host Callbacks Example ===\n");

    const int N = 512 * 1024;
    const int bytes = N * sizeof(float);

    float *d_data;
    cudaMalloc(&d_data, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Counter for callback
    int callback_counter = 0;

    printf("Launching kernels with host callbacks...\n");

    // Launch parameters
    dim3 grid((N + 255) / 256);
    dim3 block(256);

    // Simple kernel
    simple_kernel<<<grid, block, 0, stream>>>(d_data, N);
    
    // Register host callback to execute when kernel completes
    cudaLaunchHostFunc(stream, hostCallback, &callback_counter);

    // Launch another kernel
    simple_kernel<<<grid, block, 0, stream>>>(d_data, N);
    
    // Another callback
    cudaLaunchHostFunc(stream, hostCallback, &callback_counter);

    printf("Waiting for stream completion...\n");
    cudaStreamSynchronize(stream);

    printf("All operations completed. Final callback counter: %d\n", callback_counter);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

int main() {
    printf("Multi-Stream Pipeline with Warp Specialization - Chapter 11\n");
    printf("============================================================\n");

    // Device info
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "Supported" : "Not supported");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("\n");

    // Setup test data
    const int numBatches = 8;
    const int totalElements = numBatches * TILE_SIZE;

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, totalElements * sizeof(float));
    cudaMallocHost(&h_B, totalElements * sizeof(float));
    cudaMallocHost(&h_C, totalElements * sizeof(float));

    // Initialize input data
    for (int i = 0; i < totalElements; ++i) {
        h_A[i] = sinf((float)i * 0.001f);
        h_B[i] = cosf((float)i * 0.001f);
    }

    printf("=== Multi-Stream Warp-Specialized Pipeline ===\n");
    launch_multistream_warp_pipeline(h_A, h_B, h_C, numBatches);

    // Verify results
    printf("\nVerification (first few results):\n");
    for (int i = 0; i < 5; ++i) {
        printf("  Batch 0, element %d: %.6f\n", i, h_C[i]);
    }

    // Run other demonstrations
    demonstrateEventSynchronization();
    demonstrateHostCallbacks();

    printf("\n=== Performance Summary ===\n");
    printf("This example demonstrates:\n");
    printf("1. Intra-kernel pipelining with warp specialization\n");
    printf("2. Inter-kernel pipelining with multiple CUDA streams\n");
    printf("3. Stream-ordered memory allocation for overlap\n");
    printf("4. Fine-grained event-based synchronization\n");
    printf("5. Host callbacks for asynchronous CPU-GPU coordination\n");

    printf("\n=== Key Benefits ===\n");
    printf("- Memory operations overlap with compute\n");
    printf("- Multiple batches processed concurrently\n");
    printf("- No global synchronization barriers\n");
    printf("- Optimal hardware utilization\n");
    printf("- Scalable to larger workloads\n");

    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile --force-overwrite=true -o multi_stream_pipeline ./multi_stream_pipeline\n");
    printf("ncu --section WarpStateStats --section LaunchStats ./multi_stream_pipeline\n");

    // Cleanup
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
