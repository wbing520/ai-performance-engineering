// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <cstdio>

// We will use two streams to overlap batches, but you can increase this if you have memory to spare.
#define NUM_STREAMS 2
#define TILE_SIZE 1024

// Forward declaration of the device kernel
__global__ void warp_specialized_pipeline_kernel(
    const float* A_global,
    const float* B_global,
    float* C_global,
    int numTiles);

// Launch multiple mini-batches in parallel streams using cudaMallocAsync, cudaMemcpyAsync, and cudaFreeAsync.
void launch_multistream_warp_pipeline(
    const float* h_A, // Host pointer to A (length = numBatches * TILE_SIZE)
    const float* h_B, // Host pointer to B (length = numBatches * TILE_SIZE)
    float* h_C, // Host pointer for output (length = numBatches * TILE_SIZE)
    int numBatches)
{
    // 1) Create multiple CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 2) For each mini-batch, enqueue work into one of the streams
    for (int batch = 0; batch < numBatches; ++batch) {
        int sid = batch % NUM_STREAMS;
        cudaStream_t stream = streams[sid];
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
        size_t bytes = TILE_SIZE * sizeof(float);

        // 2a) Allocate device buffers in a stream-ordered way (no global sync)
        cudaMallocAsync(&d_A, bytes, stream);
        cudaMallocAsync(&d_B, bytes, stream);
        cudaMallocAsync(&d_C, bytes, stream);

        // 2b) Asynchronously copy inputs from host to device in this stream.
        // h_A and h_B must point to pinned (page-locked) memory for true overlap.
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

        // 2c) Launch the warp-specialized pipeline kernel in this stream.
        // We set gridDim=1, blockDim=3 warps × 32 threads = 96 threads. We allocate
        // 3 × TILE_SIZE floats in shared memory for the pipeline's three stages.
        dim3 blockDim(96);
        dim3 gridDim(1);
        size_t shmemBytes = 3 * TILE_SIZE * sizeof(float);
        warp_specialized_pipeline_kernel<<<
            gridDim, blockDim, shmemBytes, stream>>>(
            d_A, d_B, d_C,
            /*numTiles=*/1);

        // 2d) Asynchronously copy the result back from device to host
        cudaMemcpyAsync(h_C + batch * TILE_SIZE,
            d_C,
            bytes,
            cudaMemcpyDeviceToHost,
            stream);

        // 2e) Free device memory in this stream (non-blocking)
        cudaFreeAsync(d_A, stream);
        cudaFreeAsync(d_B, stream);
        cudaFreeAsync(d_C, stream);
    }

    // 3) Synchronize and destroy all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    // Example usage
    int numBatches = 4;
    int totalSize = numBatches * TILE_SIZE;
    
    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    cudaHostAlloc(&h_A, totalSize * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_B, totalSize * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_C, totalSize * sizeof(float), cudaHostAllocDefault);
    
    // Initialize data
    for (int i = 0; i < totalSize; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
    
    // Launch the multi-stream pipeline
    launch_multistream_warp_pipeline(h_A, h_B, h_C, numBatches);
    
    // Print some results
    printf("Result[0]: %f\n", h_C[0]);
    printf("Result[1024]: %f\n", h_C[1024]);
    
    // Cleanup
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    
    return 0;
}
