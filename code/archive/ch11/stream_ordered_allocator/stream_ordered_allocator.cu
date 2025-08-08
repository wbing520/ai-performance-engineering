// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <cstdio>

// Example compute kernel
__global__ void computeKernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f; // Simple computation
    }
}

int main() {
    // Initialize the async memory allocator
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    
    // Desired number of bytes to keep in pool before
    // releasing back to the OS (tune as needed)
    size_t threshold = 1024 * 1024 * 1024; // 1GB
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate memory using stream-ordered async allocation
    void *d_data1, *d_result1;
    void *d_data2, *d_result2;
    size_t dataSizeBytes = 1024 * sizeof(float);

    // Use cudaMallocAsync on a given stream (best practice in modern multi-stream apps)
    cudaMallocAsync(&d_data1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_result1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_data2, dataSizeBytes, stream2);
    cudaMallocAsync(&d_result2, dataSizeBytes, stream2);

    // Allocate pinned host memory for async transfers
    float *h_data1, *h_data2, *h_result1, *h_result2;
    cudaHostAlloc(&h_data1, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_data2, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_result1, dataSizeBytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_result2, dataSizeBytes, cudaHostAllocDefault);

    // Initialize host data
    for (int i = 0; i < 1024; i++) {
        h_data1[i] = (float)i;
        h_data2[i] = (float)(i * 2);
    }

    // Define grid and block dimensions
    dim3 gridDim(4);
    dim3 blockDim(256);

    // Asynchronously copy first chunk and launch its kernel in stream1
    cudaMemcpyAsync(d_data1, h_data1, dataSizeBytes,
                    cudaMemcpyHostToDevice, stream1);
    computeKernel<<<gridDim, blockDim, 0, stream1>>>((float*)d_data1, (float*)d_result1);
    cudaMemcpyAsync(h_result1, d_result1, dataSizeBytes,
                    cudaMemcpyDeviceToHost, stream1);

    // In parallel, do the same on stream2
    cudaMemcpyAsync(d_data2, h_data2, dataSizeBytes, cudaMemcpyHostToDevice, stream2);
    computeKernel<<<gridDim, blockDim, 0, stream2>>>((float*)d_data2, (float*)d_result2);
    cudaMemcpyAsync(h_result2, d_result2, dataSizeBytes,
                    cudaMemcpyDeviceToHost, stream2);

    // Wait for both streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Print some results
    printf("Stream 1 result[0]: %f\n", h_result1[0]);
    printf("Stream 2 result[0]: %f\n", h_result2[0]);

    // Cleanup
    cudaFreeAsync(d_data1, stream1);
    cudaFreeAsync(d_result1, stream1);
    cudaFreeAsync(d_data2, stream2);
    cudaFreeAsync(d_result2, stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Free host memory
    cudaFreeHost(h_data1);
    cudaFreeHost(h_data2);
    cudaFreeHost(h_result1);
    cudaFreeHost(h_result2);

    return 0;
}
