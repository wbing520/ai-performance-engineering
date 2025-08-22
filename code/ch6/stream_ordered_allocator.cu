// Architecture-specific optimizations for CUDA 12.8
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// stream_ordered_allocator.cu
// Example demonstrating stream-ordered memory allocation

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

__global__ void computeKernel(float* data, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = data[idx] * data[idx] + 1.0f;
    }
}

int main() {
    const int N = 1024 * 1024;
    
    // initialize the async memory allocator
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    
    // Desired number of bytes to keep in pool before
    // releasing back to the OS (tune as needed)
    uint64_t threshold = 64 * 1024 * 1024; // 64 MB
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Allocate memory using stream-ordered async allocation
    void *d_data1, *d_result1;
    void *d_data2, *d_result2;
    size_t dataSizeBytes = N * sizeof(float);
    
    // Use cudaMallocAsync on a given stream (best practice in modern multi-stream apps)
    cudaMallocAsync(&d_data1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_result1, dataSizeBytes, stream1);
    cudaMallocAsync(&d_data2, dataSizeBytes, stream2);
    cudaMallocAsync(&d_result2, dataSizeBytes, stream2);
    
    // Host data for initialization
    float* h_data1 = new float[N];
    float* h_data2 = new float[N];
    for (int i = 0; i < N; ++i) {
        h_data1[i] = float(i);
        h_data2[i] = float(i) * 2.0f;
    }
    
    // Asynchronously copy first chunk and launch its kernel in stream1
    cudaMemcpyAsync(d_data1, h_data1, dataSizeBytes, cudaMemcpyHostToDevice, stream1);
    dim3 gridDim((N + 255) / 256);
    dim3 blockDim(256);
    computeKernel<<<gridDim, blockDim, 0, stream1>>>((float*)d_data1, (float*)d_result1, N);
    
    // In parallel, do the same on stream2
    cudaMemcpyAsync(d_data2, h_data2, dataSizeBytes, cudaMemcpyHostToDevice, stream2);
    computeKernel<<<gridDim, blockDim, 0, stream2>>>((float*)d_data2, (float*)d_result2, N);
    
    // Wait for both streams to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // Copy results back
    float* h_result1 = new float[N];
    float* h_result2 = new float[N];
    cudaMemcpyAsync(h_result1, d_result1, dataSizeBytes, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_result2, d_result2, dataSizeBytes, cudaMemcpyDeviceToHost, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    printf("Stream 1 result[0]: %.1f\n", h_result1[0]);
    printf("Stream 2 result[0]: %.1f\n", h_result2[0]);
    
    // Cleanup
    cudaFreeAsync(d_data1, stream1);
    cudaFreeAsync(d_result1, stream1);
    cudaFreeAsync(d_data2, stream2);
    cudaFreeAsync(d_result2, stream2);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    delete[] h_data1;
    delete[] h_data2;
    delete[] h_result1;
    delete[] h_result2;
    
    return 0;
}
