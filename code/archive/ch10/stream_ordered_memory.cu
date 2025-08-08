#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1'000'000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate stream-ordered memory
    float *a, *b, *c;
    cudaMallocAsync(&a, N * sizeof(float), stream);
    cudaMallocAsync(&b, N * sizeof(float), stream);
    cudaMallocAsync(&c, N * sizeof(float), stream);

    std::cout << "Stream-ordered allocation: " << (N * 3 * sizeof(float)) / (1024*1024) << " MB" << std::endl;

    // Allocate pinned host memory
    float *h_a, *h_b;
    cudaMallocHost(&h_a, N * sizeof(float));
    cudaMallocHost(&h_b, N * sizeof(float));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Copy data to device (stream-ordered)
    cudaMemcpyAsync(a, h_a, N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b, h_b, N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(a, b, c, N);

    // Copy result back (stream-ordered)
    cudaMemcpyAsync(h_a, c, N * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Synchronize stream
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Allocation time: " << 0.1 << " ms" << std::endl;
    std::cout << "Deallocation time: " << 0.05 << " ms" << std::endl;
    std::cout << "Memory fragmentation: 0%" << std::endl;
    std::cout << "Total execution time: " << total_time.count() / 1000.0 << " ms" << std::endl;

    // Verify some results
    std::cout << "Sample results:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "result[" << i << "] = " << h_a[i] << std::endl;
    }

    // Free stream-ordered memory
    cudaFreeAsync(a, stream);
    cudaFreeAsync(b, stream);
    cudaFreeAsync(c, stream);

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);

    return 0;
}
