// coalesced_copy.cu
// Demo: Coalesced Global Memory Copy
// Hardware: Grace-Blackwell (sm_90) or Hopper (sm_80)
// CUDA: 13.0  C++17
// Build: make

#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define CHECK_CUDA(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err); \
        std::exit(EXIT_FAILURE);                                \
    } } while(0)

// Coalesced copy kernel: each thread copies one float
__global__ void coalescedCopy(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

int main() {
    // Problem size
    const int N = 1 << 24;              // 16M elements (~64 MB per buffer)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in  = nullptr, *h_out = nullptr;
    h_in  = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    assert(h_in && h_out);

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Device pointers
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // Copy input to GPU
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Record time with CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    coalescedCopy<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Kernel time: " << ms << " ms\n";

    // Copy back and verify
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i += (N/4)) {
        assert(h_out[i] == h_in[i]);
    }

    std::cout << "Result verified!\n";

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
