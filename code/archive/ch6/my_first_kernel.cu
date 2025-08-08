#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;
    float *h_input = nullptr;
    float *d_input = nullptr;

    // 1) Allocate input array of size N on host (pinned memory for faster transfer)
    cudaMallocHost(&h_input, N * sizeof(float));
    // 2) Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }
    // 3) Allocate device memory for input
    cudaMalloc(&d_input, N * sizeof(float));
    // 4) Copy data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    // 5) Choose kernel launch parameters
    const int threadsPerBlock = 256;               // e.g., 256 threads per block
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // 6) Launch kernel
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    // 7) Wait for kernel to finish
    cudaDeviceSynchronize();
    // 8) Copy results from device back to host
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
    // Cleanup
    cudaFree(d_input);
    cudaFreeHost(h_input);
    return 0;
}
