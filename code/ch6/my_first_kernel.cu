// Architecture-specific optimizations for CUDA 12.8
// Targets Blackwell B200/B300 (sm_100)
// my_first_kernel.cu
// Basic CUDA kernel example from Chapter 6

#include <cuda_runtime.h>
#include <stdio.h>

//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
// - input : device pointer to float array of length N
// - N : total number of elements in the input
//-------------------------------------------------------
__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;
    float *h_input = nullptr;
    float *d_input = nullptr;
    
    // 1) Allocate input float array of size N on host
    cudaMallocHost(&h_input, N * sizeof(float));
    
    // 2) Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }
    
    // 3) Allocate device memory for input on the device
    cudaMalloc(&d_input, N * sizeof(float));
    
    // 4) Copy data from the host to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 5) Choose kernel launch parameters
    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;
    // Number of blocks per grid (3,907 for N = 1000000)
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 6) Launch myKernel across blocksPerGrid blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    
    // 7) Wait for the kernel to finish running on device
    cudaDeviceSynchronize();
    
    // 8) When finished, copy the results
    // (stored in d_input) from the device back to
    // host (stored in h_input)
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("First 5 values: %.1f %.1f %.1f %.1f %.1f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    
    // Cleanup: Free memory on the device and host
    cudaFree(d_input);
    cudaFreeHost(h_input);
    
    // return 0 for success!
    return 0;
}

// CUDA 12.8 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 12.8 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
