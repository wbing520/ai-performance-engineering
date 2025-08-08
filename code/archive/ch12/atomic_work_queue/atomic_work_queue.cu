#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>

using namespace cooperative_groups;

// Device global counter for work queue
__device__ unsigned int globalIndex = 0;

// Before batching - each thread does one atomic operation
__global__ void computeKernelBeforeBatching(const float* input, float* output, int N) {
    int idx = atomicAdd(&globalIndex, 1);
    if (idx < N) {
        // Each thread does a variable amount of work based on idx
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}

// After batching - each warp claims a batch of work
__global__ void computeKernelAfterBatching(const float* input, float* output, int N) {
    // Warp-level mask and lane ID
    const unsigned mask = __activemask(); // active threads in this warp
    int lane = threadIdx.x & (warpSize - 1); // lane ∈ [0, 31]

    while (true) {
        // Warp leader atomically claims the next batch of 32 indices
        unsigned int base;
        if (lane == 0) {
            // one atomic per warp
            base = atomicAdd(&globalIndex, warpSize);
        }
        
        // Broadcast base to all lanes in the warp
        // register-level warp shuffle
        base = __shfl_sync(mask, base, 0);
        
        // Compute each thread's global index and exit if out of range
        unsigned int idx = base + lane;
        // dynamic termination
        if (idx >= (unsigned int)N) break;
        
        // Per‑index work: variable loop bound
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}

int main() {
    const int N = 1 << 20;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    // Initialize input data
    float *h_in;
    cudaMallocHost(&h_in, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_in[i] = float(i) / N;
    }
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Reset global counter
    unsigned int zero = 0;
    cudaMemcpyToSymbol(globalIndex, &zero, sizeof(unsigned int));
    
    // Launch with 256 threads per block
    dim3 block(256), grid((N + 255) / 256);
    
    // Choose which version to run
    bool use_batching = true;
    
    if (use_batching) {
        computeKernelAfterBatching<<<grid, block>>>(d_in, d_out, N);
    } else {
        computeKernelBeforeBatching<<<grid, block>>>(d_in, d_out, N);
    }
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    
    return 0;
}
