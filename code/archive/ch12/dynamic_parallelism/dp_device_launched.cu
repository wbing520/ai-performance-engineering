#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void childKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * data[idx];
    }
}

__global__ void parentKernel(float* data, int N) {
    // Only one thread enqueues to avoid duplicates
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int half = N / 2;
        
        // Prepare launch parameters
        void* args1[] = { &data, &half };
        float* secondPtr = data + half;
        void* args2[] = { &secondPtr, &half };
        
        dim3 grid((half + 255) / 256), block(256);
        
        // Device-side launches (fire-and-forget)
        cudaLaunchKernel((void*)childKernel,
            grid, block, args1, 0, 0);
        cudaLaunchKernel((void*)childKernel,
            grid, block, args2, 0, 0);
    }
    // Other threads could assist, but are idle here
}

int main() {
    const int N = 1 << 20;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // Initialize data
    float *h_data;
    cudaMallocHost(&h_data, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Single launch: parentKernel will dispatch its children
    parentKernel<<<1,1>>>(d_data, N);
    cudaDeviceSynchronize(); // waits for parent + children
    
    // Cleanup
    cudaFree(d_data);
    cudaFreeHost(h_data);
    
    return 0;
}
