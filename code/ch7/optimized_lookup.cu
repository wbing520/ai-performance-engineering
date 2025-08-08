#include <cuda_runtime.h>

#define T 1024

__global__ void lookup(const float* __restrict__ table,
                      float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int t = idx % T;
        // Compiler will turn this into an LDG load from the read-only cache for faster loads
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;
    
    float* h_table = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_table, T * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));
    
    for (int i = 0; i < T; ++i) h_table[i] = float(i);
    
    float *d_table, *d_out;
    cudaMalloc(&d_table, T * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    
    cudaMemcpy(d_table, h_table, T * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(256), grid((N + 255) / 256);
    lookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFreeHost(h_table);
    cudaFreeHost(h_out);
    
    return 0;
}
