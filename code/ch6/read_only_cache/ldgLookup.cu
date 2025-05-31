#include <cuda_runtime.h>
#define T 1024

__global__ void ldgLookup(const float* __restrict__ table, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        int t = idx % T;
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;
    float *h_table = new float[T], *h_out = new float[N];
    float *d_table, *d_out;
    cudaMalloc(&d_table, T*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_table, h_table, T*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(256), grid((N+255)/256);
    ldgLookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_table); cudaFree(d_out);
    delete[] h_table; delete[] h_out;
    return 0;
}
