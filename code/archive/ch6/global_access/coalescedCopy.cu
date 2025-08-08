#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedCopy(const float* __restrict__ in, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

int main() {
    const int n = 1 << 20;
    float *h_in = new float[n], *h_out = new float[n];
    for(int i = 0; i < n; ++i) h_in[i] = float(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;
    coalescedCopy<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
