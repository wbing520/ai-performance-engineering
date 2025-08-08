#include <cuda_runtime.h>
#include <iostream>

__global__ void uncoalescedCopy(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx * stride];
    }
}

int main() {
    const int n = 1 << 20, stride = 2;
    float *h_in = new float[n * stride], *h_out = new float[n];
    for(int i = 0; i < n * stride; ++i) h_in[i] = float(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * stride * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * stride * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (n + threads - 1) / threads;
    uncoalescedCopy<<<blocks, threads>>>(d_in, d_out, n, stride);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
