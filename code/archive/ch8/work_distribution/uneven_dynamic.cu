#include <cuda_runtime.h>
#include <iostream>

// Dynamic workload via atomic queue
__global__ void computeKernelDynamic(const float* input, float* output, int N) {
    __shared__ unsigned int idx;
    if (threadIdx.x == 0) idx = 0;
    __syncthreads();
    while (true) {
        unsigned int i = atomicAdd(&idx, 1);
        if (i >= N) break;
        float result = 0.0f;
        int work = i % 256;
        for (int j = 0; j < work; ++j) {
            result += sinf(input[i]) * cosf(input[i]);
        }
        output[i] = result;
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);
    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = float(i)/N;
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 block(256), grid((N+255)/256);
    computeKernelDynamic<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
