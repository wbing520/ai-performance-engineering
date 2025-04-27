#include <cuda_runtime.h>
#include <cmath>

// Static uneven workload per thread
__global__ void computeKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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
    size_t bytes = N * sizeof(float);
    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = float(i)/N;
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    dim3 block(256), grid((N+255)/256);
    computeKernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
