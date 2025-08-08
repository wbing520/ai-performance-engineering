// uneven_dynamic.cu
#include <cuda_runtime.h>
#include <cmath>

__device__ unsigned int globalIndex = 0;

__global__ void computeKernelDynamic(const float* input, float* output, int N) {
    unsigned int idx;
    while (true) {
        if (threadIdx.x == 0) {
            idx = atomicAdd(&globalIndex, 1);
        }
        idx = __shfl_sync(0xFFFFFFFF, idx, 0);
        if (idx >= N) break;
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}

int main() {
    const int N = 1<<20;
    float *h_in = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; ++i) h_in[i] = float(i)/N;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(256), grid((N+255)/256);
    computeKernelDynamic<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
