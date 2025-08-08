#include <cuda_runtime.h>
struct float4 { float x, y, z, w; };

__global__ void copyVector(const float4* __restrict__ in, float4* __restrict__ out, int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        out[idx] = in[idx];
    }
}

int main() {
    const int N4 = (1 << 20) / 4;
    float4 *h_in = new float4[N4], *h_out = new float4[N4];
    for(int i = 0; i < N4; ++i) {
        h_in[i] = {float(4*i), float(4*i+1), float(4*i+2), float(4*i+3)};
    }

    float4 *d_in, *d_out;
    cudaMalloc(&d_in, N4 * sizeof(float4));
    cudaMalloc(&d_out, N4 * sizeof(float4));
    cudaMemcpy(d_in, h_in, N4 * sizeof(float4), cudaMemcpyHostToDevice);

    int threads = 256, blocks = (N4 + threads - 1) / threads;
    copyVector<<<blocks, threads>>>(d_in, d_out, N4);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
