#include <cuda_runtime.h>
#include <cmath>

__global__ void independentOps(const float *a, const float *b, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = a[i];
        float y = b[i];
        float u = x * x;
        float v = y * y;
        out[i] = sqrtf(u + v);
    }
}

int main() {
    const int N = 1 << 20;
    float *h_a = new float[N], *h_b = new float[N], *h_out = new float[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = float(i);
        h_b[i] = float(2*i);
    }
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (N+threads-1)/threads;
    independentOps<<<blocks, threads>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    delete[] h_a; delete[] h_b; delete[] h_out;
    return 0;
}
