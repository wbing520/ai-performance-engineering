#include <cuda_runtime.h>
#include <iostream>

__global__ void threshold_predicated(const float* X, float* Y, float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = X[i];
        float val = (x > threshold) ? x : 0.0f;
        Y[i] = val;
    }
}

int main() {
    const int N = 1 << 20;
    float *h_X = new float[N], *h_Y = new float[N];
    for (int i = 0; i < N; ++i) h_X[i] = float(rand())/RAND_MAX - 0.5f;
    float *d_X, *d_Y;
    cudaMalloc(&d_X, N*sizeof(float));
    cudaMalloc(&d_Y, N*sizeof(float));
    cudaMemcpy(d_X, h_X, N*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256, blocks = (N+threads-1)/threads;
    threshold_predicated<<<blocks, threads>>>(d_X, d_Y, 0.0f, N);
    cudaDeviceSynchronize();
    cudaFree(d_X); cudaFree(d_Y);
    delete[] h_X; delete[] h_Y;
    return 0;
}
