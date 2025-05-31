#include <cuda_runtime.h>
const int N = 1000000;

__global__ void addSequential(const float* A, const float* B, float* C, int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for(int i = 0; i < N; ++i) C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for(int i = 0; i < N; ++i){ h_A[i] = i; h_B[i] = 2*i; }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    addSequential<<<1,1>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
