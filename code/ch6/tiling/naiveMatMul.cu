#include <cuda_runtime.h>
#include <iostream>

__global__ void naiveMatMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        float sum = 0;
        for(int k=0; k<N; ++k) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N*N*sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    dim3 block(32,32), grid((N+31)/32,(N+31)/32);
    naiveMatMul<<<grid, block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
