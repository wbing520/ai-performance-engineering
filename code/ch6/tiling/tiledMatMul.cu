#include <cuda_runtime.h>
#include <iostream>
#define TILE_SIZE 32

__global__ void tiledMatMul(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int col = blockIdx.x*TILE_SIZE + threadIdx.x;
    float sum=0.0f;
    for(int t=0; t<N; t+=TILE_SIZE) {
        sA[threadIdx.y][threadIdx.x] = A[row*N + t + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y)*N + col];
        __syncthreads();
        for(int k=0; k<TILE_SIZE; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if(row<N && col<N) C[row*N+col] = sum;
}

int main() {
    const int N = 1024;
    size_t bytes = N*N*sizeof(float);
    float *h_A = new float[N*N], *h_B = new float[N*N], *h_C = new float[N*N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    dim3 block(TILE_SIZE,TILE_SIZE), grid((N+TILE_SIZE-1)/TILE_SIZE,(N+TILE_SIZE-1)/TILE_SIZE);
    tiledMatMul<<<grid, block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}
