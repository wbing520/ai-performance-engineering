#include <cuda_runtime.h>
#include <iostream>

// Separate add and mul kernels
__global__ void addKernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}
__global__ void mulKernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) c[i] *= b[i];
}

int main() {
    const int N = 1<<20;
    float *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));
    for (int i = 0; i < N; ++i) a[i]=b[i]=1.0f;
    dim3 bdim(256), gdim((N+255)/256);
    addKernel<<<gdim,bdim>>>(a,b,c,N);
    mulKernel<<<gdim,bdim>>>(c,b,c,N);
    cudaDeviceSynchronize();
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
