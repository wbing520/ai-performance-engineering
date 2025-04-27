#include <cuda_runtime.h>
#include <iostream>

// Fused add+mul kernel
__global__ void fusedKernel(const float* a, const float* b, float* c, int N) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) c[i] = (a[i] + b[i]) * b[i];
}

int main() {
    const int N = 1<<20;
    float *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));
    for (int i = 0; i < N; ++i) a[i]=b[i]=1.0f;
    dim3 bdim(256), gdim((N+255)/256);
    fusedKernel<<<gdim,bdim>>>(a,b,c,N);
    cudaDeviceSynchronize();
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
