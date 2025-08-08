// dp_host_launched.cu
#include <cuda_runtime.h>

__global__ void childKernel(float* data, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= data[idx];
}

__global__ void parentKernel(float* data, int N) {
    // placeholder
}

int main() {
    const int N = 1<<20;
    float* d_data;
    cudaMalloc(&d_data, N*sizeof(float));
    parentKernel<<<1,1>>>(d_data, N);
    cudaDeviceSynchronize();

    int half = N/2;
    childKernel<<<(half+255)/256,256>>>(d_data, half);
    childKernel<<<(half+255)/256,256>>>(d_data+half, half);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}
