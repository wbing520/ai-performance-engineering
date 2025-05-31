// dp_device_launched.cu
#include <cuda_runtime.h>

__global__ void childKernel(float* data, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= data[idx];
}

__global__ void parentKernel(float* data, int N) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        int half = N/2;
        void* args1[] = { &data, &half };
        void* args2[] = { &data+half, &half };
        dim3 grid((half+255)/256), block(256);
        cudaLaunchKernel((void*)childKernel, grid, block, args1, 0, 0);
        cudaLaunchKernel((void*)childKernel, grid, block, args2, 0, 0);
    }
}

int main() {
    const int N = 1<<20;
    float* d_data;
    cudaMalloc(&d_data, N*sizeof(float));
    parentKernel<<<1,1>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    return 0;
}
