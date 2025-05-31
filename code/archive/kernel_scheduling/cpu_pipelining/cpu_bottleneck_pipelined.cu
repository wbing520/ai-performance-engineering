#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>

void cpuWork(int i) {
    volatile float x = 0;
    for(int j=0;j<10000;++j) x += sinf(j)*cosf(j);
}

__global__ void gpuWork(float* data, int N) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < N) data[idx] += 1.0f;
}

int main() {
    const int N = 1<<20;
    float *d; cudaMalloc(&d, N*sizeof(float));
    for(int i=0;i<100; ++i) {
        std::thread cpu_thread(cpuWork, i);
        gpuWork<<<N/256,256>>>(d,N);
        cpu_thread.join();
        cudaDeviceSynchronize();
    }
    cudaFree(d);
    return 0;
}
