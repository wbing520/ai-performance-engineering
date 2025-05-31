#include <cuda_runtime.h>
#include <iostream>

__global__ void ker(float* ptr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) ptr[idx] += 1.0f;
}

int main() {
    const int N = 1<<20;
    float *ptr;
    cudaMallocManaged(&ptr, N * sizeof(float));

    cudaMemPrefetchAsync(ptr, N * sizeof(float), 0, 0);

    int threads = 256, blocks = (N + threads-1)/threads;
    ker<<<blocks, threads>>>(ptr, N);
    cudaDeviceSynchronize();

    std::cout << "ptr[0] = " << ptr[0] << std::endl;
    cudaFree(ptr);
    return 0;
}
