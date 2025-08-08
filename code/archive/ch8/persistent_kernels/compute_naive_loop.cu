#include <cuda_runtime.h>

__global__ void computeStep(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 0.5f + 1.0f;
    }
}

int main() {
    const int N = 1<<10;
    float *d;
    cudaMalloc(&d, N*sizeof(float));
    dim3 b(256), g((N+255)/256);
    for (int i = 0; i < 1000; ++i) {
        computeStep<<<g,b>>>(d, N);
        cudaDeviceSynchronize();
    }
    cudaFree(d);
    return 0;
}
