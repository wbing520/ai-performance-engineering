#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ int g_index;

__global__ void persistentKernel(float* data, int N, int iterations) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int it = 0; it < iterations; ++it) {
        if (idx < N) data[idx] = data[idx] * 0.5f + 1.0f;
        grid.sync();
    }
}

int main() {
    const int N = 1024, iters = 1000;
    float* d;
    cudaMalloc(&d, N * sizeof(float));
    cudaMemset(&g_index, 0, sizeof(int));
    void* args[] = { &d, &N, &iters };
    cudaLaunchCooperativeKernel((void*)persistentKernel, (N + 255) / 256, 256, args);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
