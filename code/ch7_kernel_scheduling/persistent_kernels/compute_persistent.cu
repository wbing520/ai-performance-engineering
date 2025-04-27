#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void persistentKernel(float* data, int N, int iters) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
        if (idx < N) data[idx] = data[idx] * 0.5f + 1.0f;
        grid.sync();
    }
}

int main() {
    const int N = 1<<10;
    float *d;
    cudaMalloc(&d, N*sizeof(float));
    dim3 b(256), g((N+255)/256);
    persistentKernel<<<g,b>>>(d, N, 1000);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
