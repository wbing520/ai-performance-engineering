#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
namespace cg = cooperative_groups;

__global__ void persistentKernel(float* data, int N, int iters) {
    cg::grid_group grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; ++i) {
        if (idx < N) data[idx] = data[idx] * 0.5f + 1.0f;
        grid.sync();
    }
}

torch::Tensor run_persistent(torch::Tensor x, int iters) {
    float* data = x.data_ptr<float>();
    int N = x.numel();
    int threads=256, blocks=(N+threads-1)/threads;
    persistentKernel<<<blocks,threads>>>(data, N, iters);
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_persistent", &run_persistent, "Persistent Kernel");
}
