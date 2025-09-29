// unified_memory.cu
// Minimal example using CUDA managed memory with prefetching.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * data[idx] + 1.0f;
  }
}

int main() {
  constexpr int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  float* data = nullptr;
  cudaMallocManaged(&data, bytes);

  for (int i = 0; i < N; ++i) {
    data[i] = static_cast<float>(i);
  }

  int device = 0;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(data, bytes, device);

  int block = 256;
  int grid = (N + block - 1) / block;
  kernel<<<grid, block>>>(data, N);
  cudaDeviceSynchronize();

  cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId);
  cudaDeviceSynchronize();

  printf("First value: %.1f\n", data[0]);

  cudaFree(data);
  return 0;
}
