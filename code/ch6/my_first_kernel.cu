// my_first_kernel.cu
// Minimal CUDA kernel example used in Chapter 6.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void double_values(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= 2.0f;
  }
}

int main() {
  constexpr int N = 1'000'000;
  size_t bytes = N * sizeof(float);

  float* h_data;
  cudaMallocHost(&h_data, bytes);
  for (int i = 0; i < N; ++i) {
    h_data[i] = 1.0f;
  }

  float* d_data;
  cudaMalloc(&d_data, bytes);
  cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (N + block - 1) / block;
  double_values<<<grid, block>>>(d_data, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
  printf("First value: %.1f\n", h_data[0]);

  cudaFree(d_data);
  cudaFreeHost(h_data);
  return 0;
}
