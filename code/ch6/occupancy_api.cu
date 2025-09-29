// occupancy_api.cu
// Demonstration of cudaOccupancyMaxPotentialBlockSize with error handling
// and correct interpretation of grid size.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void sampleKernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
  }
}

int main() {
  constexpr int N = 1 << 20;
  float *h_data = new float[N];
  for (int i = 0; i < N; ++i) {
    h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
  }

  float *d_data = nullptr;
  cudaMalloc(&d_data, N * sizeof(float));
  cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

  int min_grid = 0;
  int block_size = 0;
  cudaOccupancyMaxPotentialBlockSize(
      &min_grid,
      &block_size,
      sampleKernel,
      /*dynamicSmemBytes=*/0,
      /*blockSizeLimit=*/0);

  printf("Suggested block size: %d\n", block_size);
  printf("Minimum grid for full occupancy: %d\n", min_grid);

  int grid = (N + block_size - 1) / block_size;
  grid = (grid < min_grid) ? min_grid : grid;

  sampleKernel<<<grid, block_size>>>(d_data, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("First result = %.3f\n", h_data[0]);

  cudaFree(d_data);
  delete[] h_data;
  return 0;
}
