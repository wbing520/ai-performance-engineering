// occupancy_api.cu -- demonstrate cudaOccupancyMaxPotentialBlockSize (CUDA 12.9).

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void sampleKernel(float* data, int n) {
  extern __shared__ float tile[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    tile[threadIdx.x] = data[idx];
    __syncthreads();
    data[idx] = sqrtf(tile[threadIdx.x] * tile[threadIdx.x] + 1.0f);
  }
}

int main() {
  constexpr int N = 1 << 20;
  float* h_data = new float[N];
  for (int i = 0; i < N; ++i) {
    h_data[i] = static_cast<float>(i % 1000) / 1000.0f;
  }

  float* d_data = nullptr;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  int min_grid = 0;
  int block_size = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &min_grid,
      &block_size,
      sampleKernel,
      block_size * sizeof(float),
      0));

  std::printf("Suggested block size: %d\n", block_size);
  std::printf("Minimum grid for full occupancy: %d\n", min_grid);

  int grid = (N + block_size - 1) / block_size;
  if (grid < min_grid) grid = min_grid;

  sampleKernel<<<grid, block_size, block_size * sizeof(float)>>>(d_data, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("First result = %.3f\n", h_data[0]);

  CUDA_CHECK(cudaFree(d_data));
  delete[] h_data;
  return 0;
}
