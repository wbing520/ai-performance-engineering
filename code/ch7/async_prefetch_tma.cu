// async_prefetch_tma.cu -- simplified tiled streaming example (no TMA).

#include <cuda_runtime.h>
#include <cstdio>

constexpr int TILE_SIZE = 1024;

__global__ void kernel(const float* data, float* out, int tiles) {
  extern __shared__ float smem[];
  const int tid = threadIdx.x;

  for (int t = 0; t < tiles; ++t) {
    const float* tile = data + t * TILE_SIZE;
    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
      smem[i] = tile[i];
    }
    __syncthreads();

    for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
      out[t * TILE_SIZE + i] = smem[i] * 2.0f;
    }
    __syncthreads();
  }
}

int main() {
  constexpr int tiles = 64;
  constexpr int total = tiles * TILE_SIZE;

  float *h_in, *h_out;
  cudaMallocHost(&h_in, total * sizeof(float));
  cudaMallocHost(&h_out, total * sizeof(float));
  for (int i = 0; i < total; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  cudaMalloc(&d_in, total * sizeof(float));
  cudaMalloc(&d_out, total * sizeof(float));
  cudaMemcpy(d_in, h_in, total * sizeof(float), cudaMemcpyHostToDevice);

  kernel<<<1, 256, TILE_SIZE * sizeof(float)>>>(d_in, d_out, tiles);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
