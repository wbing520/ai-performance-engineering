// transpose_naive.cu -- naive matrix transpose for Chapter 7.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int WIDTH = 1024;

__global__ void transpose_naive(const float* in, float* out, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < width) {
    out[y * width + x] = in[x * width + y];
  }
}

int main() {
  size_t bytes = WIDTH * WIDTH * sizeof(float);
  float *h_in, *h_out;
  cudaMallocHost(&h_in, bytes);
  cudaMallocHost(&h_out, bytes);
  for (int i = 0; i < WIDTH * WIDTH; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

  dim3 block(32, 32);
  dim3 grid((WIDTH + block.x - 1) / block.x, (WIDTH + block.y - 1) / block.y);
  transpose_naive<<<grid, block>>>(d_in, d_out, WIDTH);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
