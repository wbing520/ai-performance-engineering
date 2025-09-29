// loop_unrolling.cu -- loop unrolling example with separate input/output arrays.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void kernel_unrolled(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
#pragma unroll 4
    for (int i = 0; i < 16; ++i) {
      val = val * 1.001f + 0.001f;
    }
    out[idx] = val;
  }
}

int main() {
  float *h_in, *h_out;
  cudaMallocHost(&h_in, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  kernel_unrolled<<<grid, block>>>(d_in, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.3f\n", h_out[0]);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
