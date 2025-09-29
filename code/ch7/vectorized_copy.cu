// vectorized_copy.cu -- vectorized global load using float4 for Chapter 7.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void copyVectorized(const float4* in, float4* out, int n_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vec) {
    out[idx] = in[idx];
  }
}

int main() {
  float *h_in, *h_out;
  cudaMallocHost(&h_in, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }

  float *d_in_raw, *d_out_raw;
  cudaMalloc(&d_in_raw, N * sizeof(float));
  cudaMalloc(&d_out_raw, N * sizeof(float));
  cudaMemcpy(d_in_raw, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  int n_vec = N / 4;
  dim3 block(256);
  dim3 grid((n_vec + block.x - 1) / block.x);
  copyVectorized<<<grid, block>>>(reinterpret_cast<float4*>(d_in_raw),
                                  reinterpret_cast<float4*>(d_out_raw),
                                  n_vec);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out_raw, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_in_raw);
  cudaFree(d_out_raw);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
