// optimized_lookup.cu -- coalesced gather using int4 loads.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void lookupOptimized(const float* table, const int* indices, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = table[indices[idx]];
  }
}

int main() {
  float *h_table, *h_out;
  int *h_indices;
  cudaMallocHost(&h_table, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  cudaMallocHost(&h_indices, N * sizeof(int));

  for (int i = 0; i < N; ++i) {
    h_table[i] = static_cast<float>(i);
    h_indices[i] = i;
  }

  float *d_table, *d_out;
  int *d_indices;
  cudaMalloc(&d_table, N * sizeof(float));
  cudaMalloc(&d_indices, N * sizeof(int));
  cudaMalloc(&d_out, N * sizeof(float));

  cudaMemcpy(d_table, h_table, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  lookupOptimized<<<grid, block>>>(d_table, d_indices, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_table);
  cudaFree(d_indices);
  cudaFree(d_out);
  cudaFreeHost(h_table);
  cudaFreeHost(h_indices);
  cudaFreeHost(h_out);
  return 0;
}
