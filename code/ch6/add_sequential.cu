// add_sequential.cu
// Naive sequential CUDA example for Chapter 6 (illustrates poor GPU utilization).

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1'000'000;

__global__ void addSequential(const float* A, const float* B, float* C, int n) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) {
      C[i] = A[i] + B[i];
    }
  }
}

int main() {
  float *h_A, *h_B, *h_C;
  cudaMallocHost(&h_A, N * sizeof(float));
  cudaMallocHost(&h_B, N * sizeof(float));
  cudaMallocHost(&h_C, N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(2 * i);
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, N * sizeof(float));
  cudaMalloc(&d_B, N * sizeof(float));
  cudaMalloc(&d_C, N * sizeof(float));

  cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

  addSequential<<<1, 1>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("C[0]=%.1f, C[N-1]=%.1f\n", h_C[0], h_C[N - 1]);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  return 0;
}
