// tiled_matmul.cu -- simple tiled matmul example (Chapter 7 optimized version).

#include <cuda_runtime.h>
#include <cstdio>

constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;
constexpr int TILE = 32;

__global__ void matmul_tiled(const float* A, const float* B, float* C, int m, int n, int k) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;
  for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
    int tiled_col = t * TILE + threadIdx.x;
    int tiled_row = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < m && tiled_col < k) ? A[row * k + tiled_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (tiled_row < k && col < n) ? B[tiled_row * n + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < TILE; ++i) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    C[row * n + col] = sum;
  }
}

int main() {
  size_t bytesA = M * K * sizeof(float);
  size_t bytesB = K * N * sizeof(float);
  size_t bytesC = M * N * sizeof(float);

  float *h_A, *h_B, *h_C;
  cudaMallocHost(&h_A, bytesA);
  cudaMallocHost(&h_B, bytesB);
  cudaMallocHost(&h_C, bytesC);

  for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
  for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytesA);
  cudaMalloc(&d_B, bytesB);
  cudaMalloc(&d_C, bytesC);
  cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
  printf("C[0]=%.1f\n", h_C[0]);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  return 0;
}
