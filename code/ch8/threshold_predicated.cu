// threshold_predicated.cu -- predicated version of thresholding.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void threshold_predicated(const float* x, float* y, float threshold, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = x[idx];
    y[idx] = (val > threshold) ? val : 0.0f;
  }
}

int main() {
  float *h_x, *h_y;
  cudaMallocHost(&h_x, N * sizeof(float));
  cudaMallocHost(&h_y, N * sizeof(float));
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  threshold_predicated<<<grid, block>>>(d_x, d_y, 0.5f, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("y[0]=%.3f\n", h_y[0]);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  return 0;
}
