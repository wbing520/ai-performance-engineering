// occupancy_tuning.cu -- simple __launch_bounds__ illustration.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ __launch_bounds__(256, 4)
void kernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    out[idx] = sqrtf(val * val + 1.0f);
  }
}

int main() {
  float *h_in, *h_out;
  cudaMallocHost(&h_in, N * sizeof(float));
  cudaMallocHost(&h_out, N * sizeof(float));
  for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

  float *d_in, *d_out;
  cudaMalloc(&d_in, N * sizeof(float));
  cudaMalloc(&d_out, N * sizeof(float));
  cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  kernel<<<grid, block>>>(d_in, d_out, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
  printf("out[0]=%.1f\n", h_out[0]);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
