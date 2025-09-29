// basic_streams.cu -- CUDA 12.9 stream overlap demo (Blackwell-ready)

#include <cuda_runtime.h>
#include <cstdio>

__global__ void scale_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * scale + 0.001f;
  }
}

int main() {
  constexpr int N = 1 << 20;
  constexpr size_t BYTES = N * sizeof(float);

  float *h_a, *h_b;
  cudaMallocHost(&h_a, BYTES);
  cudaMallocHost(&h_b, BYTES);
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a, *d_b;
  cudaMalloc(&d_a, BYTES);
  cudaMalloc(&d_b, BYTES);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  // Pipeline: copy -> compute -> copy back on each stream
  cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream2);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  scale_kernel<<<grid, block, 0, stream2>>>(d_b, N, 0.9f);

  cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  printf("stream1 result: %.3f\n", h_a[0]);
  printf("stream2 result: %.3f\n", h_b[0]);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  return 0;
}
