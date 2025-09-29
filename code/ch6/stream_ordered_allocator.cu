// stream_ordered_allocator.cu
// Minimal example of cudaMallocAsync + streams.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(const float* inp, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = inp[idx] * inp[idx] + 1.0f;
  }
}

int main() {
  constexpr int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  float *h_in, *h_out;
  cudaMallocHost(&h_in, bytes);
  cudaMallocHost(&h_out, bytes);
  for (int i = 0; i < N; ++i) {
    h_in[i] = static_cast<float>(i);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float *d_in, *d_out;
  cudaMallocAsync(&d_in, bytes, stream);
  cudaMallocAsync(&d_out, bytes, stream);

  cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream);

  int block = 256;
  int grid = (N + block - 1) / block;
  kernel<<<grid, block, 0, stream>>>(d_in, d_out, N);

  cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  printf("Result[0]=%.1f\n", h_out[0]);

  cudaFreeAsync(d_in, stream);
  cudaFreeAsync(d_out, stream);
  cudaStreamDestroy(stream);
  cudaFreeHost(h_in);
  cudaFreeHost(h_out);
  return 0;
}
