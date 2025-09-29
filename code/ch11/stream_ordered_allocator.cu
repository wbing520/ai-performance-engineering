// stream_ordered_allocator.cu -- CUDA 12.9 async allocation across streams.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int N = 1 << 20;

__global__ void compute_kernel(const float* in, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    out[idx] = val * val + 1.0f;
  }
}

int main() {
  float *h_src, *h_dst1, *h_dst2;
  cudaMallocHost(&h_src, N * sizeof(float));
  cudaMallocHost(&h_dst1, N * sizeof(float));
  cudaMallocHost(&h_dst2, N * sizeof(float));
  for (int i = 0; i < N; ++i) h_src[i] = static_cast<float>(i);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  float *d_in1, *d_out1, *d_in2, *d_out2;
  cudaMallocAsync(&d_in1, N * sizeof(float), stream1);
  cudaMallocAsync(&d_out1, N * sizeof(float), stream1);
  cudaMallocAsync(&d_in2, N * sizeof(float), stream2);
  cudaMallocAsync(&d_out2, N * sizeof(float), stream2);

  cudaMemcpyAsync(d_in1, h_src, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_in2, h_src, N * sizeof(float), cudaMemcpyHostToDevice, stream2);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  compute_kernel<<<grid, block, 0, stream1>>>(d_in1, d_out1, N);
  compute_kernel<<<grid, block, 0, stream2>>>(d_in2, d_out2, N);

  cudaMemcpyAsync(h_dst1, d_out1, N * sizeof(float), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_dst2, d_out2, N * sizeof(float), cudaMemcpyDeviceToHost, stream2);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  printf("stream1 result[0]=%.1f\n", h_dst1[0]);
  printf("stream2 result[0]=%.1f\n", h_dst2[0]);

  cudaFreeAsync(d_in1, stream1);
  cudaFreeAsync(d_out1, stream1);
  cudaFreeAsync(d_in2, stream2);
  cudaFreeAsync(d_out2, stream2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaFreeHost(h_src);
  cudaFreeHost(h_dst1);
  cudaFreeHost(h_dst2);
  return 0;
}
