// stream_ordered_allocator.cu -- async allocator example with error checks.

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 20;

__global__ void compute_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    out[idx] = val * val + 1.0f;
  }
}

int main() {
  float *h_src = nullptr, *h_dst1 = nullptr, *h_dst2 = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_src, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_dst1, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_dst2, N * sizeof(float)));
  for (int i = 0; i < N; ++i) h_src[i] = static_cast<float>(i);

  cudaStream_t stream1 = nullptr, stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));

  float *d_in1 = nullptr, *d_out1 = nullptr;
  float *d_in2 = nullptr, *d_out2 = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_in1, N * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&d_out1, N * sizeof(float), stream1));
  CUDA_CHECK(cudaMallocAsync(&d_in2, N * sizeof(float), stream2));
  CUDA_CHECK(cudaMallocAsync(&d_out2, N * sizeof(float), stream2));

  CUDA_CHECK(cudaMemcpyAsync(d_in1, h_src, N * sizeof(float), cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(d_in2, h_src, N * sizeof(float), cudaMemcpyHostToDevice, stream2));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  compute_kernel<<<grid, block, 0, stream1>>>(d_in1, d_out1, N);
  compute_kernel<<<grid, block, 0, stream2>>>(d_in2, d_out2, N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_dst1, d_out1, N * sizeof(float), cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(h_dst2, d_out2, N * sizeof(float), cudaMemcpyDeviceToHost, stream2));

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  std::printf("stream1 result[0]=%.1f\n", h_dst1[0]);
  std::printf("stream2 result[0]=%.1f\n", h_dst2[0]);

  CUDA_CHECK(cudaFreeAsync(d_in1, stream1));
  CUDA_CHECK(cudaFreeAsync(d_out1, stream1));
  CUDA_CHECK(cudaFreeAsync(d_in2, stream2));
  CUDA_CHECK(cudaFreeAsync(d_out2, stream2));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFreeHost(h_src));
  CUDA_CHECK(cudaFreeHost(h_dst1));
  CUDA_CHECK(cudaFreeHost(h_dst2));
  return 0;
}
