// basic_streams.cu -- CUDA 13.0 stream overlap demo with error handling.

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

__global__ void scale_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * scale + 0.001f;
  }
}

int main() {
  constexpr int N = 1 << 20;
  constexpr size_t BYTES = N * sizeof(float);

  float *h_a = nullptr, *h_b = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, BYTES));
  CUDA_CHECK(cudaMallocHost(&h_b, BYTES));
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, BYTES));
  CUDA_CHECK(cudaMalloc(&d_b, BYTES));

  cudaStream_t stream1 = nullptr, stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0));

  CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream2));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  scale_kernel<<<grid, block, 0, stream2>>>(d_b, N, 0.9f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream2));

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  std::printf("stream1 result: %.3f\n", h_a[0]);
  std::printf("stream2 result: %.3f\n", h_b[0]);

  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return 0;
}
