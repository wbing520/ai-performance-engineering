#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _status = (call);                                            \
    if (_status != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(_status));                                  \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

constexpr int N = 1 << 20;

__global__ void work(float* data, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) data[idx] *= scale;
}

int main() {
  float *h_a, *h_b;
  CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a, *d_b;
  CUDA_CHECK(cudaMallocAsync(&d_a, N * sizeof(float), 0));
  CUDA_CHECK(cudaMallocAsync(&d_b, N * sizeof(float), 0));

  cudaStream_t streams[2];
  for (auto& s : streams) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice, streams[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice, streams[1]));

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  work<<<grid, block, 0, streams[0]>>>(d_a, 1.05f);
  work<<<grid, block, 0, streams[1]>>>(d_b, 0.95f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost, streams[0]));
  CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost, streams[1]));

  for (auto& s : streams) CUDA_CHECK(cudaStreamSynchronize(s));

  printf("a[0]=%.2f b[0]=%.2f\n", h_a[0], h_b[0]);

  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFreeAsync(d_a, 0));
  CUDA_CHECK(cudaFreeAsync(d_b, 0));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return 0;
}
