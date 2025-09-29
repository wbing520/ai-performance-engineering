// vectorized_copy.cu -- vectorized global load using float4 for Chapter 7.

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

struct alignas(16) Float4 { float x, y, z, w; };

constexpr int NUM_FLOATS = 1 << 20;
static_assert(NUM_FLOATS % 4 == 0, "NUM_FLOATS must be divisible by 4");
constexpr int NUM_VEC = NUM_FLOATS / 4;

__global__ void copyVectorized(const Float4* __restrict__ in,
                               Float4* __restrict__ out,
                               int n_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_vec) {
    out[idx] = in[idx];
  }
}

int main() {
  static_assert(sizeof(Float4) == 16, "Float4 must be 16 bytes");
  Float4* h_in = nullptr;
  Float4* h_out = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_in, NUM_VEC * sizeof(Float4)));
  CUDA_CHECK(cudaMallocHost(&h_out, NUM_VEC * sizeof(Float4)));
  float* h_in_f = reinterpret_cast<float*>(h_in);
  for (int i = 0; i < NUM_FLOATS; ++i) {
    h_in_f[i] = static_cast<float>(i);
  }

  Float4* d_in = nullptr;
  Float4* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, NUM_VEC * sizeof(Float4)));
  CUDA_CHECK(cudaMalloc(&d_out, NUM_VEC * sizeof(Float4)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, NUM_VEC * sizeof(Float4), cudaMemcpyHostToDevice));

  int n_vec = NUM_VEC;
  dim3 block(256);
  dim3 grid((n_vec + block.x - 1) / block.x);
  copyVectorized<<<grid, block>>>(d_in, d_out, n_vec);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, NUM_VEC * sizeof(Float4), cudaMemcpyDeviceToHost));
  const float* h_out_f = reinterpret_cast<const float*>(h_out);
  std::printf("out[0]=%.1f out[last]=%.1f\n", h_out_f[0], h_out_f[NUM_FLOATS - 1]);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}
