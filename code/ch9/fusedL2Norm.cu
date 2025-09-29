// fusedL2Norm.cu -- Chapter 9 fused L2-normalization with validation and error checks.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int HIDDEN = 4096;
constexpr int BATCHES = 512;

dim3 make_launch_dim(int blocks) { return dim3(blocks); }

__global__ void fused_l2norm(const float* __restrict__ x,
                              float* __restrict__ out,
                              int hidden) {
  extern __shared__ float sdata[];
  const int batch = blockIdx.x;
  const float* batch_ptr = x + static_cast<size_t>(batch) * hidden;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float val = batch_ptr[i];
    sum += val * val;
  }
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }
    __syncthreads();
  }
  float norm = sdata[0];
  norm = sqrtf(norm);
  norm = norm < 1e-8f ? 1.0f : norm;
  float* out_batch = out + static_cast<size_t>(batch) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    out_batch[i] = batch_ptr[i] / norm;
  }
}

static void reference(const std::vector<float>& input,
                      std::vector<float>& output,
                      int batches, int hidden) {
  for (int b = 0; b < batches; ++b) {
    double sum = 0.0;
    for (int i = 0; i < hidden; ++i) {
      float val = input[b * hidden + i];
      sum += static_cast<double>(val) * val;
    }
    double norm = std::sqrt(sum);
    if (norm < 1e-8) norm = 1.0;
    for (int i = 0; i < hidden; ++i) {
      output[b * hidden + i] = static_cast<float>(input[b * hidden + i] / norm);
    }
  }
}

int main() {
  std::vector<float> h_in(BATCHES * HIDDEN);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<float>(std::sin(i));
  }

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_in.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid = make_launch_dim(BATCHES);
  size_t smem = block.x * sizeof(float);

  fused_l2norm<<<grid, block, smem>>>(d_in, d_out, HIDDEN);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(h_in.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ref(h_in.size());
  reference(h_in, ref, BATCHES, HIDDEN);

  float max_err = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) {
    float diff = std::fabs(h_out[i] - ref[i]);
    if (diff > max_err) max_err = diff;
  }
  std::printf("Max error: %e\n", max_err);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
