// fusedL2Norm.cu
// Chapter 9 fused L2-normalization example rewritten with best practices.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

constexpr int HIDDEN = 4096;
constexpr int BATCHES = 512;

__global__ void fused_l2norm(const float* __restrict__ x,
                              float* __restrict__ out,
                              int hidden) {
  const int batch = blockIdx.x;
  extern __shared__ float sdata[];

  float sum = 0.0f;
  const float* batch_ptr = x + static_cast<size_t>(batch) * hidden;
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

  if (threadIdx.x == 0) {
    sdata[0] = sqrtf(sdata[0]);
  }
  __syncthreads();
  float norm = sdata[0];
  if (norm < 1e-8f) norm = 1.0f;

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

  float *d_in, *d_out;
  cudaMalloc(&d_in, h_in.size() * sizeof(float));
  cudaMalloc(&d_out, h_in.size() * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(BATCHES);
  dim3 block(256);
  size_t smem = block.x * sizeof(float);
  fused_l2norm<<<grid, block, smem>>>(d_in, d_out, HIDDEN);
  cudaDeviceSynchronize();

  std::vector<float> h_out(h_in.size());
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> ref(h_in.size());
  reference(h_in, ref, BATCHES, HIDDEN);

  float max_err = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) {
    max_err = fmaxf(max_err, fabsf(h_out[i] - ref[i]));
  }
  printf("Max error: %e\n", max_err);

  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
