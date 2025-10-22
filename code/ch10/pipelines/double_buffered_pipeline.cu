// double_buffered_pipeline.cu -- GEMM with CUDA Pipeline API (correct overlap).

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int TILE = 32;  // 32x32 tile keeps shared memory under 48 KB for 2-stage pipeline
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int STAGES = 2;

__device__ void load_tile(cuda::pipeline<cuda::thread_scope_block>& pipe,
                          cg::thread_block block,
                          float* tile_buffer,
                          const float* global,
                          int lda,
                          int tile_row,
                          int tile_col,
                          int rows,
                          int cols) {
  int linear = block.thread_rank();
  int stride = block.size();

  for (int offset = linear; offset < TILE_ELEMS; offset += stride) {
    int local_row = offset / TILE;
    int local_col = offset % TILE;
    int g_row = tile_row + local_row;
    int g_col = tile_col + local_col;

    if (g_row < rows && g_col < cols) {
      CUDA_CHECK(cuda::memcpy_async(block,
                                    &tile_buffer[offset],
                                    &global[g_row * lda + g_col],
                                    sizeof(float),
                                    pipe));
    } else {
      tile_buffer[offset] = 0.0f;
    }
  }
  pipe.producer_commit();
}

__global__ void gemm_pipeline_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
  cg::thread_block block = cg::this_thread_block();
  extern __shared__ float shared[];
  float* A_tiles[STAGES] = {shared, shared + TILE_ELEMS};
  float* B_tiles[STAGES] = {shared + 2 * TILE_ELEMS, shared + 3 * TILE_ELEMS};

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> pipe_state;
  auto pipe = cuda::make_pipeline(block, &pipe_state);

  int row_tile = blockIdx.y * TILE;
  int col_tile = blockIdx.x * TILE;
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;

  float accum = 0.0f;
  int num_tiles = (K + TILE - 1) / TILE;

  pipe.producer_acquire();
  load_tile(pipe, block, A_tiles[0], A, K, row_tile, 0, M, K);
  load_tile(pipe, block, B_tiles[0], B, N, 0, col_tile, K, N);

  for (int tile = 0; tile < num_tiles; ++tile) {
    pipe.consumer_wait();
    block.sync();

    int curr = tile % STAGES;
    int tile_k = min(TILE, K - tile * TILE);

    if (row_tile + thread_row < M && col_tile + thread_col < N) {
      for (int k = 0; k < tile_k; ++k) {
        float a = A_tiles[curr][thread_row * TILE + k];
        float b = B_tiles[curr][k * TILE + thread_col];
        accum += a * b;
      }
    }

    pipe.consumer_release();

    int next = (tile + 1) % STAGES;
    if (tile + 1 < num_tiles) {
      pipe.producer_acquire();
      int k_base = (tile + 1) * TILE;
      load_tile(pipe, block, A_tiles[next], A, K, row_tile, k_base, M, K);
      load_tile(pipe, block, B_tiles[next], B, N, k_base, col_tile, K, N);
    }
  }

  if (row_tile + thread_row < M && col_tile + thread_col < N) {
    C[(row_tile + thread_row) * N + (col_tile + thread_col)] = accum;
  }
}

void gemm_pipeline(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K, cudaStream_t stream = 0) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  size_t shared_bytes = 4 * TILE_ELEMS * sizeof(float);
  gemm_pipeline_kernel<<<grid, block, shared_bytes, stream>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}

int main() {
  constexpr int M = 256;
  constexpr int N = 256;
  constexpr int K = 256;

  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h_A) v = dist(rng);
  for (auto& v : h_B) v = dist(rng);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));

  gemm_pipeline(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_C(M * N);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> reference(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<double>(h_A[m * K + k]) * h_B[k * N + n];
      }
      reference[m * N + n] = static_cast<float>(sum);
    }
  }

  double max_err = 0.0;
  for (size_t i = 0; i < h_C.size(); ++i) {
    max_err = std::max(max_err, std::abs(static_cast<double>(h_C[i]) - reference[i]));
  }
  std::printf("Max error: %.6e\n", max_err);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
