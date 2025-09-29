// double_buffered_pipeline.cu
// Corrected double-buffered GEMM kernel using CUDA Pipeline API best practices

#include <cstdio>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_err));                                         \
      exit(EXIT_FAILURE);                                                        \
    }                                                                             \
  } while (0)
#endif

constexpr int TILE_SIZE = 64; // Fits comfortably within SMEM limits
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int TILE_BYTES = TILE_ELEMS * sizeof(float);

__device__ float computeTile(const float* A_sub, const float* B_sub, int tx, int ty) {
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < TILE_SIZE; ++k) {
    sum += A_sub[ty * TILE_SIZE + k] * B_sub[k * TILE_SIZE + tx];
  }
  return sum;
}

__global__ void gemm_tiled_pipeline(const float* __restrict__ A_global,
                                    const float* __restrict__ B_global,
                                    float* __restrict__ C_global,
                                    int M, int N, int K) {
  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ float shared_mem[];
  float* A_buf[2] = {shared_mem, shared_mem + TILE_ELEMS};
  float* B_buf[2] = {A_buf[1] + TILE_ELEMS, A_buf[1] + 2 * TILE_ELEMS};

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state;
  auto pipe = cuda::make_pipeline(cta, &state);

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int block_row = blockIdx.y * TILE_SIZE;
  int block_col = blockIdx.x * TILE_SIZE;

  float accum = 0.0f;

  int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  // Stage 0: synchronously populate the first tile buffers
  {
    int aRow = block_row + ty;
    int aCol = tx;
    float aval = 0.0f;
    if (aRow < M && aCol < K) {
      aval = A_global[aRow * K + aCol];
    }
    A_buf[0][ty * TILE_SIZE + tx] = aval;

    int bRow = ty;
    int bCol = block_col + tx;
    float bval = 0.0f;
    if (bRow < K && bCol < N) {
      bval = B_global[bRow * N + bCol];
    }
    B_buf[0][ty * TILE_SIZE + tx] = bval;
  }

  cg::sync(cta);

  int curr = 0;
  int next = 1;

  for (int tile = 0; tile < numTiles; ++tile) {
    if (tile + 1 < numTiles) {
      pipe.producer_acquire();

      int kTile = (tile + 1) * TILE_SIZE;

      int aRow = block_row + ty;
      int aCol = kTile + tx;
      float aval = 0.0f;
      if (aRow < M && aCol < K) {
        aval = A_global[aRow * K + aCol];
      }
      A_buf[next][ty * TILE_SIZE + tx] = aval;

      int bRow = kTile + ty;
      int bCol = block_col + tx;
      float bval = 0.0f;
      if (bRow < K && bCol < N) {
        bval = B_global[bRow * N + bCol];
      }
      B_buf[next][ty * TILE_SIZE + tx] = bval;

      pipe.producer_commit();
    }

    pipe.consumer_wait();

    accum += computeTile(A_buf[curr] + ty * TILE_SIZE,
                         B_buf[curr],
                         tx, ty);

    pipe.consumer_release();

    curr = next;
    next = 1 - curr;
  }

  int cRow = block_row + ty;
  int cCol = block_col + tx;
  if (cRow < M && cCol < N) {
    C_global[cRow * N + cCol] = accum;
  }
}

void launch_gemm(const float* d_A,
                 const float* d_B,
                 float* d_C,
                 int M, int N, int K,
                 cudaStream_t stream = 0) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE);
  size_t sharedBytes = 4 * TILE_BYTES; // two buffers for A and B

  gemm_tiled_pipeline<<<grid, block, sharedBytes, stream>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
