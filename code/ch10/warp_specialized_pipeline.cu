// warp_specialized_pipeline.cu
// Warp-specialized persistent kernel corrected to follow CUDA Pipeline best practices.
// Target architecture: CUDA 12.9+, Blackwell (sm_100).

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cg = cooperative_groups;

#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t _err = (expr);                                                    \
    if (_err != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(_err));                                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                             \
  } while (0)
#endif

constexpr int TILE_SIZE = 128;               // Shared memory footprint: 3 * 128^2 * 4 = 196608 bytes
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr size_t TILE_BYTES = TILE_ELEMS * sizeof(float);

// Simple fused computation on a tile: element-wise sum of squares followed by sqrt.
__device__ void compute_tile(float* C_tile,
                             const float* A_tile,
                             const float* B_tile,
                             int lane_id) {
  for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
    float a = A_tile[idx];
    float b = B_tile[idx];
    C_tile[idx] = sqrtf(a * a + b * b);
  }
}

extern "C" __global__ void warp_specialized_pipeline_kernel(const float* __restrict__ A_global,
                                                             const float* __restrict__ B_global,
                                                             float* __restrict__ C_global,
                                                             int tiles_per_matrix) {
  cg::thread_block cta = cg::this_thread_block();

  extern __shared__ float shared_mem[];
  float* A_tile = shared_mem;                     // [0 .. TILE_ELEMS)
  float* B_tile = A_tile + TILE_ELEMS;            // [TILE_ELEMS .. 2*TILE_ELEMS)
  float* C_tile = B_tile + TILE_ELEMS;            // [2*TILE_ELEMS .. 3*TILE_ELEMS)

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipe_state;
  auto pipe = cuda::make_pipeline(cta, &pipe_state);

  const int warp_id = threadIdx.x >> 5;           // warp within the block
  const int lane_id = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int total_warps = gridDim.x * warps_per_block;
  const int global_warp = warp_id + warps_per_block * blockIdx.x;

  for (int tile = global_warp; tile < tiles_per_matrix; tile += total_warps) {
    const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

    if (warp_id == 0) {  // loader warp
      pipe.producer_acquire();
      for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        A_tile[idx] = A_global[offset + idx];
        B_tile[idx] = B_global[offset + idx];
      }
      pipe.producer_commit();
    }

    if (warp_id == 1) {  // compute warp
      pipe.consumer_wait();
      compute_tile(C_tile, A_tile, B_tile, lane_id);
      pipe.consumer_release();
    }

    if (warp_id == 2) {  // storer warp
      pipe.consumer_wait();
      for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        C_global[offset + idx] = C_tile[idx];
      }
      pipe.consumer_release();
    }

    cta.sync();
  }
}

static void reference_compute(std::vector<float>& C,
                              const std::vector<float>& A,
                              const std::vector<float>& B) {
  for (size_t i = 0; i < C.size(); ++i) {
    C[i] = std::sqrt(A[i] * A[i] + B[i] * B[i]);
  }
}

int main(int argc, char** argv) {
  int tiles = 1024;
  if (argc > 1) tiles = std::max(1, std::atoi(argv[1]));

  const size_t elems = static_cast<size_t>(tiles) * TILE_ELEMS;
  const size_t bytes = elems * sizeof(float);

  std::vector<float> h_A(elems), h_B(elems), h_C(elems), h_ref(elems);
  std::iota(h_A.begin(), h_A.end(), 0.0f);
  std::iota(h_B.begin(), h_B.end(), 1.0f);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(96); // three warps: loader, compute, storer
  dim3 grid(std::min(tiles, 256));
  size_t shared_bytes = 3 * TILE_BYTES;

  warp_specialized_pipeline_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

  reference_compute(h_ref, h_A, h_B);

  double max_err = 0.0;
  for (size_t i = 0; i < elems; ++i) {
    max_err = std::max(max_err, std::abs(h_C[i] - h_ref[i]));
  }
  printf("Max error: %g\n", max_err);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return (max_err < 1e-3) ? EXIT_SUCCESS : EXIT_FAILURE;
}
