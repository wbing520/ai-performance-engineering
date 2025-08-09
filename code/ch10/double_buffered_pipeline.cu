// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// double_buffered_pipeline.cu
// Two-stage double-buffering example using the CUDA C++ Pipeline API

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

namespace cg = cooperative_groups;

#define TILE_SIZE 128
#define TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(float))

// Per-thread micro-GEMM over one TILE_SIZE×TILE_SIZE sub-block
__device__ float computeTile(const float* A_sub, const float* B_sub, int tx, int ty) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += A_sub[ty * TILE_SIZE + k] * B_sub[k * TILE_SIZE + tx];
    }
    return sum;
}

extern "C"
__global__ void gemm_tiled_pipeline(
    const float* __restrict__ A_global, // [M × K]
    const float* __restrict__ B_global, // [K × N]
    float* __restrict__ C_global,       // [M × N]
    int M, int N, int K)
{
    // 1) Thread-block handle
    cg::thread_block cta = cg::this_thread_block();

    // 2) Shared-memory buffers: 2 for A, 2 for B
    extern __shared__ float shared_mem[];
    float* A_buf[2] = {
        shared_mem,
        shared_mem + TILE_SIZE * TILE_SIZE
    };
    float* B_buf[2] = {
        A_buf[1] + TILE_SIZE * TILE_SIZE,
        A_buf[1] + 2 * TILE_SIZE * TILE_SIZE
    };

    // 3) Two-stage pipeline object
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipelineState;
    auto pipe = cuda::make_pipeline(cta, &pipelineState);

    // 4) Thread coords in the tile
    int tx = threadIdx.x, ty = threadIdx.y;

    // 5) Block's output origin in C
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    // 6) Accumulator for C[sub]
    float accum = 0.0f;

    // 7) How many tiles along K
    int numTiles = K / TILE_SIZE; // assume divisible

    // 8) **Initial synchronous load of tile 0** into buffer 0
    {
        int aRow = block_row + ty;
        int aCol = 0 * TILE_SIZE + tx;
        if (aRow < M && aCol < K) {
            A_buf[0][ty * TILE_SIZE + tx] = A_global[aRow * K + aCol];
        } else {
            A_buf[0][ty * TILE_SIZE + tx] = 0.0f;
        }

        int bRow = 0 * TILE_SIZE + ty;
        int bCol = block_col + tx;
        if (bRow < K && bCol < N) {
            B_buf[0][ty * TILE_SIZE + tx] = B_global[bRow * N + bCol];
        } else {
            B_buf[0][ty * TILE_SIZE + tx] = 0.0f;
        }
    }
    cg::sync(cta);

    // 9) Double-buffered loop with true 2-stage overlap
    int curr = 0, next = 1;
    for (int tile = 0; tile < numTiles; ++tile) {
        // ---- Stage 0: prefetch tile+1 into buffer[next] ----
        if (tile + 1 < numTiles) {
            pipe.producer_acquire();

            // A next-tile
            int aRow = block_row + ty;
            int aCol = (tile + 1) * TILE_SIZE + tx;
            if (aRow < M && aCol < K) {
                cuda::memcpy_async(
                    cta,
                    A_buf[next] + ty * TILE_SIZE + tx,
                    &A_global[aRow * K + aCol],
                    sizeof(float),
                    pipe
                );
            }

            // B next-tile
            int bRow = (tile + 1) * TILE_SIZE + ty;
            int bCol = block_col + tx;
            if (bRow < K && bCol < N) {
                cuda::memcpy_async(
                    cta,
                    B_buf[next] + ty * TILE_SIZE + tx,
                    &B_global[bRow * N + bCol],
                    sizeof(float),
                    pipe
                );
            }

            pipe.producer_commit();
        }

        // ---- Stage 1: compute on buffer[curr] ----
        pipe.consumer_wait();
        accum += computeTile(
            A_buf[curr],
            B_buf[curr],
            tx, ty
        );
        pipe.consumer_release();

        // ---- Swap buffers for next iteration ----
        curr = next;
        next = 1 - curr;
    }

    // 10) Write the final result into C_global
    int cRow = block_row + ty;
    int cCol = block_col + tx;
    if (cRow < M && cCol < N) {
        C_global[cRow * N + cCol] = accum;
    }
}

// Naive tiled GEMM for comparison
__global__ void gemm_tiled_naive(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int M, int N, int K)
{
    extern __shared__ float naive_shared_mem[];
    float* A_tile = naive_shared_mem;
    float* B_tile = naive_shared_mem + TILE_SIZE * TILE_SIZE;

    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    float accum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        // Load A tile
        int aRow = block_row + ty;
        int aCol = tile * TILE_SIZE + tx;
        if (aRow < M && aCol < K) {
            A_tile[ty * TILE_SIZE + tx] = A_global[aRow * K + aCol];
        } else {
            A_tile[ty * TILE_SIZE + tx] = 0.0f;
        }

        // Load B tile
        int bRow = tile * TILE_SIZE + ty;
        int bCol = block_col + tx;
        if (bRow < K && bCol < N) {
            B_tile[ty * TILE_SIZE + tx] = B_global[bRow * N + bCol];
        } else {
            B_tile[ty * TILE_SIZE + tx] = 0.0f;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            accum += A_tile[ty * TILE_SIZE + k] * B_tile[k * TILE_SIZE + tx];
        }

        __syncthreads();
    }

    // Write result
    int cRow = block_row + ty;
    int cCol = block_col + tx;
    if (cRow < M && cCol < N) {
        C_global[cRow * N + cCol] = accum;
    }
}

// Host code
int main(int argc, char** argv) {
    // Matrix dimensions
    int M = 1024, N = 1024, K = 1024;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    printf("=== Double-Buffered Pipeline GEMM ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C_naive = new float[M * N];
    float *h_C_pipeline = new float[M * N];

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_pipeline;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C_naive, M * N * sizeof(float));
    cudaMalloc(&d_C_pipeline, M * N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch parameters
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float); // 4 tiles for pipeline

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark naive version
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        gemm_tiled_naive<<<grid, block, 2 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(
            d_A, d_B, d_C_naive, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    naive_time /= 10.0f; // Average

    // Benchmark pipeline version
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        gemm_tiled_pipeline<<<grid, block, shared_mem_size>>>(
            d_A, d_B, d_C_pipeline, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pipeline_time;
    cudaEventElapsedTime(&pipeline_time, start, stop);
    pipeline_time /= 10.0f; // Average

    // Copy results back
    cudaMemcpy(h_C_naive, d_C_naive, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_pipeline, d_C_pipeline, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C_naive[i] - h_C_pipeline[i]);
        if (diff > max_diff) max_diff = diff;
    }

    // Calculate performance metrics
    double flops = 2.0 * M * N * K; // multiply-add operations
    double naive_gflops = flops / (naive_time * 1e6);
    double pipeline_gflops = flops / (pipeline_time * 1e6);

    printf("\n=== Results ===\n");
    printf("Naive tiled GEMM:     %.2f ms (%.1f GFLOPS)\n", naive_time, naive_gflops);
    printf("Pipeline GEMM:        %.2f ms (%.1f GFLOPS)\n", pipeline_time, pipeline_gflops);
    printf("Speedup:              %.2fx\n", naive_time / pipeline_time);
    printf("Max difference:       %.2e\n", max_diff);

    printf("\n=== Profiling Commands ===\n");
    printf("ncu --section MemoryWorkloadAnalysis ./double_buffered_pipeline %d %d %d\n", M, N, K);
    printf("nsys profile --force-overwrite=true -o pipeline ./double_buffered_pipeline %d %d %d\n", M, N, K);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_pipeline;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_pipeline);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
