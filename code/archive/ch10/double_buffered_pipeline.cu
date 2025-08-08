// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// double_buffered_pipeline.cu
// Chapter 10: Example demonstrating double-buffered pipeline using CUDA Pipeline API

#include <cuda/pipeline>
#include <cooperative_groups>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace cg = cooperative_groups;

#define TILE_SIZE 128
#define TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(float))

// Per-thread micro‑GEMM over one TILE_SIZE×TILE_SIZE sub-block.
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
    
    // 2) Shared‑memory buffers: 2 for A, 2 for B
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
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, 2> state;
    auto pipe = cuda::make_pipeline(cta, &state);
    
    // 4) Thread coords in the tile
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // 5) Block's output origin in C
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    
    // 6) Accumulator for C[sub]
    float accum = 0.0f;
    
    // 7) How many tiles along K
    int numTiles = K / TILE_SIZE; // assume divisible
    
    // 8) Issue first async copy
    if (numTiles > 0) {
        // A tile: from A_global[block_row:block_row+TILE_SIZE, 0:TILE_SIZE]
        // B tile: from B_global[0:TILE_SIZE, block_col:block_col+TILE_SIZE]
        
        // Simplified: each thread copies one element
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            pipe.async_memcpy(cta, 
                A_buf[0] + ty * TILE_SIZE + tx,
                A_global + (block_row + ty) * K + tx,
                sizeof(float));
            
            pipe.async_memcpy(cta,
                B_buf[0] + ty * TILE_SIZE + tx,
                B_global + ty * N + (block_col + tx),
                sizeof(float));
        }
        pipe.commit();
    }
    
    // 9) Main loop: overlap loads with compute
    for (int tile = 0; tile < numTiles; ++tile) {
        int cur_buf = tile % 2;
        int nxt_buf = 1 - cur_buf;
        
        // Issue next async copy (if not the last tile)
        if (tile + 1 < numTiles) {
            int k_offset = (tile + 1) * TILE_SIZE;
            
            if (tx < TILE_SIZE && ty < TILE_SIZE) {
                pipe.async_memcpy(cta,
                    A_buf[nxt_buf] + ty * TILE_SIZE + tx,
                    A_global + (block_row + ty) * K + (k_offset + tx),
                    sizeof(float));
                
                pipe.async_memcpy(cta,
                    B_buf[nxt_buf] + ty * TILE_SIZE + tx,
                    B_global + (k_offset + ty) * N + (block_col + tx),
                    sizeof(float));
            }
            pipe.commit();
        }
        
        // Wait for current tile to arrive
        pipe.wait();
        
        // Compute on current tile
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            accum += computeTile(A_buf[cur_buf], B_buf[cur_buf], tx, ty);
        }
    }
    
    // 10) Store result
    if (tx < TILE_SIZE && ty < TILE_SIZE &&
        (block_row + ty) < M && (block_col + tx) < N) {
        C_global[(block_row + ty) * N + (block_col + tx)] = accum;
    }
}

int main() {
    const int M = 512, N = 512, K = 512;
    
    std::cout << "Double-Buffered Pipeline GEMM (Chapter 10)" << std::endl;
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 blockSize(16, 16); // TILE_SIZE must be >= 16 for this to work
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Calculate shared memory requirements
    size_t shared_mem_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float); // 2 A buffers + 2 B buffers
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    gemm_tiled_pipeline<<<gridSize, blockSize, shared_mem_size>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Benchmark
    cudaEventRecord(start);
    gemm_tiled_pipeline<<<gridSize, blockSize, shared_mem_size>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance
    double flops = 2.0 * M * N * K;
    double gflops = flops / (milliseconds * 1e6);
    
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;
    
    // Copy result back and verify (simplified verification)
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Simple verification: check if results are reasonable
    bool reasonable = true;
    for (int i = 0; i < std::min(100, M * N); i++) {
        if (std::isnan(h_C[i]) || std::isinf(h_C[i]) || h_C[i] < 0) {
            reasonable = false;
            break;
        }
    }
    
    std::cout << "Results: " << (reasonable ? "REASONABLE" : "UNREASONABLE") << std::endl;
    
    // Show benefits of double buffering
    std::cout << "\nDouble-buffering benefits:" << std::endl;
    std::cout << "- Overlaps memory transfers with computation" << std::endl;
    std::cout << "- Hides DRAM latency behind arithmetic operations" << std::endl;
    std::cout << "- Uses CUDA Pipeline API for fine-grained synchronization" << std::endl;
    std::cout << "- Achieves better SM utilization and higher throughput" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// CUDA 12.9 Stream-ordered Memory Allocation Example
__global__ void stream_ordered_memory_example() {
    // Example of stream-ordered memory allocation
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your kernel code here
}

// CUDA 12.9 TMA (Tensor Memory Accelerator) Example
__global__ void tma_example() {
    // Example of TMA usage for Blackwell B200/B300
    // This is a placeholder for actual implementation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Your TMA code here
}
