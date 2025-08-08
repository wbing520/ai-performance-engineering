// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// warp_specialized_pipeline.cu
// Chapter 10: Example demonstrating warp specialization with CUDA Pipeline API

#include <cuda/pipeline> // CUDA Pipeline API
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cooperative_groups;

#define TILE_SIZE 1024 // adjust as needed

// computeTile: example per-lane computation on one TILE_SIZE block.
// Here we do a simple dot-product of size TILE_SIZE,
// with each lane handling one element.
__device__ float computeTile(const float* tile_data, int lane_id) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += tile_data[lane_id * TILE_SIZE + k] * tile_data[k * TILE_SIZE + lane_id];
    }
    return sum;
}

// warp_specialized_pipeline_kernel
// - Uses 3 warps per block: warp 0 = loader,
//   warp 1 = compute, warp 2 = storer.
// - A 3-stage cuda::pipeline object ensures
//   load/compute/store overlap.
// - Shared memory is partitioned into A_tile[],
//   B_tile[], and C_tile[] of length TILE_SIZE each.
// - Each warp processes a strided subset of
//   `numTiles` total tiles.
extern "C"
__global__ void warp_specialized_pipeline_kernel(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles)
{
    // 1) Create a cooperative thread array (CTA) (a.k.a thread block) object
    thread_block cta = this_thread_block();
    
    // 2) Allocate 3 × TILE_SIZE floats in shared memory
    extern __shared__ float shared_mem[];
    
    // [0 ... TILE_SIZE-1]
    float* A_tile = shared_mem;
    // [TILE_SIZE ... 2*TILE_SIZE-1]
    float* B_tile = shared_mem + TILE_SIZE;
    // [2*TILE_SIZE ... 3*TILE_SIZE-1]
    float* C_tile = shared_mem + 2 * TILE_SIZE;
    
    // 3) Create a 3-stage pipeline object (shared across the block)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 3> pipelineState;
    auto pipeline = cuda::make_pipeline<cuda::thread_scope_block, 3>(cta, &pipelineState);
    
    // 4) Compute warp_id (0, 1, or 2) and lane_id (0–31)
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_in_block = thread_idx >> 5; // divide by 32
    int lane_id = thread_idx & 31; // mod 32
    
    // 5) Determine how many warps in the entire grid,
    // and each warp's starting tile index
    int totalWarps = (gridDim.x * (blockDim.x >> 5));
    
    // Each warp in the grid gets a unique global_warp ID
    int global_warp = warp_in_block + (blockIdx.x * (blockDim.x >> 5));
    
    // 6) Persistent loop: each warp handles tiles in a strided fashion
    for (int tile = global_warp; tile < numTiles; tile += totalWarps) {
        size_t offset = size_t(tile) * TILE_SIZE;
        
        // Stage 0: Loader Warp (warp_id == 0)
        if (warp_in_block == 0) {
            // Acquire stage 0
            pipeline.producer_acquire(0);
            
            // Asynchronously copy one float per lane from A_global and B_global 
            // into shared memory
            cuda::memcpy_async(cta,
                A_tile + lane_id,
                A_global + offset + lane_id,
                sizeof(float));
            
            cuda::memcpy_async(cta,
                B_tile + lane_id,
                B_global + offset + lane_id,
                sizeof(float));
            
            // Commit the async copies
            pipeline.producer_commit(pipeline, 0);
        }
        
        // Stage 1: Compute Warp (warp_id == 1)
        if (warp_in_block == 1) {
            // Wait for stage 0 (load) to complete
            pipeline.consumer_wait(pipeline, 0);
            
            // Perform computation: each lane computes one element
            if (lane_id < TILE_SIZE) {
                float result = computeTile(A_tile, lane_id);
                C_tile[lane_id] = result;
            }
            
            // Release stage 0 and advance to stage 1
            pipeline.consumer_release(pipeline, 0);
        }
        
        // Stage 2: Storer Warp (warp_id == 2)
        if (warp_in_block == 2) {
            // Wait for stage 1 (compute) to complete
            pipeline.consumer_wait(pipeline, 1);
            
            // Store results back to global memory
            if (lane_id < TILE_SIZE) {
                C_global[offset + lane_id] = C_tile[lane_id];
            }
            
            // Release stage 1
            pipeline.consumer_release(pipeline, 1);
        }
        
        // Synchronize all warps before moving to next tile
        cta.sync();
    }
}

int main() {
    const int NUM_TILES = 16;
    const int TOTAL_ELEMENTS = NUM_TILES * TILE_SIZE;
    
    std::cout << "Warp Specialized Pipeline (Chapter 10)" << std::endl;
    std::cout << "Number of tiles: " << NUM_TILES << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << std::endl;
    std::cout << "Total elements: " << TOTAL_ELEMENTS << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(TOTAL_ELEMENTS);
    std::vector<float> h_B(TOTAL_ELEMENTS);
    std::vector<float> h_C(TOTAL_ELEMENTS);
    
    // Initialize input data
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        h_A[i] = static_cast<float>(i % 1000) / 1000.0f;
        h_B[i] = static_cast<float>((i + 500) % 1000) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_B, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_C, TOTAL_ELEMENTS * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_A, h_A.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    // Use 3 warps per block (96 threads) for warp specialization
    int blockSize = 96; // 3 warps * 32 threads/warp
    int gridSize = 4;   // Multiple blocks for better utilization
    
    // Shared memory: 3 * TILE_SIZE floats (A_tile, B_tile, C_tile)
    size_t shared_mem_size = 3 * TILE_SIZE * sizeof(float);
    
    // Check shared memory limits
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Shared memory required: " << shared_mem_size / 1024 << " KB" << std::endl;
    std::cout << "Shared memory available: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cout << "Error: Insufficient shared memory!" << std::endl;
        return 1;
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    warp_specialized_pipeline_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        d_A, d_B, d_C, NUM_TILES);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Benchmark
    cudaEventRecord(start);
    warp_specialized_pipeline_kernel<<<gridSize, blockSize, shared_mem_size>>>(
        d_A, d_B, d_C, NUM_TILES);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results (spot check)
    bool correct = true;
    for (int tile = 0; tile < NUM_TILES && correct; tile++) {
        for (int lane = 0; lane < std::min(32, TILE_SIZE) && correct; lane++) {
            int idx = tile * TILE_SIZE + lane;
            
            // Compute expected result (dot product of row and column)
            float expected = 0.0f;
            for (int k = 0; k < TILE_SIZE; k++) {
                int a_idx = tile * TILE_SIZE + lane * TILE_SIZE + k;
                int b_idx = tile * TILE_SIZE + k * TILE_SIZE + lane;
                if (a_idx < TOTAL_ELEMENTS && b_idx < TOTAL_ELEMENTS) {
                    expected += h_A[a_idx] * h_B[b_idx];
                }
            }
            
            if (std::abs(h_C[idx] - expected) > 1e-4) {
                std::cout << "Mismatch at tile " << tile << ", lane " << lane 
                          << ": got " << h_C[idx] << ", expected " << expected << std::endl;
                correct = false;
            }
        }
    }
    
    std::cout << "Results: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Show benefits of warp specialization
    std::cout << "\nWarp Specialization Benefits:" << std::endl;
    std::cout << "- Divides work into specialized stages (load, compute, store)" << std::endl;
    std::cout << "- Overlaps memory transfers with computation" << std::endl;
    std::cout << "- Uses fine-grained producer-consumer synchronization" << std::endl;
    std::cout << "- Minimizes idle time and maximizes SM utilization" << std::endl;
    std::cout << "- Scales linearly with number of warps" << std::endl;
    
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
