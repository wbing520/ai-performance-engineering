// pipeline_example.cu
// Chapter 9: Example demonstrating CUDA Pipeline API for TMA transfers

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <iostream>
#include <vector>

#define TILE_SIZE 128

__global__ void pipeline_example_kernel(const float* globalA, float* globalB, int numTiles) {
    // Declare shared pipeline state in __shared__ memory
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_device, 2> state;
    
    // Create the pipeline with modern API
    auto pipe = cuda::make_pipeline(cuda::this_thread_group(), &state);
    
    // Shared memory buffers for double buffering
    __shared__ float tileA0[TILE_SIZE][TILE_SIZE];
    __shared__ float tileA1[TILE_SIZE][TILE_SIZE];
    
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int tile = 0;
    size_t offset = 0;
    int buf = 0;
    
    // Initial async copy
    if (tile < numTiles) {
        size_t tileBytes = TILE_SIZE * TILE_SIZE * sizeof(float);
        pipe.async_memcpy(&tileA0[0][0], globalA + offset, tileBytes);
        pipe.commit();
        offset += TILE_SIZE * TILE_SIZE;
        tile++;
    }
    
    while (tile <= numTiles) {
        float (*curA)[TILE_SIZE] = (buf == 0 ? tileA0 : tileA1);
        float (*nxtA)[TILE_SIZE] = (buf == 0 ? tileA1 : tileA0);
        
        // Wait for current tile to be ready
        pipe.wait();
        
        // Compute on current tile (example: simple copy with transformation)
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            float val = curA[ty][tx];
            // Example computation: square the value
            val = val * val;
            curA[ty][tx] = val;
        }
        __syncthreads();
        
        // Launch async copy of next tile (if available)
        if (tile < numTiles) {
            size_t tileBytes = TILE_SIZE * TILE_SIZE * sizeof(float);
            pipe.async_memcpy(&nxtA[0][0], globalA + offset, tileBytes);
            pipe.commit();
            offset += TILE_SIZE * TILE_SIZE;
        }
        
        // Write processed tile back to global memory
        if (tx < TILE_SIZE && ty < TILE_SIZE) {
            size_t global_offset = (tile - 1) * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
            globalB[global_offset] = curA[ty][tx];
        }
        
        buf ^= 1;
        tile++;
    }
    
    // Wait for last copy to complete
    pipe.wait();
}

int main() {
    const int NUM_TILES = 8;
    const int TOTAL_ELEMENTS = NUM_TILES * TILE_SIZE * TILE_SIZE;
    
    // Allocate host memory
    std::vector<float> h_input(TOTAL_ELEMENTS);
    std::vector<float> h_output(TOTAL_ELEMENTS);
    
    // Initialize input data
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        h_input[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with tile dimensions
    dim3 blockSize(16, 16); // 256 threads total
    dim3 gridSize(1, 1);
    
    // Warm up
    pipeline_example_kernel<<<gridSize, blockSize>>>(d_input, d_output, NUM_TILES);
    cudaDeviceSynchronize();
    
    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    pipeline_example_kernel<<<gridSize, blockSize>>>(d_input, d_output, NUM_TILES);
    cudaEventRecord(stop);
    
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Pipeline kernel time: " << milliseconds << " ms" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results (should be squared input values)
    bool correct = true;
    for (int i = 0; i < TOTAL_ELEMENTS && correct; i++) {
        float expected = h_input[i] * h_input[i];
        if (std::abs(h_output[i] - expected) > 1e-6) {
            std::cout << "Mismatch at index " << i << ": got " << h_output[i] 
                      << ", expected " << expected << std::endl;
            correct = false;
        }
    }
    
    std::cout << "Results: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
