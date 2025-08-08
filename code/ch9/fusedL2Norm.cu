// fusedL2Norm.cu
// Fused L2 normalization kernel from Chapter 9

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Non-fused version (two separate kernels)
__global__ void computeNorms(const float* x, float* norms, int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x[batch_idx * hidden_size + i];
        sum += val * val;
    }
    
    // Reduce within block
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        norms[batch_idx] = sqrtf(sdata[0]);
    }
}

__global__ void normalizeVector(const float* x, const float* norms, float* output, 
                               int batch_size, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;
    
    int batch_idx = idx / hidden_size;
    float norm = norms[batch_idx];
    output[idx] = (norm > 1e-8f) ? x[idx] / norm : 0.0f;
}

// Fused version (single kernel)
__global__ void fusedL2Norm(const float* x, float* output, int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    
    // Phase 1: Compute L2 norm
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x[batch_idx * hidden_size + i];
        sum += val * val;
    }
    
    // Reduce within block
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Broadcast norm to all threads
    float norm = sqrtf(sdata[0]);
    __syncthreads();
    
    // Phase 2: Normalize and write output
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int idx = batch_idx * hidden_size + i;
        output[idx] = (norm > 1e-8f) ? x[idx] / norm : 0.0f;
    }
}

// Optimized fused version with better memory coalescing
__global__ void fusedL2NormOptimized(const float* __restrict__ x, float* __restrict__ output, 
                                     int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Phase 1: Compute L2 norm with optimized access pattern
    float local_sum = 0.0f;
    const float* batch_ptr = x + batch_idx * hidden_size;
    
    // Vectorized loads when possible
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = batch_ptr[i];
        local_sum += val * val;
    }
    
    // Optimized reduction
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Unroll the reduction for better performance
    if (block_size >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (block_size >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (block_size >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    
    // Final warp reduction
    if (tid < 32) {
        volatile float* vdata = sdata;
        if (block_size >= 64) vdata[tid] += vdata[tid + 32];
        if (block_size >= 32) vdata[tid] += vdata[tid + 16];
        if (block_size >= 16) vdata[tid] += vdata[tid + 8];
        if (block_size >= 8)  vdata[tid] += vdata[tid + 4];
        if (block_size >= 4)  vdata[tid] += vdata[tid + 2];
        if (block_size >= 2)  vdata[tid] += vdata[tid + 1];
    }
    
    // Broadcast norm
    float norm = (tid == 0) ? sqrtf(sdata[0]) : 0.0f;
    norm = __shfl_sync(0xffffffff, norm, 0);
    
    // Phase 2: Normalize with coalesced writes
    float* output_batch_ptr = output + batch_idx * hidden_size;
    for (int i = tid; i < hidden_size; i += block_size) {
        output_batch_ptr[i] = (norm > 1e-8f) ? batch_ptr[i] / norm : 0.0f;
    }
}

void runBenchmark(int batch_size, int hidden_size, int iterations) {
    // Allocate memory
    size_t input_size = batch_size * hidden_size * sizeof(float);
    size_t norm_size = batch_size * sizeof(float);
    
    float *h_input = new float[batch_size * hidden_size];
    float *h_output = new float[batch_size * hidden_size];
    
    // Initialize input
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        h_input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    float *d_input, *d_output, *d_norms;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_norms, norm_size);
    
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    
    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threads = 256;
    int shared_mem = threads * sizeof(float);
    
    printf("=== L2 Normalization Benchmark ===\n");
    printf("Batch size: %d, Hidden size: %d\n", batch_size, hidden_size);
    printf("Iterations: %d\n\n", iterations);
    
    // Non-fused version
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        computeNorms<<<batch_size, threads, shared_mem>>>(d_input, d_norms, batch_size, hidden_size);
        int norm_threads = 256;
        int norm_blocks = (batch_size * hidden_size + norm_threads - 1) / norm_threads;
        normalizeVector<<<norm_blocks, norm_threads>>>(d_input, d_norms, d_output, batch_size, hidden_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float non_fused_time;
    cudaEventElapsedTime(&non_fused_time, start, stop);
    
    // Fused version
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        fusedL2Norm<<<batch_size, threads, shared_mem>>>(d_input, d_output, batch_size, hidden_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fused_time;
    cudaEventElapsedTime(&fused_time, start, stop);
    
    // Optimized fused version
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        fusedL2NormOptimized<<<batch_size, threads, shared_mem>>>(d_input, d_output, batch_size, hidden_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float optimized_time;
    cudaEventElapsedTime(&optimized_time, start, stop);
    
    // Verify correctness
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);
    
    float sample_norm = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sample_norm += h_output[i] * h_output[i];
    }
    sample_norm = sqrtf(sample_norm);
    
    printf("Results:\n");
    printf("Non-fused time:     %.2f ms (avg: %.3f ms)\n", non_fused_time, non_fused_time / iterations);
    printf("Fused time:         %.2f ms (avg: %.3f ms) - %.1fx speedup\n", 
           fused_time, fused_time / iterations, non_fused_time / fused_time);
    printf("Optimized time:     %.2f ms (avg: %.3f ms) - %.1fx speedup\n", 
           optimized_time, optimized_time / iterations, non_fused_time / optimized_time);
    printf("Sample L2 norm:     %.6f (should be ~1.0)\n", sample_norm);
    
    printf("\nProfiling commands:\n");
    printf("ncu --section MemoryWorkloadAnalysis ./fusedL2Norm\n");
    printf("nsys profile --force-overwrite=true -o l2norm ./fusedL2Norm\n");
    
    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_norms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int batch_size = 128;
    int hidden_size = 4096;
    int iterations = 100;
    
    if (argc > 1) batch_size = atoi(argv[1]);
    if (argc > 2) hidden_size = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    
    runBenchmark(batch_size, hidden_size, iterations);
    
    return 0;
}
