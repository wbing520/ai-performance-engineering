#include <cuda_runtime.h>
#include <iostream>
#include <cuda/pipeline>

#define NUM_STREAMS 2
#define TILE_ELEMS 1024

// Blackwell B200/B300 specific optimizations
#if CUDA_VERSION >= 12.9
#define BLACKWELL_OPTIMIZED
#endif

void launch_batched_pipeline(const float* hA, const float* hB, float* hC, int batches) {
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    for (int b = 0; b < batches; ++b) {
        int sid = b % NUM_STREAMS;
        auto s = streams[sid];
        float *dA, *dB, *dC;
        size_t bytes = TILE_ELEMS * sizeof(float);
        
        // Use stream-ordered memory allocation for better performance
        cudaMallocAsync(&dA, bytes, s);
        cudaMallocAsync(&dB, bytes, s);
        cudaMallocAsync(&dC, bytes, s);
        
        cudaMemcpyAsync(dA, hA + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(dB, hB + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, s);
        
        extern void pipelineKernel(float*, float*, float*, int);
        pipelineKernel<<<1, 96, 3 * TILE_ELEMS * sizeof(float), s>>>(dA, dB, dC, 1);
        
        cudaMemcpyAsync(hC + b * TILE_ELEMS, dC, bytes, cudaMemcpyDeviceToHost, s);
        
        // Use stream-ordered deallocation
        cudaFreeAsync(dA, s);
        cudaFreeAsync(dB, s);
        cudaFreeAsync(dC, s);
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    // Check CUDA version and device capabilities
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "CUDA Version: " << CUDA_VERSION << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    return 0;
}
