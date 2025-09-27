// Architecture-specific optimizations for CUDA 12.9
// Targets Blackwell B200/B300 (sm_100)
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define TILE_DIM 32
#define PAD 1 // padding columns to avoid bank conflicts

__global__ void transposePadded(const float *idata, float *odata, int width) {
    // Each row is TILE_DIM+1 elements to shift bank mapping
    __shared__ float tile[TILE_DIM][TILE_DIM + PAD];
    cg::thread_block block = cg::this_thread_block();
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    tile[threadIdx.x][threadIdx.y] = idata[y * width + x];
    
    block.sync();
    
    odata[x * width + y] = tile[threadIdx.y][threadIdx.x];
}

int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);
    
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);
    
    // Initialize input data
    for (int i = 0; i < N * N; ++i) {
        h_idata[i] = static_cast<float>(i);
    }
    
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    
    transposePadded<<<grid, block>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();
    
    // Copy result back and verify (optional)
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    
    return 0;
}
