#include <cuda/pipeline>
#include <cuda_runtime.h>

#define TILE_SIZE 1024

__device__ void processTile(const float* tile) {
    // example compute
    // no-op
}

__global__ void kernelWithTMA(const float* __restrict__ global_ptr, int nTiles) {
    __shared__ float tile0[TILE_SIZE], tile1[TILE_SIZE];
    float* tiles[2] = {tile0, tile1};
    auto pipe = cuda::pipeline<cuda::thread_scope::block>();
    size_t bytes = TILE_SIZE * sizeof(float);

    // prime first tile
    pipe.producer_acquire();
    cuda::memcpy_async(pipe, tiles[0], global_ptr, bytes, cuda::pipeline_scope::cta);
    pipe.producer_commit();

    for (int t = 1; t < nTiles; ++t) {
        pipe.consumer_wait();
        processTile(tiles[(t - 1) & 1]);
        pipe.consumer_release();
        pipe.producer_acquire();
        cuda::memcpy_async(pipe, tiles[t & 1], global_ptr + t*TILE_SIZE, bytes, cuda::pipeline_scope::cta);
        pipe.producer_commit();
    }
    pipe.consumer_wait();
    processTile(tiles[(nTiles - 1) & 1]);
    pipe.consumer_release();
}

int main() {
    const int nTiles = 10;
    float* d_ptr; cudaMalloc(&d_ptr, nTiles * TILE_SIZE * sizeof(float));
    kernelWithTMA<<<1, 256>>>(d_ptr, nTiles);
    cudaDeviceSynchronize();
    cudaFree(d_ptr);
    return 0;
}
