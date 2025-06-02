#include <cuda/pipeline>
#include <cooperative_groups>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
#define TILE_SIZE 1024

__global__ void pipelineKernel(float* A, float* B, float* C, int nTiles) {
    extern __shared__ float mem[];
    float* a = mem;
    float* b = a + TILE_SIZE;
    float* c = b + TILE_SIZE;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    cuda::pipeline<3> pipe(cg::this_thread_block());
    for (int t = 0; t < nTiles; ++t) {
        if (warp == 0) {
            __pipeline_memcpy_async(a + lane, A + lane, TILE_SIZE * sizeof(float));
            pipe.producer_commit(0);
        }
        if (warp == 1) {
            pipe.consumer_wait(0);
            c[lane] = a[lane] + b[lane];
            pipe.producer_commit(1);
            pipe.consumer_release(0);
        }
        if (warp == 2) {
            pipe.consumer_wait(1);
            C[lane] = c[lane];
            pipe.producer_commit(2);
        }
    }
}
