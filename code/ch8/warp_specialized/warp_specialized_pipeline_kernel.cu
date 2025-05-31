#include <cuda/pipeline>
#include <cooperative_groups>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
#define TILE_SIZE 1024

__device__ float computeTile(const float* data, int lane) {
    return data[lane]*1.0f;
}

__global__ void warp_specialized_pipeline_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int nTiles) {
    cg::thread_block block = cg::this_thread_block();
    extern __shared__ float shared_mem[];
    float* buf = shared_mem;
    cuda::pipeline<3> pipe(block);
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int warp_id = tid>>5, lane = tid&31;
    int totalWarps = (gridDim.x*blockDim.x)>>5;
    for(int t=warp_id; t<nTiles; t+=totalWarps) {
        size_t offset=t*TILE_SIZE;
        if(warp_id==0){
            pipe.producer_acquire(0);
            __pipeline_memcpy_async(buf+lane, A+offset+lane, TILE_SIZE*sizeof(float));
            pipe.producer_commit(0);
        }
        if(warp_id==1){
            pipe.consumer_wait(0);
            pipe.producer_acquire(1);
            float v=computeTile(buf, lane);
            buf[lane]=v;
            pipe.producer_commit(1);
            pipe.consumer_release(0);
        }
        if(warp_id==2){
            pipe.consumer_wait(1);
            pipe.producer_acquire(2);
            C[offset+lane]=buf[lane];
            pipe.producer_commit(2);
            pipe.consumer_release(1);
        }
    }
}

int main(){
    const int nTiles=4;
    float *dA,*dB,*dC;
    cudaMalloc(&dA,nTiles*TILE_SIZE*sizeof(float));
    cudaMalloc(&dB,nTiles*TILE_SIZE*sizeof(float));
    cudaMalloc(&dC,nTiles*TILE_SIZE*sizeof(float));
    warp_specialized_pipeline_kernel<<<1,96, TILE_SIZE*sizeof(float)>>>(dA,dB,dC,nTiles);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}