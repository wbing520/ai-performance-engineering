#include <cuda/pipeline>
#include <cooperative_groups>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
#define TILE_SIZE 128
#define TILE_BYTES (TILE_SIZE*TILE_SIZE*sizeof(float))

__device__ float computeTile(const float* A, const float* B, int tx, int ty) {
    float sum = 0;
    for(int k=0; k<TILE_SIZE; ++k) {
        sum += A[ty*TILE_SIZE + k] * B[k*TILE_SIZE + tx];
    }
    return sum;
}

__global__ void gemm_tiled_pipeline(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ float shared_mem[];
    float* bufA[2] = {shared_mem, shared_mem + TILE_SIZE*TILE_SIZE};
    float* bufB[2] = {bufA[1] + TILE_SIZE*TILE_SIZE, bufA[1] + 2*TILE_SIZE*TILE_SIZE};
    int tx = threadIdx.x, ty = threadIdx.y;
    int thread_id = ty*blockDim.x + tx;
    cuda::pipeline<2> pipe(cta);
    int numTiles = N / TILE_SIZE;
    float acc=0;
    for(int t=0; t<numTiles; ++t) {
        int curr = t&1, next=curr^1;
        if(t < numTiles-1) {
            pipe.producer_acquire(0);
            pipe.memcpy_async(bufA[next]+thread_id, A + ((t+1)*TILE_SIZE)*N + thread_id, TILE_BYTES);
            pipe.memcpy_async(bufB[next]+thread_id, B + ((t+1)*TILE_SIZE) + thread_id*N, TILE_BYTES);
            pipe.producer_commit(0);
        }
        pipe.consumer_wait(0);
        acc += computeTile(bufA[curr], bufB[curr], tx, ty);
        pipe.consumer_release(0);
    }
    int row = blockIdx.y*blockDim.y + ty, col = blockIdx.x*blockDim.x + tx;
    if(row<N && col<N) C[row*N+col]=acc;
}

int main(){
    const int N=512;
    size_t bytes=N*N*sizeof(float);
    float *hA=new float[N*N], *hB=new float[N*N], *hC=new float[N*N];
    float *dA,*dB,*dC;
    cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes); cudaMalloc(&dC,bytes);
    cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice);
    dim3 threads(32,32), grid((N+31)/32,(N+31)/32);
    size_t shmem=2*2*TILE_BYTES;
    gemm_tiled_pipeline<<<grid,threads,shmem>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
