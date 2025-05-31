#include <cuda_runtime.h>
#define TILE_DIM 32

__global__ void transposeNaive(const float *idata, float *odata, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();
    odata[x * width + y] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);
    float *h_idata = (float*)malloc(size), *h_odata = (float*)malloc(size);
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, size); cudaMalloc(&d_odata, size);
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    dim3 block(TILE_DIM, TILE_DIM), grid(N/TILE_DIM, N/TILE_DIM);
    transposeNaive<<<grid, block>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();
    cudaFree(d_idata); cudaFree(d_odata);
    free(h_idata); free(h_odata);
    return 0;
}
