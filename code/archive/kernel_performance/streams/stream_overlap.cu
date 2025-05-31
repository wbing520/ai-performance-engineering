// stream_overlap.cu
#include <cuda_runtime.h>
#include <iostream>
#define N (1<<20)

__global__ void dummy_compute(float *data) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N) data[idx]*=2.0f;
}

int main(){
    float *h_buf[2], *d_buf[2];
    for(int i=0;i<2;i++){
        cudaMallocHost(&h_buf[i], N*sizeof(float));
        cudaMalloc(&d_buf[i], N*sizeof(float));
        for(int j=0;j<N;j++) h_buf[i][j]=1.0f;
    }
    cudaStream_t st_copy, st_comp;
    cudaStreamCreate(&st_copy);
    cudaStreamCreate(&st_comp);

    for(int i=0;i<4;i++){
        int bi=i%2, pi=(i+1)%2;
        cudaMemcpyAsync(d_buf[bi],h_buf[bi],N*sizeof(float),cudaMemcpyHostToDevice,st_copy);
        if(i>0){
            dummy_compute<<<(N+255)/256,256,0,st_comp>>>(d_buf[pi]);
            cudaMemcpyAsync(h_buf[pi],d_buf[pi],N*sizeof(float),cudaMemcpyDeviceToHost,st_copy);
        }
    }
    dummy_compute<<<(N+255)/256,256,0,st_comp>>>(d_buf[0]);
    cudaMemcpyAsync(h_buf[0],d_buf[0],N*sizeof(float),cudaMemcpyDeviceToHost,st_copy);

    cudaStreamSynchronize(st_copy);
    cudaStreamSynchronize(st_comp);
    std::cout<<"Done"<<std::endl;
    return 0;
}