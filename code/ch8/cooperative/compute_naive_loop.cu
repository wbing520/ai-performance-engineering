#include <cuda_runtime.h>
#include <iostream>
__global__ void computeStep(float* data,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) data[idx]=data[idx]*0.5f+1.0f;
}
int main(){
    const int N=1024;
    float* d; cudaMalloc(&d,N*sizeof(float));
    for(int i=0;i<1000;++i){
        computeStep<<<(N+255)/256,256>>>(d,N);
        cudaDeviceSynchronize();
    }
    cudaFree(d);
    return 0;
}