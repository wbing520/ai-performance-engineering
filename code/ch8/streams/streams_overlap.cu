#include <cuda_runtime.h>
#include <iostream>
__global__ void computeKernel(float* data,float* out,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out[idx]=data[idx]*2.0f;
}
int main(){
    const int N=1<<20;
    float *hA=new float[N], *hB=new float[N], *hC=new float[N], *hD=new float[N];
    for(int i=0;i<N;++i) hA[i]=i;
    float *dA,*dB,*dC,*dD;
    cudaStream_t s1,s2;
    cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cudaMallocAsync(&dA,N*sizeof(float),s1);
    cudaMallocAsync(&dC,N*sizeof(float),s1);
    cudaMallocAsync(&dB,N*sizeof(float),s2);
    cudaMallocAsync(&dD,N*sizeof(float),s2);
    cudaMemcpyAsync(dA,hA,N*sizeof(float),cudaMemcpyHostToDevice,s1);
    computeKernel<<<(N+255)/256,256,0,s1>>>(dA,dC,N);
    cudaMemcpyAsync(hC,dC,N*sizeof(float),cudaMemcpyDeviceToHost,s1);
    cudaMemcpyAsync(dB,hB,N*sizeof(float),cudaMemcpyHostToDevice,s2);
    computeKernel<<<(N+255)/256,256,0,s2>>>(dB,dD,N);
    cudaMemcpyAsync(hD,dD,N*sizeof(float),cudaMemcpyDeviceToHost,s2);
    cudaStreamSynchronize(s1); cudaStreamSynchronize(s2);
    cudaFreeAsync(dA,s1); cudaFreeAsync(dC,s1);
    cudaFreeAsync(dB,s2); cudaFreeAsync(dD,s2);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    delete[] hA; delete[] hB; delete[] hC; delete[] hD;
    return 0;
}