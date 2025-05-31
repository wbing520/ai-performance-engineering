// add_naive.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_naive(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<N) C[idx]=A[idx]+B[idx];
}

int main(){
    const int N=1<<20;
    float *h_A=new float[N],*h_B=new float[N],*h_C=new float[N];
    for(int i=0;i<N;i++){h_A[i]=1;h_B[i]=2;}
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,N*sizeof(float));
    cudaMalloc(&d_B,N*sizeof(float));
    cudaMalloc(&d_C,N*sizeof(float));
    cudaMemcpy(d_A,h_A,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*sizeof(float),cudaMemcpyHostToDevice);

    add_naive<<<(N+255)/256,256>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,N*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"C[0]="<<h_C[0]<<"
";
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
    delete[] h_A;delete[] h_B;delete[] h_C;
    return 0;
}