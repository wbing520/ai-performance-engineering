// reduce_naive.cu
// Hardware: GB200/H100, CUDA 13.0, C++17, Python 3.11, OpenAI Triton 2.5.0
#include <cuda_runtime.h>
#include <iostream>

__global__ void reduce_naive(const float* data, float* out, int N) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for(int i=tid;i<N;i+=blockDim.x) sum += data[i];
    smem[tid] = sum;
    __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1) {
        if(tid<s) smem[tid]+=smem[tid+s];
        __syncthreads();
    }
    if(tid==0) out[blockIdx.x]=smem[0];
}

int main(){
    const int N = 1<<20;
    float *h_data=new float[N];
    for(int i=0;i<N;i++) h_data[i]=1.0f;
    float *d_data,*d_out;
    cudaMalloc(&d_data,N*sizeof(float));
    cudaMalloc(&d_out,sizeof(float));
    cudaMemcpy(d_data,h_data,N*sizeof(float),cudaMemcpyHostToDevice);

    reduce_naive<<<1,256>>>(d_data,d_out,N);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result,d_out,sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<"Sum = "<<result<<"
";
    cudaFree(d_data);cudaFree(d_out);
    delete[] h_data;
    return 0;
}