#include <cuda_runtime.h>
#include <iostream>

__global__ void myKernel(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

int main() {
    const int N = 1000000;
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float* d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "h_input[0] = " << h_input[0] << std::endl;

    cudaFree(d_input);
    delete[] h_input;
    return 0;
}
