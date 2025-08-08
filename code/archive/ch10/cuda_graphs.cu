#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorMultiply(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    const int N = 1'000'000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, N * sizeof(float));
    cudaMallocHost(&h_b, N * sizeof(float));
    cudaMallocHost(&h_c, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c, *d_temp;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Regular execution (not using graphs)
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_temp, N);
        vectorMultiply<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_temp, d_b, d_c, N);
    }
    
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Create CUDA Graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, d_temp, N);
    vectorMultiply<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_temp, d_b, d_c, N);
    
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // Graph execution
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        cudaGraphLaunch(graphExec, stream);
    }
    
    cudaStreamSynchronize(stream);
    end = std::chrono::high_resolution_clock::now();
    auto graph_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "CUDA Graph created successfully" << std::endl;
    std::cout << "Graph execution time: " << graph_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Regular execution time: " << regular_time.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Speedup: " << (double)regular_time.count() / graph_time.count() << "x" << std::endl;

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_temp);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}
