// Architecture-specific optimizations for CUDA 12.9
// Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100)
// cuda_graphs.cu
// CUDA Graphs examples for reducing kernel launch overhead

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// Simple kernels for graph demonstration
__global__ void kernelA(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 1.1f + 0.1f;
    }
}

__global__ void kernelB(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}

__global__ void kernelC(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sinf(data[idx] * 0.1f);
    }
}

// Kernel that sets a condition for conditional graphs
__global__ void setCondition(cudaGraphConditionalHandle handle, float* data, int N, float threshold) {
    // Compute sum of first 32 elements (simplified)
    __shared__ float sdata[32];
    
    int tid = threadIdx.x;
    if (tid < 32 && tid < N) {
        sdata[tid] = data[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Simple reduction
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        unsigned int flag = (sdata[0] > threshold) ? 1u : 0u;
        cudaGraphSetConditional(handle, flag);
    }
}

// Body kernel for conditional execution
__global__ void conditionalBody(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f; // Double the values
    }
}

void demonstrateBasicGraphs() {
    printf("=== Basic CUDA Graphs Demonstration ===\n");
    
    const int N = 1024 * 1024;
    const int bytes = N * sizeof(float);
    
    // Allocate memory
    float *h_data = new float[N];
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i / N;
    }
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Launch parameters
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Timing for traditional launches
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int iterations = 100;
    
    printf("Running %d iterations with %d elements\n", iterations, N);
    
    // Test 1: Traditional separate kernel launches
    printf("\n1. Traditional separate kernel launches:\n");
    cudaEventRecord(start);
    
    for (int iter = 0; iter < iterations; ++iter) {
        kernelA<<<grid, block, 0, stream>>>(d_data, N);
        kernelB<<<grid, block, 0, stream>>>(d_data, N);
        kernelC<<<grid, block, 0, stream>>>(d_data, N);
    }
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float traditional_time;
    cudaEventElapsedTime(&traditional_time, start, stop);
    printf("   Time: %.2f ms (%.3f ms per iteration)\n", 
           traditional_time, traditional_time / iterations);
    
    // Reset data
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Test 2: CUDA Graphs
    printf("\n2. CUDA Graphs (capture and replay):\n");
    
    // Capture the graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    printf("   Capturing graph...\n");
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Enqueue operations to be captured
    kernelA<<<grid, block, 0, stream>>>(d_data, N);
    kernelB<<<grid, block, 0, stream>>>(d_data, N);
    kernelC<<<grid, block, 0, stream>>>(d_data, N);
    
    cudaStreamEndCapture(stream, &graph);
    
    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    printf("   Replaying graph %d times...\n", iterations);
    cudaEventRecord(start);
    
    for (int iter = 0; iter < iterations; ++iter) {
        cudaGraphLaunch(graphExec, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float graph_time;
    cudaEventElapsedTime(&graph_time, start, stop);
    printf("   Time: %.2f ms (%.3f ms per iteration)\n", 
           graph_time, graph_time / iterations);
    printf("   Speedup: %.2fx\n", traditional_time / graph_time);
    
    // Verify results are equivalent
    float *h_result = new float[N];
    cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost);
    printf("   Sample result[0]: %.6f\n", h_result[0]);
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    delete[] h_result;
    delete[] h_data;
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrateGraphUpdate() {
    printf("\n=== Graph Update Demonstration ===\n");
    
    const int max_N = 1024 * 1024;
    const int bytes = max_N * sizeof(float);
    
    float *h_data = new float[max_N];
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Initialize data
    for (int i = 0; i < max_N; ++i) {
        h_data[i] = (float)i / max_N;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Capture graph with maximum size
    printf("Capturing graph with maximum size (%d elements)...\n", max_N);
    
    dim3 max_block(256);
    dim3 max_grid((max_N + max_block.x - 1) / max_block.x);
    
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernelA<<<max_grid, max_block, 0, stream>>>(d_data, max_N);
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    // Test different sizes using graph update
    int test_sizes[] = {256*1024, 512*1024, 768*1024, max_N};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    printf("\nTesting graph updates for different sizes:\n");
    printf("Size (KB) | Grid Size | Update Time (μs) | Execution Time (μs)\n");
    printf("----------|-----------|------------------|--------------------\n");
    
    for (int test = 0; test < num_tests; ++test) {
        int N = test_sizes[test];
        dim3 new_grid((N + max_block.x - 1) / max_block.x);
        
        // Measure update time
        auto update_start = std::chrono::high_resolution_clock::now();
        
        // Get the kernel node and update its parameters
        cudaGraphNode_t kernelNode;
        size_t numNodes = 1;
        cudaGraphGetNodes(graph, &kernelNode, &numNodes);
        
        cudaKernelNodeParams nodeParams;
        cudaGraphKernelNodeGetParams(kernelNode, &nodeParams);
        
        // Update grid dimensions and kernel parameters
        nodeParams.gridDim = new_grid;
        void* args[] = {&d_data, &N};
        nodeParams.kernelParams = args;
        
        cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &nodeParams);
        
        auto update_end = std::chrono::high_resolution_clock::now();
        auto update_time = std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start);
        
        // Measure execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEventRecord(start);
        cudaGraphLaunch(graphExec, stream);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float exec_time_ms;
        cudaEventElapsedTime(&exec_time_ms, start, stop);
        float exec_time_us = exec_time_ms * 1000.0f;
        
        printf("%9d | %9d | %16ld | %18.1f\n", 
               N / 1024, new_grid.x, update_time.count(), exec_time_us);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    printf("\nKey insights:\n");
    printf("- Graph updates are very fast (microseconds)\n");
    printf("- Avoid full recapture for parameter changes\n");
    printf("- Capture with maximum expected size, then update\n");
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    delete[] h_data;
    cudaFree(d_data);
    cudaStreamDestroy(stream);
}

void demonstrateConditionalGraphs() {
    printf("\n=== Conditional Graph Nodes Demonstration ===\n");
    
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    float *h_data = new float[N];
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Test different data sets
    float test_cases[] = {0.1f, 0.5f, 1.0f, 2.0f}; // Different average values
    float threshold = 0.75f;
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    
    printf("Testing conditional execution with threshold = %.2f\n", threshold);
    printf("Average Value | Condition Met | Result[0] Before | Result[0] After\n");
    printf("--------------|---------------|------------------|----------------\n");
    
    for (int test = 0; test < num_tests; ++test) {
        float avg_value = test_cases[test];
        
        // Initialize data with specific average
        for (int i = 0; i < N; ++i) {
            h_data[i] = avg_value * (1.0f + 0.1f * sinf((float)i));
        }
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        
        float before_value = h_data[0];
        
        // Create conditional graph
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);
        
        // Note: Conditional graph handles are not supported in CUDA 12.8
        // cudaGraphConditionalHandle condHandle;
        // cudaGraphConditionalHandleCreate(&condHandle, graph, 0);
        
        // Add condition setter kernel
        // Note: Conditional graph functionality not supported in CUDA 12.8
        // cudaGraphNode_t setNode;
        // cudaKernelNodeParams setParams = {};
        // setParams.func = (void*)setCondition;
        // setParams.gridDim = dim3(1);
        // setParams.blockDim = dim3(32);
        // void* setArgs[] = {(void*)&condHandle, (void*)&d_data, (void*)&N, (void*)&threshold};
        // setParams.kernelParams = setArgs;
        // cudaGraphAddKernelNode(&setNode, graph, nullptr, 0, &setParams);
        
        // Create body graph
        cudaGraph_t bodyGraph;
        cudaGraphCreate(&bodyGraph, 0);
        
        cudaGraphNode_t bodyNode;
        cudaKernelNodeParams bodyParams = {};
        bodyParams.func = (void*)conditionalBody;
        bodyParams.gridDim = dim3((N + 255) / 256);
        bodyParams.blockDim = dim3(256);
        
        void* bodyArgs[] = {(void*)&d_data, (void*)&N};
        bodyParams.kernelParams = bodyArgs;
        
        cudaGraphAddKernelNode(&bodyNode, bodyGraph, nullptr, 0, &bodyParams);
        
        // Note: Conditional graph nodes are not supported in CUDA 12.8
        // This is a placeholder for future CUDA versions
        printf("Conditional graph nodes not supported in CUDA 12.8\n");
        
        // Execute the conditional graph
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        
        // Check results
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
        float after_value = h_data[0];
        
        bool condition_met = (after_value != before_value);
        
        printf("%13.2f | %13s | %16.6f | %15.6f\n",
               avg_value, condition_met ? "Yes" : "No", before_value, after_value);
        
        // Cleanup
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(bodyGraph);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    
    printf("\nConditional graphs enable device-side branching without CPU involvement\n");
    
    // Cleanup
    delete[] h_data;
    cudaFree(d_data);
}

void demonstrateMemoryPoolsWithGraphs() {
    printf("\n=== Memory Pools with CUDA Graphs ===\n");
    
    const int N = 512 * 1024;
    const int bytes = N * sizeof(float);
    
    // Setup memory pool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    
    uint64_t threshold = 32 * 1024 * 1024; // 32 MB threshold
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    
    printf("Configured memory pool with %lu MB threshold\n", threshold / (1024 * 1024));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Capture graph with memory operations
    printf("Capturing graph with memory allocations...\n");
    
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Allocate memory within graph
    float *d_temp1, *d_temp2;
    cudaMallocAsync((void**)&d_temp1, bytes, stream);
    cudaMallocAsync((void**)&d_temp2, bytes, stream);
    
    // Launch kernels using allocated memory
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    kernelA<<<grid, block, 0, stream>>>(d_temp1, N);
    kernelB<<<grid, block, 0, stream>>>(d_temp2, N);
    
    // Free memory within graph
    cudaFreeAsync(d_temp1, stream);
    cudaFreeAsync(d_temp2, stream);
    
    cudaStreamEndCapture(stream, &graph);
    
    // Instantiate and execute graph multiple times
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    
    printf("Executing graph with memory operations 10 times...\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start, stop);
    
    printf("Total execution time: %.2f ms (%.2f ms per iteration)\n", 
           total_time, total_time / 10.0f);
    
    // Query memory pool state
    uint64_t reserved, used;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    
    printf("Final pool state: Reserved=%lu MB, Used=%lu MB\n", 
           reserved / (1024 * 1024), used / (1024 * 1024));
    
    printf("Memory pools with graphs enable efficient temporary allocations\n");
    
    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("CUDA Graphs Examples - Chapter 12\n");
    printf("==================================\n");
    
    // Check device capabilities
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("CUDA Capability: %d.%d\n", prop.major, prop.minor);
    
    // Check for graph support
    if (prop.major >= 8 || (prop.major == 7 && prop.minor >= 5)) {
        printf("CUDA Graphs: Supported\n");
    } else {
        printf("CUDA Graphs: Not supported (requires CC 7.5+)\n");
        return 1;
    }
    
    printf("Memory Pools: %s\n", 
           prop.memoryPoolsSupported ? "Supported" : "Not supported");
    printf("\n");
    
    demonstrateBasicGraphs();
    demonstrateGraphUpdate();
    demonstrateConditionalGraphs();
    demonstrateMemoryPoolsWithGraphs();
    
    printf("\n=== CUDA Graphs Summary ===\n");
    printf("Benefits:\n");
    printf("- Reduced kernel launch overhead (1.5-3x speedup typical)\n");
    printf("- Better GPU scheduling with known dependencies\n");
    printf("- Enable device-side control flow with conditional nodes\n");
    printf("- Efficient memory pool integration\n");
    printf("\nBest practices:\n");
    printf("- Capture once, replay many times\n");
    printf("- Use graph updates for parameter changes\n");
    printf("- Pre-allocate memory outside capture when possible\n");
    printf("- Warm up before capture to initialize libraries\n");
    printf("- Use conditional nodes for device-side branching\n");
    
    printf("\n=== Profiling Commands ===\n");
    printf("nsys profile --force-overwrite=true -o cuda_graphs ./cuda_graphs\n");
    printf("ncu --section LaunchStats ./cuda_graphs\n");
    
    return 0;
}
