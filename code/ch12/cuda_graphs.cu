// cuda_graphs.cu -- CUDA 13.0 graph capture/update demonstrations for Blackwell.

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void kernel_a(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = data[idx] * 1.1f + 0.1f;
}

__global__ void kernel_b(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
}

__global__ void kernel_c(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = sinf(data[idx] * 0.1f);
}

static void basic_capture_and_replay() {
  printf("\n=== Basic graph capture/replay ===\n");
  constexpr int N = 1 << 20;
  constexpr int ITER = 100;
  size_t bytes = N * sizeof(float);

  std::vector<float> host(N);
  for (int i = 0; i < N; ++i) host[i] = float(i) / N;

  float* device;
  cudaMalloc(&device, bytes);
  cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Baseline launches
  cudaEventRecord(start, stream);
  for (int i = 0; i < ITER; ++i) {
    kernel_a<<<grid, block, 0, stream>>>(device, N);
    kernel_b<<<grid, block, 0, stream>>>(device, N);
    kernel_c<<<grid, block, 0, stream>>>(device, N);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float baseline_ms;
  cudaEventElapsedTime(&baseline_ms, start, stop);
  printf("Separate launches: %.3f ms (%.3f ms/iter)\n", baseline_ms, baseline_ms / ITER);

  // Reset data
  cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);

  // Capture
  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel_a<<<grid, block, 0, stream>>>(device, N);
  kernel_b<<<grid, block, 0, stream>>>(device, N);
  kernel_c<<<grid, block, 0, stream>>>(device, N);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t exec;
  cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

  cudaEventRecord(start, stream);
  for (int i = 0; i < ITER; ++i) {
    cudaGraphLaunch(exec, stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float graph_ms;
  cudaEventElapsedTime(&graph_ms, start, stop);
  printf("Graph replay     : %.3f ms (%.3f ms/iter, %.2fx speedup)\n",
         graph_ms, graph_ms / ITER, baseline_ms / graph_ms);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cudaFree(device);
}

static void graph_update_demo() {
  printf("\n=== Graph update demo ===\n");
  constexpr int MAX_N = 1 << 20;
  size_t bytes = MAX_N * sizeof(float);

  std::vector<float> host(MAX_N, 1.0f);
  float* device;
  cudaMalloc(&device, bytes);
  cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  dim3 block(256);
  dim3 grid((MAX_N + block.x - 1) / block.x);

  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel_a<<<grid, block, 0, stream>>>(device, MAX_N);
  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t exec;
  cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

  std::vector<int> sizes = {256 * 1024, 512 * 1024, 768 * 1024, MAX_N};
  for (int n : sizes) {
    dim3 new_grid((n + block.x - 1) / block.x);

    cudaGraphNode_t node;
    size_t num_nodes = 1;
    cudaGraphGetNodes(graph, &node, &num_nodes);

    cudaKernelNodeParams params;
    cudaGraphKernelNodeGetParams(node, &params);
    params.gridDim = new_grid;
    void* args[] = {&device, &n};
    params.kernelParams = args;
    cudaGraphExecKernelNodeSetParams(exec, node, &params);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    cudaGraphLaunch(exec, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("N=%7d -> %.3f ms\n", n, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(device);
}

int main() {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("CUDA Graphs demo on %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
  if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
    printf("CUDA Graphs require compute capability 7.5 or newer.\n");
    return 0;
  }

  basic_capture_and_replay();
  graph_update_demo();
  printf("\nInspect with: ncu --section LaunchStats ./cuda_graphs\n");
  return 0;
}
