// p2p_bandwidth_bench.cu
// Measures GPU peer-to-peer bandwidth between device 0 and 1.
// Requires: CUDA 13.0, C++17

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(err));      \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while(0)

int main() {
    int devCount = 0;
    CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        std::cerr << "Need at least 2 GPUs for peer-to-peer benchmark\n";
        return 1;
    }

    // Enable peer access
    CHECK(cudaSetDevice(0));
    CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaSetDevice(1));
    CHECK(cudaDeviceEnablePeerAccess(0, 0));

    std::vector<size_t> sizes = {1<<20, 4<<20, 16<<20, 64<<20, 256<<20, 1<<30}; //bytes
    std::cout << "Size(MB)    Bandwidth(GB/s)\n";

    for (auto bytes : sizes) {
        // allocate on src and dst
        CHECK(cudaSetDevice(0));
        void* src = nullptr;
        CHECK(cudaMalloc(&src, bytes));
        CHECK(cudaMemset(src, 0, bytes));

        CHECK(cudaSetDevice(1));
        void* dst = nullptr;
        CHECK(cudaMalloc(&dst, bytes));

        // events
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        // record, copy, record
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpyPeer(dst, 1, src, 0, bytes));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));

        float gb = bytes / 1e9f;
        float bw = gb / (ms / 1e3f);
        std::printf("%8zu    %10.2f\n", bytes/(1<<20), bw);

        // cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        CHECK(cudaFree(src));
        CHECK(cudaFree(dst));
    }

    return 0;
}
