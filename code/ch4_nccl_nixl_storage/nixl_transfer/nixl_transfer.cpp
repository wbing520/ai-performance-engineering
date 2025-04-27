#include <nixl/nixl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>

#define CHECK_CUDA(call) do { cudaError_t err = call;   if (err != cudaSuccess) { std::cerr<<"CUDA Error: "<<cudaGetErrorString(err); std::exit(EXIT_FAILURE);} } while(0)

int main() {
    // Initialize source and destination agents
    nixlAgentHandle agentSrc, agentDst;
    nixlInitAgent(&agentSrc);
    nixlInitAgent(&agentDst);

    // Allocate buffers
    const size_t dataSize = 1ull << 30; // 1 GiB
    float *srcPtr, *dstPtr;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&srcPtr, dataSize));
    std::vector<float> init(dataSize/4, 1.0f);
    CHECK_CUDA(cudaMemcpy(srcPtr, init.data(), dataSize, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&dstPtr, dataSize));

    // Register with NIXL
    nixlMemHandle memSrc, memDst;
    nixlRegisterMemory(agentSrc, srcPtr, dataSize, &memSrc);
    nixlRegisterMemory(agentDst, dstPtr, dataSize, &memDst);

    // Create & post transfer request
    nixlXferRequest req;
    nixlCreateXferRequest(agentSrc, NIXL_WRITE, memSrc, agentDst, memDst, &req);
    nixlSetNotification(req, agentDst, "transfer_done");
    nixlPostRequest(req);
    std::cout<<"Transfer posted, running concurrently..."<<std::endl;

    // Dummy compute on GPU0
    CHECK_CUDA(cudaSetDevice(0));
    const int N=1<<20;
    float* tmp; CHECK_CUDA(cudaMalloc(&tmp, N*sizeof(float)));
    CHECK_CUDA(cudaFree(tmp));
    std::cout<<"Computation done."<<std::endl;

    // Wait for completion
    while(!nixlPollNotification(agentDst, "transfer_done")) {
        std::this_thread::yield();
    }
    uint64_t usec=0; nixlGetRequestDuration(req, &usec);
    std::cout<<"Transfer completed in "<<(usec/1e3)<<" ms"<<std::endl;

    // Verify result
    float check;
    CHECK_CUDA(cudaMemcpy(&check, dstPtr, sizeof(float), cudaMemcpyDeviceToHost));
    assert(check==1.0f);
    std::cout<<"Result verified!"<<std::endl;

    // Cleanup
    nixlReleaseRequest(req);
    nixlDeregisterMemory(memSrc);
    nixlDeregisterMemory(memDst);
    cudaFree(srcPtr);
    cudaFree(dstPtr);
    nixlShutdownAgent(agentSrc);
    nixlShutdownAgent(agentDst);
    return 0;
}
