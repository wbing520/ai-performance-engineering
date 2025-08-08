#include <cutlass/gemm/device/gemm.h>
#include <iostream>
#include <cuda_runtime.h>

// Blackwell B200/B300 specific optimizations
#if CUDA_VERSION >= 12.9
#define BLACKWELL_OPTIMIZED
#endif

using Gemm = cutlass::gemm::device::Gemm<
    half, cutlass::layout::RowMajor,
    half, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    half, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm100>;

int main() {
    // Check CUDA version and device capabilities
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "CUDA Version: " << CUDA_VERSION << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    int M = 1024, N = 1024, K = 1024;
    half *A, *B; float *C;
    size_t sizeA = M*K*sizeof(half), sizeB = K*N*sizeof(half), sizeC = M*N*sizeof(float);
    
    // Use stream-ordered memory allocation for better performance
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMallocAsync(&A, sizeA, stream);
    cudaMallocAsync(&B, sizeB, stream);
    cudaMallocAsync(&C, sizeC, stream);
    
    Gemm gemm_op;
    cutlass::Status status = gemm_op({M, N, K}, half(1.0), A, K, B, N, float(0), C, N);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed" << std::endl;
    }
    cudaDeviceSynchronize();
    std::cout << "CUTLASS GEMM completed" << std::endl;
    
    // Cleanup with stream-ordered deallocation
    cudaFreeAsync(A, stream);
    cudaFreeAsync(B, stream);
    cudaFreeAsync(C, stream);
    cudaStreamDestroy(stream);
    return 0;
}
