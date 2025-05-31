#include <cutlass/gemm/device/gemm.h>
#include <iostream>
using Gemm = cutlass::gemm::device::Gemm<
    half, cutlass::layout::RowMajor,
    half, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor,
    half, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90>;

int main() {
    int M = 1024, N = 1024, K = 1024;
    half *A, *B; float *C;
    size_t sizeA = M*K*sizeof(half), sizeB = K*N*sizeof(half), sizeC = M*N*sizeof(float);
    cudaMallocManaged(&A, sizeA); cudaMallocManaged(&B, sizeB); cudaMallocManaged(&C, sizeC);
    Gemm gemm_op;
    cutlass::Status status = gemm_op({M, N, K}, half(1.0), A, K, B, N, float(0), C, N);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed
";
    }
    cudaDeviceSynchronize();
    std::cout << "CUTLASS GEMM completed
";
    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
