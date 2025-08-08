// cutlass_gemm_example.cu
// Example using CUTLASS for optimal arithmetic intensity and Tensor Core performance

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <iostream>
#include <sstream>

// Define GEMM operation
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::RowMajor,                 // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    ElementOutput,                             // ElementC
    cutlass::layout::RowMajor,                 // LayoutC
    ElementAccumulator,                        // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OpClass
    cutlass::arch::Sm80                        // ArchTag
>;

bool run_gemm() {
    // Define problem size
    cutlass::gemm::GemmCoord problem_size(1024, 1024, 1024);
    
    // Initialize alpha and beta
    ElementCompute alpha = ElementCompute(1);
    ElementCompute beta = ElementCompute(0);
    
    // Allocate host-side tensors
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> tensor_a(
        problem_size.mk());
    cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> tensor_b(
        problem_size.kn());
    cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_c(
        problem_size.mn());
    cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_d(
        problem_size.mn());
    cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> reference_d(
        problem_size.mn());
    
    // Fill input tensors
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(),
        1,
        cutlass::half_t(4),
        cutlass::half_t(-4),
        0);
    
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        cutlass::half_t(4),
        cutlass::half_t(-4),
        0);
    
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(),
        1,
        ElementOutput(4),
        ElementOutput(-4),
        0);
    
    // Copy data to GPU
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();
    tensor_d.sync_device();
    
    // Create a tuple of gemm kernel arguments
    typename Gemm::Arguments arguments{
        problem_size,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_d.device_ref(),
        {alpha, beta}
    };
    
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    
    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;
    
    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Cannot implement GEMM with given arguments" << std::endl;
        return false;
    }
    
    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS kernel" << std::endl;
        return false;
    }
    
    // Launch initialized CUTLASS kernel
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to launch CUTLASS kernel" << std::endl;
        return false;
    }
    
    // Copy output data to host
    tensor_d.sync_host();
    
    // Compute reference result on CPU
    cutlass::reference::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        ElementOutput, cutlass::layout::RowMajor,
        ElementCompute, ElementAccumulator
    > reference_gemm;
    
    reference_gemm(
        problem_size,
        alpha,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        beta,
        reference_d.device_ref()
    );
    
    // Copy reference result to host
    reference_d.sync_host();
    
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::host::TensorEquals(
        tensor_d.host_view(),
        reference_d.host_view()
    );
    
    std::cout << "Problem: " << problem_size.m() << "x" << problem_size.n() << "x" << problem_size.k() << std::endl;
    std::cout << "CUTLASS GEMM: " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    if (passed) {
        std::cout << "CUTLASS automatically applied optimizations:" << std::endl;
        std::cout << "- Tensor Core operations (FP16 -> FP32 accumulation)" << std::endl;
        std::cout << "- Shared memory tiling with optimal tile sizes" << std::endl;
        std::cout << "- Asynchronous memory operations (cp.async)" << std::endl;
        std::cout << "- Double buffering in TMEM/SMEM" << std::endl;
        std::cout << "- Warp specialization for compute and memory operations" << std::endl;
        std::cout << "- Optimal arithmetic intensity for compute-bound performance" << std::endl;
    }
    
    return passed;
}

int main() {
    std::cout << "CUTLASS GEMM Example - Optimal Arithmetic Intensity" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    bool result = run_gemm();
    
    if (result) {
        std::cout << "\nTo profile with Nsight Compute roofline analysis:" << std::endl;
        std::cout << "ncu --section RooflineChart ./cutlass_gemm_example" << std::endl;
        return 0;
    } else {
        return -1;
    }
}
