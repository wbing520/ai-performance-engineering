# triton_examples.py
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch import Tensor
import time
import math

# Basic vector addition kernel
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Simple vector addition kernel demonstrating basic Triton concepts.
    """
    pid = tl.program_id(axis=0)  # unique program ID for each block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # each program handles BLOCK_SIZE elements
    
    # Create a mask to guard against out-of-bounds
    mask = offsets < n_elements
    
    # Masked loads and stores
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)

def vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Launch vector addition kernel.
    """
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    
    # Launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vector_add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Matrix multiplication kernel with shared memory
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Matrix multiplication kernel using tiling and shared memory.
    """
    # Program IDs for the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to the start of each matrix for this block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks from A and B
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        
        # Perform matrix multiplication on the loaded blocks
        accumulator += tl.dot(a, b)
        
        # Advance pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matrix multiplication using Triton.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch configuration
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c

# Fused activation kernel
@triton.jit
def fused_activation_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Fused activation kernel supporting multiple activation functions.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Apply activation function
    if ACTIVATION == 0:  # ReLU
        output = tl.where(x > 0, x, 0.0)
    elif ACTIVATION == 1:  # GELU (approximation)
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        x_cubed = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x_cubed)  # sqrt(2/π) ≈ 0.7978845608
        tanh_inner = tl.tanh(inner)
        output = 0.5 * x * (1.0 + tanh_inner)
    elif ACTIVATION == 2:  # Sigmoid
        output = 1.0 / (1.0 + tl.exp(-x))
    else:  # Identity
        output = x
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_activation_triton(x: torch.Tensor, activation: str = "relu") -> torch.Tensor:
    """
    Apply fused activation function using Triton.
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Map activation names to constants
    activation_map = {"relu": 0, "gelu": 1, "sigmoid": 2, "identity": 3}
    activation_code = activation_map.get(activation.lower(), 3)
    
    # Launch configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_activation_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ACTIVATION=activation_code,
    )
    
    return output

# Advanced: Attention kernel
@triton.jit
def attention_kernel(
    Q, K, V, Out,
    seq_len, head_dim,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simplified Flash Attention kernel implementation.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, head_dim)
    
    # Load Q block
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len)
    
    # Initialize running max and sum for numerical stability
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Loop over K and V
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K and V blocks
        k_ptrs = K + off_hz * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk
        v_ptrs = V + off_hz * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk
        
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < seq_len)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < seq_len)
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk = qk / math.sqrt(head_dim)  # Scale
        
        # Apply causal mask (for decoder-style attention)
        mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        # Update running max and compute softmax
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), 1)
        
        # Update accumulator
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # Compute weighted values
        p = tl.exp(qk - m_ij[:, None])
        acc += tl.dot(p, v) * (beta / l_i_new)[:, None]
        
        # Update running statistics
        l_i = l_i_new
        m_i = m_i_new
    
    # Store output
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)

def attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Simplified Flash Attention implementation using Triton.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    assert k.shape == v.shape == q.shape
    
    output = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
    
    attention_kernel[grid](
        q, k, v, output,
        seq_len, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output

# Register Triton operations with PyTorch
@triton_op("mylib::vector_add")
def vector_add_op(x: Tensor, y: Tensor) -> Tensor:
    return vector_add_triton(x, y)

@triton_op("mylib::matmul")  
def matmul_op(a: Tensor, b: Tensor) -> Tensor:
    return matmul_triton(a, b)

@triton_op("mylib::fused_activation")
def fused_activation_op(x: Tensor, activation: str = "relu") -> Tensor:
    return fused_activation_triton(x, activation)

def benchmark_triton_vs_pytorch():
    """
    Benchmark Triton kernels against PyTorch built-in operations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping Triton benchmarks")
        return
    
    print("Benchmarking Triton vs PyTorch operations")
    print("=" * 50)
    
    # Vector addition benchmark
    sizes = [1024, 10240, 102400, 1024000]
    
    for size in sizes:
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = vector_add_triton(x, y)
            _ = x + y
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start = time.time()
        for _ in range(100):
            result_triton = vector_add_triton(x, y)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(100):
            result_pytorch = x + y
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        speedup = pytorch_time / triton_time
        print(f"Size {size:>7}: Triton {triton_time:.4f}s, PyTorch {pytorch_time:.4f}s, Speedup: {speedup:.2f}x")
    
    # Matrix multiplication benchmark
    print(f"\nMatrix Multiplication Benchmark:")
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    for M, N in sizes:
        K = M
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N, device=device)
        
        # Warmup
        for _ in range(5):
            _ = matmul_triton(a, b)
            _ = torch.mm(a, b)
        
        torch.cuda.synchronize()
        
        # Benchmark Triton
        start = time.time()
        for _ in range(20):
            result_triton = matmul_triton(a, b)
        torch.cuda.synchronize()
        triton_time = time.time() - start
        
        # Benchmark PyTorch
        start = time.time()
        for _ in range(20):
            result_pytorch = torch.mm(a, b)
        torch.cuda.synchronize()
        pytorch_time = time.time() - start
        
        speedup = pytorch_time / triton_time
        print(f"Size {M}x{K}x{N}: Triton {triton_time:.4f}s, PyTorch {pytorch_time:.4f}s, Speedup: {speedup:.2f}x")

def test_custom_kernels():
    """
    Test all custom Triton kernels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping Triton tests")
        return
    
    print("Testing Triton Custom Kernels")
    print("=" * 30)
    
    # Test vector addition
    x = torch.randn(1000, device=device)
    y = torch.randn(1000, device=device)
    
    result_triton = vector_add_triton(x, y)
    result_pytorch = x + y
    
    print(f"Vector Addition - Max diff: {torch.max(torch.abs(result_triton - result_pytorch)).item()}")
    
    # Test matrix multiplication
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    
    result_triton = matmul_triton(a, b)
    result_pytorch = torch.mm(a, b)
    
    print(f"Matrix Multiplication - Max diff: {torch.max(torch.abs(result_triton - result_pytorch)).item()}")
    
    # Test fused activations
    x = torch.randn(1000, device=device)
    
    activations = ["relu", "gelu", "sigmoid"]
    for activation in activations:
        result_triton = fused_activation_triton(x, activation)
        
        if activation == "relu":
            result_pytorch = torch.relu(x)
        elif activation == "gelu":
            result_pytorch = torch.nn.functional.gelu(x)
        elif activation == "sigmoid":
            result_pytorch = torch.sigmoid(x)
        
        max_diff = torch.max(torch.abs(result_triton - result_pytorch)).item()
        print(f"Fused {activation.upper()} - Max diff: {max_diff}")

def main():
    """
    Run all Triton examples.
    """
    print("Triton GPU Kernel Examples")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a GPU-enabled system.")
        return
    
    print("\n1. Testing Custom Kernels:")
    test_custom_kernels()
    
    print("\n2. Benchmarking Performance:")
    benchmark_triton_vs_pytorch()
    
    print("\nAll Triton examples completed!")

if __name__ == "__main__":
    main()
