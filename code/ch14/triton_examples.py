import torch
import triton
import triton.language as tl


@triton.jit
def tiled_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create block descriptors to read A and B tiles; Triton will stage via SMEM for pipelining.
    A_blk = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(offs_m, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    B_blk = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, offs_n),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_tiles = (K + BLOCK_K - 1) // BLOCK_K
    # Software pipeline with two stages; Triton lowers to cp.async on NVIDIA.
    for kt in tl.range(0, K_tiles, num_stages=2):
        k0 = kt * BLOCK_K

        # Guards for tail tiles
        a_mask = (offs_m[:, None] < M) & ((k0 + tl.arange(0, BLOCK_K))[None, :] < K)
        b_mask = ((k0 + tl.arange(0, BLOCK_K))[:, None] < K) & (offs_n[None, :] < N)

        # Advance block pointers
        A_cur = tl.advance(A_blk, (0, k0))
        B_cur = tl.advance(B_blk, (k0, 0))

        # Load tiles (masked, zero-padded)
        a = tl.load(A_cur, mask=a_mask, other=0.0)
        b = tl.load(B_cur, mask=b_mask, other=0.0)

        # FMA on fp32 accumulators (BF16/FP16 inputs recommended in driver)
        acc += tl.dot(a, b)

    # Store results with masking
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Tunables (good starting points for B200 / Triton 3.4)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    tiled_gemm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    np = tl.num_programs(axis=0)

    MT = tl.cdiv(M, BLOCK_M)
    NT = tl.cdiv(N, BLOCK_N)
    TILE_COUNT = MT * NT

    for tile_idx in range(pid, TILE_COUNT, np):
        pid_m = tile_idx // NT
        pid_n = tile_idx % NT

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        for k0 in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
            b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float32))
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    MT = triton.cdiv(M, 128)  # matches autotune defaults; Triton will try both configs
    NT = triton.cdiv(N, 128)
    grid = lambda META: (min(65536, MT * NT),)  # cap to keep launch overhead bounded

    matmul_kernel_persistent[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
