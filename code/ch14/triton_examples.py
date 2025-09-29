"""Triton examples aligned with Chapter 14 best practices."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _grid(numel: int, block: int) -> tuple[int]:
    return ((numel + block - 1) // block,)


@triton.jit
def _vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    BLOCK = 1024
    _vector_add[_grid(x.numel(), BLOCK)](x, y, out, x.numel(), BLOCK=BLOCK)
    return out


@triton.jit
def _matmul(a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        A = tl.load(a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0)
        B = tl.load(b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(A, B)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0] and a.is_cuda and b.is_cuda
    M, K = a.shape
    _, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK = 32
    _matmul[
        (_grid(M, BLOCK)[0] * _grid(N, BLOCK)[0],)
    ](a, b, out, M, N, K,
      a.stride(0), a.stride(1),
      b.stride(0), b.stride(1),
      out.stride(0), out.stride(1),
      BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK)
    return out
