#!/usr/bin/env python3
"""
PyTorch 2.9 Native FlexAttention Example
=========================================

Demonstrates the new flex_attention API introduced in PyTorch 2.9, which provides
efficient implementations of various attention patterns without custom CUDA kernels.

FlexAttention is more efficient than manual implementations for:
- Sliding window attention
- Block-sparse attention
- Prefix caching
- ALiBi/RoPE-style position biases
- Causal masking variants

Requirements:
- PyTorch 2.9+
- CUDA GPU (Blackwell B200/B300 recommended)

Expected Runtime: ~2-5 seconds
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import time
from typing import Callable, Optional

# Check if FlexAttention is available (PyTorch 2.9+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("WARNING: FlexAttention not available. Requires PyTorch 2.9+")


def sliding_window_causal(b: int, h: int, q_idx: int, kv_idx: int, window_size: int = 2048) -> bool:
    """Sliding window causal mask function.
    
    Args:
        b: Batch index
        h: Head index
        q_idx: Query position
        kv_idx: Key/value position
        window_size: Size of attention window
        
    Returns:
        True if attention is allowed, False otherwise
    """
    return (q_idx >= kv_idx) and (q_idx - kv_idx < window_size)


def prefix_causal(b: int, h: int, q_idx: int, kv_idx: int, prefix_len: int = 128) -> bool:
    """Prefix + causal attention mask (for prefix caching).
    
    All queries can attend to prefix, then causal for the rest.
    """
    # Can attend to prefix
    if kv_idx < prefix_len:
        return True
    # Causal for non-prefix tokens
    return q_idx >= kv_idx


def block_sparse_mask(b: int, h: int, q_idx: int, kv_idx: int, block_size: int = 64) -> bool:
    """Block-sparse attention mask.
    
    Only attend within the same block and to previous blocks.
    """
    q_block = q_idx // block_size
    kv_block = kv_idx // block_size
    return kv_block <= q_block


def benchmark_attention_patterns(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 4096,
    head_dim: int = 64,
    device: str = "cuda",
) -> None:
    """Benchmark different attention patterns using FlexAttention.
    
    Args:
        batch_size: Number of sequences
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        device: Device to run on
    """
    if not FLEX_ATTENTION_AVAILABLE:
        print("FlexAttention not available. Install PyTorch 2.9+")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available. FlexAttention requires GPU.")
        return
    
    print("=" * 80)
    print("PyTorch 2.9 FlexAttention Benchmark")
    print("=" * 80)
    print(f"Configuration: B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print()
    
    # Create input tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    
    # 1. Standard causal attention (baseline)
    start = time.time()
    for _ in range(10):
        output_baseline = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 10 * 1000
    
    print(f"1. Standard Causal Attention (SDPA):     {baseline_time:.2f} ms/iter")
    
    # 2. FlexAttention with sliding window
    window_size = 2048
    
    def sliding_window_fn(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) and (q_idx - kv_idx < window_size)
    
    block_mask_sliding = create_block_mask(
        sliding_window_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    
    # Warmup
    _ = flex_attention(query, key, value, block_mask=block_mask_sliding)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        output_sliding = flex_attention(query, key, value, block_mask=block_mask_sliding)
    torch.cuda.synchronize()
    sliding_time = (time.time() - start) / 10 * 1000
    
    print(f"2. Sliding Window ({window_size}):            {sliding_time:.2f} ms/iter ({baseline_time/sliding_time:.2f}x speedup)")
    
    # 3. FlexAttention with prefix caching
    prefix_len = 128
    
    def prefix_fn(b, h, q_idx, kv_idx):
        if kv_idx < prefix_len:
            return True
        return q_idx >= kv_idx
    
    block_mask_prefix = create_block_mask(
        prefix_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    
    # Warmup
    _ = flex_attention(query, key, value, block_mask=block_mask_prefix)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        output_prefix = flex_attention(query, key, value, block_mask=block_mask_prefix)
    torch.cuda.synchronize()
    prefix_time = (time.time() - start) / 10 * 1000
    
    print(f"3. Prefix Caching (prefix={prefix_len}):       {prefix_time:.2f} ms/iter ({baseline_time/prefix_time:.2f}x speedup)")
    
    # 4. FlexAttention with block-sparse
    block_size = 64
    
    def block_sparse_fn(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        return kv_block <= q_block
    
    block_mask_sparse = create_block_mask(
        block_sparse_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    
    # Warmup
    _ = flex_attention(query, key, value, block_mask=block_mask_sparse)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        output_sparse = flex_attention(query, key, value, block_mask=block_mask_sparse)
    torch.cuda.synchronize()
    sparse_time = (time.time() - start) / 10 * 1000
    
    print(f"4. Block-Sparse (block={block_size}):          {sparse_time:.2f} ms/iter ({baseline_time/sparse_time:.2f}x speedup)")
    
    print()
    print("=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("1. FlexAttention enables efficient custom attention patterns without CUDA")
    print("2. Sliding window reduces memory and compute for long sequences")
    print("3. Prefix caching improves performance for common prefix reuse")
    print("4. Block-sparse attention trades accuracy for speed")
    print("5. On Blackwell, FlexAttention uses hardware-accelerated kernels")
    print("=" * 80)


def demonstrate_score_mod() -> None:
    """Demonstrate score modification in FlexAttention.
    
    Score modification allows applying custom transformations to attention scores,
    useful for:
    - Position biases (ALiBi, RoPE-style)
    - Custom attention scaling
    - Token importance weighting
    """
    if not FLEX_ATTENTION_AVAILABLE:
        return
    
    if not torch.cuda.is_available():
        return
    
    print("\n" + "=" * 80)
    print("FlexAttention Score Modification Example")
    print("=" * 80)
    
    batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
    device = "cuda"
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Example 1: ALiBi-style linear bias
    def alibi_score_mod(score, b, h, q_idx, kv_idx):
        """Apply ALiBi-style linear position bias."""
        # ALiBi: subtract a penalty based on distance
        distance = q_idx - kv_idx
        # Different slopes for different heads
        slope = 1.0 / (2 ** (h + 1))
        bias = -distance * slope
        return score + bias
    
    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    block_mask = create_block_mask(
        causal_mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
    )
    
    # Apply FlexAttention with score modification
    output = flex_attention(
        query, key, value,
        block_mask=block_mask,
        score_mod=alibi_score_mod,
    )
    
    print(f"✓ ALiBi-style attention computed")
    print(f"  Input shape: {query.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Position bias applied per head with different slopes")
    
    # Example 2: Custom temperature scaling
    def temperature_score_mod(score, b, h, q_idx, kv_idx):
        """Apply temperature scaling that varies by head."""
        temperature = 1.0 + (h / num_heads) * 0.5  # Temperature from 1.0 to 1.5
        return score / temperature
    
    output_temp = flex_attention(
        query, key, value,
        block_mask=block_mask,
        score_mod=temperature_score_mod,
    )
    
    print(f"✓ Temperature-scaled attention computed")
    print(f"  Different temperatures per head: {[f'{1.0 + (h/num_heads)*0.5:.2f}' for h in range(4)]}...")
    
    print("\n" + "=" * 80)
    print("Score Modification Benefits:")
    print("- Implement position encodings without additional ops")
    print("- Apply per-head or per-token importance weighting")
    print("- Fused with attention computation (no extra memory)")
    print("- Hardware-accelerated on Blackwell GPUs")
    print("=" * 80)


def main() -> None:
    """Run all FlexAttention examples."""
    if not FLEX_ATTENTION_AVAILABLE:
        print("\n" + "!" * 80)
        print("FlexAttention is not available in your PyTorch installation.")
        print("This feature requires PyTorch 2.9 or later.")
        print("\nTo install PyTorch 2.9:")
        print("  pip install torch==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130")
        print("!" * 80)
        return
    
    # Run benchmarks with different sequence lengths
    for seq_len in [1024, 4096]:
        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'=' * 80}")
        benchmark_attention_patterns(seq_len=seq_len)
    
    # Demonstrate score modification
    demonstrate_score_mod()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("PyTorch 2.9's FlexAttention provides:")
    print("✓ Efficient custom attention patterns without CUDA kernels")
    print("✓ Block-mask based sparsity for memory and compute savings")
    print("✓ Score modification for position biases and custom scaling")
    print("✓ Hardware acceleration on Blackwell GPUs")
    print("✓ Automatic fusion with FlashAttention-3 backend")
    print("\nUse FlexAttention instead of custom kernels for production systems!")
    print("=" * 80)


if __name__ == "__main__":
    main()

