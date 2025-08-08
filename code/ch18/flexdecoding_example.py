import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "9.0":
        return "hopper"  # H100/H200
    elif compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return {
            "name": "Hopper H100/H200",
            "compute_capability": "9.0",
            "sm_version": "sm_90",
            "memory_bandwidth": "3.35 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3", "Transformer Engine", "Dynamic Programming"]
        }
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "3.2 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    else:
        return {
            "name": "Other",
            "compute_capability": "Unknown",
            "sm_version": "Unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": []
        }
#!/usr/bin/env python3
"""
flexdecoding_example.py
Chapter 18: FlexDecoding for Autoregressive Inference

Implementation demonstrating PyTorch's FlexDecoding capabilities for
efficient autoregressive inference with flexible attention patterns.

Based on Chapter 18 content about FlexDecoding optimizations.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import flex_attention
import math
import time
from typing import Optional, Tuple, Callable
import numpy as np


class FlexDecodingAttention(torch.nn.Module):
    """
    FlexDecoding implementation using PyTorch's flex_attention.
    Demonstrates efficient autoregressive inference with custom attention patterns.
    """
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        self.o_proj = torch.nn.Linear(dim, dim, bias=False)
        
        # KV cache for autoregressive generation
        self.register_buffer("k_cache", torch.zeros(1, max_seq_len, num_heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, max_seq_len, num_heads, self.head_dim))
        
        # Compiled kernels for different phases
        self._prefill_fn = None
        self._decode_fn = None
        
    def _get_causal_mask_fn(self):
        """Create causal mask function for autoregressive attention."""
        def causal_mask(batch, head, q_idx, kv_idx):
            return q_idx >= kv_idx
        return causal_mask
    
    def _get_local_attention_fn(self, window_size: int = 128):
        """Create local attention mask with sliding window."""
        def local_mask(batch, head, q_idx, kv_idx):
            return (q_idx >= kv_idx) and (q_idx - kv_idx <= window_size)
        return local_mask
    
    def _get_block_sparse_fn(self, block_size: int = 64):
        """Create block-sparse attention pattern."""
        def block_sparse_mask(batch, head, q_idx, kv_idx):
            q_block = q_idx // block_size
            kv_block = kv_idx // block_size
            
            # Allow attention within blocks and to previous blocks
            return (q_idx >= kv_idx) and (
                (q_block == kv_block) or  # Same block
                (kv_block % 2 == 0)       # Even blocks (strided pattern)
            )
        return block_sparse_mask
    
    def compile_kernels(self, pattern: str = "causal"):
        """
        Compile specialized kernels for prefill and decode phases.
        This is the key FlexDecoding optimization from Chapter 18.
        """
        print(f"Compiling FlexDecoding kernels for {pattern} pattern...")
        
        # Get mask function based on pattern
        if pattern == "causal":
            mask_fn = self._get_causal_mask_fn()
        elif pattern == "local":
            mask_fn = self._get_local_attention_fn()
        elif pattern == "block_sparse":
            mask_fn = self._get_block_sparse_fn()
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Create sample tensors for compilation
        seq_len = 512
        q_prefill = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        k_prefill = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_prefill = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        
        q_decode = torch.randn(1, 1, self.num_heads, self.head_dim)  # Single token
        k_decode = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        v_decode = torch.randn(1, seq_len, self.num_heads, self.head_dim)
        
        # Compile prefill kernel (Q_len >> KV_len scenario)
        print("  Compiling prefill kernel...")
        self._prefill_fn = torch.compile(
            lambda q, k, v: flex_attention(q, k, v, score_mod=mask_fn),
            mode="max-autotune"
        )
        
        # Warmup compilation with sample data
        with torch.no_grad():
            _ = self._prefill_fn(q_prefill, k_prefill, v_prefill)
        
        # Compile decode kernel (Q_len = 1 scenario) 
        print("  Compiling decode kernel...")
        self._decode_fn = torch.compile(
            lambda q, k, v: flex_attention(q, k, v, score_mod=mask_fn),
            mode="max-autotune"
        )
        
        # Warmup compilation
        with torch.no_grad():
            _ = self._decode_fn(q_decode, k_decode, v_decode)
        
        print("  Kernel compilation complete!")
    
    def prefill(self, x: torch.Tensor, past_kv_len: int = 0) -> torch.Tensor:
        """
        Prefill phase: Process entire prompt sequence.
        Uses compiled prefill kernel optimized for long sequences.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim) 
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Update KV cache
        self.k_cache[:, past_kv_len:past_kv_len + seq_len] = k
        self.v_cache[:, past_kv_len:past_kv_len + seq_len] = v
        
        # Use compiled prefill kernel
        if self._prefill_fn is not None:
            attn_out = self._prefill_fn(q, k, v)
        else:
            # Fallback to standard attention
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Project output
        attn_out = attn_out.view(batch_size, seq_len, self.dim)
        return self.o_proj(attn_out)
    
    def decode_step(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """
        Decode phase: Process single token attending to full KV cache.
        Uses compiled decode kernel optimized for Q_len=1 case.
        """
        batch_size, _, _ = x.shape  # x should be [batch, 1, dim]
        
        # Project single token to Q, K, V
        q = self.q_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, 1, self.num_heads, self.head_dim)
        
        # Update KV cache at current position
        self.k_cache[:, position:position+1] = k
        self.v_cache[:, position:position+1] = v
        
        # Get relevant KV from cache (up to current position)
        k_past = self.k_cache[:, :position+1]
        v_past = self.v_cache[:, :position+1]
        
        # Use compiled decode kernel (FlashDecoding optimization)
        if self._decode_fn is not None:
            attn_out = self._decode_fn(q, k_past, v_past)
        else:
            # Fallback
            attn_out = F.scaled_dot_product_attention(q, k_past, v_past, is_causal=True)
        
        # Project output
        attn_out = attn_out.view(batch_size, 1, self.dim)
        return self.o_proj(attn_out)
    
    def clear_cache(self):
        """Clear KV cache for new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()


class NestedJaggedTensorDemo:
    """
    Demonstrate nested jagged tensor support mentioned in Chapter 18.
    Handles variable-length sequences efficiently.
    """
    
    @staticmethod
    def create_jagged_tensor(sequences: list, max_len: Optional[int] = None):
        """Create jagged tensor from variable-length sequences."""
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        batch_size = len(sequences)
        dim = sequences[0].shape[-1] if len(sequences) > 0 else 512
        
        # Create padded tensor and offsets
        padded = torch.zeros(batch_size, max_len, dim)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long)
        
        total_tokens = 0
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_len)
            padded[i, :seq_len] = seq[:seq_len]
            offsets[i + 1] = total_tokens + seq_len
            total_tokens += seq_len
        
        return padded, offsets
    
    @staticmethod
    def batch_inference_jagged(model: FlexDecodingAttention, 
                              sequences: list) -> list:
        """
        Efficient batched inference for variable-length sequences.
        Demonstrates Chapter 18's jagged tensor optimization.
        """
        padded_input, offsets = NestedJaggedTensorDemo.create_jagged_tensor(sequences)
        batch_size = len(sequences)
        
        # Process each sequence length separately for efficiency
        outputs = []
        
        for i in range(batch_size):
            seq_len = len(sequences[i])
            seq_input = padded_input[i:i+1, :seq_len]  # [1, seq_len, dim]
            
            # Prefill phase
            model.clear_cache()
            prefill_out = model.prefill(seq_input)
            
            # Decode phase - generate a few tokens
            last_token = prefill_out[:, -1:, :]  # [1, 1, dim]
            generated = [last_token]
            
            for step in range(5):  # Generate 5 tokens
                decode_out = model.decode_step(last_token, seq_len + step)
                generated.append(decode_out)
                last_token = decode_out
            
            outputs.append(torch.cat(generated, dim=1))
        
        return outputs


def benchmark_flexdecoding():
    """
    Benchmark FlexDecoding vs standard attention.
    Demonstrates the performance benefits from Chapter 18.
    """
    print("\n=== FlexDecoding Benchmark ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Model parameters
    dim = 2048
    num_heads = 16
    max_seq_len = 2048
    batch_size = 8
    
    # Create models
    flex_model = FlexDecodingAttention(dim, num_heads, max_seq_len).to(device)
    
    # Test different attention patterns
    patterns = ["causal", "local", "block_sparse"]
    
    for pattern in patterns:
        print(f"\n--- Testing {pattern} pattern ---")
        
        # Compile kernels
        flex_model.compile_kernels(pattern)
        
        # Create test data
        seq_len = 1024
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        # Benchmark prefill phase
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            with torch.no_grad():
                _ = flex_model.prefill(x)
        
        torch.cuda.synchronize()
        prefill_time = (time.time() - start_time) / 10
        
        # Benchmark decode phase
        single_token = torch.randn(batch_size, 1, dim, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for step in range(50):
            with torch.no_grad():
                _ = flex_model.decode_step(single_token, seq_len + step)
        
        torch.cuda.synchronize()
        decode_time = (time.time() - start_time) / 50
        
        print(f"  Prefill time: {prefill_time*1000:.2f} ms")
        print(f"  Decode time: {decode_time*1000:.2f} ms")
        print(f"  Tokens/sec (decode): {batch_size / decode_time:.1f}")
        
        # Clear cache for next pattern
        flex_model.clear_cache()


def demonstrate_paged_attention_integration():
    """
    Demonstrate PagedAttention integration mentioned in Chapter 18.
    Shows how to scatter logical KV blocks to physical memory layout.
    """
    print("\n=== PagedAttention Integration Demo ===")
    
    # Parameters
    batch_size = 4
    num_heads = 8
    head_dim = 64
    block_size = 16  # Tokens per page
    num_blocks = 32
    
    # Simulate logical KV blocks (what the model sees)
    logical_k = torch.randn(batch_size, num_blocks * block_size, num_heads, head_dim)
    logical_v = torch.randn(batch_size, num_blocks * block_size, num_heads, head_dim)
    
    # Physical memory layout (how it's actually stored)
    physical_k = torch.zeros(num_blocks, block_size, num_heads, head_dim)
    physical_v = torch.zeros(num_blocks, block_size, num_heads, head_dim)
    
    # Block mapping table (which logical blocks map to which physical blocks)
    block_table = torch.randint(0, num_blocks, (batch_size, num_blocks))
    
    def scatter_to_physical(logical_kv, physical_kv, block_table, batch_idx):
        """Scatter logical KV blocks to physical memory layout."""
        seq_len = logical_kv.shape[1]
        blocks_in_seq = seq_len // block_size
        
        for logical_block_idx in range(blocks_in_seq):
            # Get physical block index from table
            physical_block_idx = block_table[batch_idx, logical_block_idx]
            
            # Copy logical block to physical location
            logical_start = logical_block_idx * block_size
            logical_end = logical_start + block_size
            
            physical_kv[physical_block_idx] = logical_kv[batch_idx, logical_start:logical_end]
    
    # Demonstrate scattering for one batch
    print("Scattering logical KV blocks to physical memory...")
    scatter_to_physical(logical_k, physical_k, block_table, 0)
    scatter_to_physical(logical_v, physical_v, block_table, 0)
    
    print(f"Logical shape: {logical_k.shape}")
    print(f"Physical shape: {physical_k.shape}")
    print(f"Block table shape: {block_table.shape}")
    print("PagedAttention allows efficient sharing of KV blocks between sequences")


def main():
    """Main demonstration of FlexDecoding capabilities."""
    print("Chapter 18: FlexDecoding for Autoregressive Inference")
    print("=" * 55)
    
    # Check PyTorch version and capabilities
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Demonstrate basic FlexDecoding
    print("\n=== Basic FlexDecoding Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexDecodingAttention(512, 8, 1024).to(device)
    
    # Compile kernels
    model.compile_kernels("causal")
    
    # Test prefill
    prompt = torch.randn(1, 64, 512, device=device)
    print("Running prefill phase...")
    output = model.prefill(prompt)
    print(f"Prefill output shape: {output.shape}")
    
    # Test decode steps
    print("Running decode steps...")
    for step in range(5):
        token = torch.randn(1, 1, 512, device=device)
        output = model.decode_step(token, 64 + step)
        print(f"Decode step {step + 1} output shape: {output.shape}")
    
    # Demonstrate jagged tensor support
    print("\n=== Jagged Tensor Demo ===")
    sequences = [
        torch.randn(32, 512),   # Short sequence
        torch.randn(64, 512),   # Medium sequence  
        torch.randn(128, 512),  # Long sequence
    ]
    
    jagged_demo = NestedJaggedTensorDemo()
    outputs = jagged_demo.batch_inference_jagged(model, sequences)
    
    print(f"Processed {len(sequences)} variable-length sequences")
    for i, out in enumerate(outputs):
        print(f"  Sequence {i + 1}: input {sequences[i].shape[0]} -> output {out.shape[1]} tokens")
    
    # Run benchmarks
    if torch.cuda.is_available():
        benchmark_flexdecoding()
    
    # Demonstrate PagedAttention integration
    demonstrate_paged_attention_integration()
    
    print(f"\n=== Key FlexDecoding Benefits ===")
    print("- JIT-compiled kernels for arbitrary attention patterns")
    print("- Separate optimization for prefill vs decode phases")
    print("- Support for variable-length sequences (jagged tensors)")
    print("- Integration with PagedAttention for memory efficiency")
    print("- No custom CUDA required - pure PyTorch solution")
    print("- Near-optimal performance for complex sparsity patterns")


if __name__ == "__main__":
    main()
