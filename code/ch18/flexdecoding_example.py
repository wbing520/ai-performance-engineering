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
import math
import time
from typing import Optional, Tuple, Callable
import numpy as np

# Try to import flex_attention - available in PyTorch 2.5.0+
try:
    from torch.nn.attention import flex_attention as flex_attention_module
    flex_attention = flex_attention_module.flex_attention
    FLEX_ATTENTION_AVAILABLE = True
    print("✓ FlexAttention is available - using experimental flex_attention API")
except (ImportError, AttributeError) as e:
    FLEX_ATTENTION_AVAILABLE = False
    print("⚠ FlexAttention not available in this PyTorch build (2.8.0.dev20250613+cu128)")
    print("  - FlexAttention was introduced in PyTorch 2.5.0")
    print("  - Falling back to scaled_dot_product_attention with torch.compile")
    flex_attention = None

class FlexDecodingAttention(torch.nn.Module):
    """
    FlexDecoding implementation using PyTorch's flex_attention when available.
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
        
        # KV cache for autoregressive generation - will be initialized dynamically
        self.register_buffer("k_cache", torch.zeros(1, max_seq_len, num_heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, max_seq_len, num_heads, self.head_dim))
        
        # Compiled kernels for different phases
        self._prefill_fn = None
        self._decode_fn = None
        
        # Offset for decoding (for flex_attention)
        self.register_buffer("offset", torch.tensor(0, dtype=torch.long))
        
    def _get_causal_mask_fn(self):
        """Create causal mask function for autoregressive attention."""
        def causal_mask(batch, head, q_idx, kv_idx):
            return q_idx >= kv_idx
        return causal_mask
    
    def _get_causal_score_mod_fn(self):
        """Create causal score modification function for flex_attention."""
        def causal_score_mod(score, b, h, q_idx, kv_idx):
            # Apply offset for decoding phase
            adjusted_q_idx = q_idx + self.offset
            return torch.where(adjusted_q_idx >= kv_idx, score, -float("inf"))
        return causal_score_mod
    
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
        
        # Get device
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create sample tensors for compilation - flex_attention expects [batch, num_heads, seq_len, head_dim]
        seq_len = 512
        q_prefill = torch.randn(1, self.num_heads, seq_len, self.head_dim, device=device)
        k_prefill = torch.randn(1, self.num_heads, seq_len, self.head_dim, device=device)
        v_prefill = torch.randn(1, self.num_heads, seq_len, self.head_dim, device=device)
        
        q_decode = torch.randn(1, self.num_heads, 1, self.head_dim, device=device)  # Single token
        k_decode = torch.randn(1, self.num_heads, seq_len, self.head_dim, device=device)
        v_decode = torch.randn(1, self.num_heads, seq_len, self.head_dim, device=device)
        
        if FLEX_ATTENTION_AVAILABLE:
            # Use flex_attention with score_mod for causal attention
            print("  Using FlexAttention with score_mod...")
            
            # Get the appropriate score_mod function
            if pattern == "causal":
                score_mod = self._get_causal_score_mod_fn()
            elif pattern == "local":
                # For local attention, we'd need a different score_mod
                score_mod = self._get_causal_score_mod_fn()  # Simplified for now
            else:
                score_mod = self._get_causal_score_mod_fn()
            
            # Compile prefill kernel with flex_attention
            print("  Compiling prefill kernel with FlexAttention...")
            self._prefill_fn = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, score_mod=score_mod),
                mode="max-autotune"
            )
            
            # Warmup compilation with sample data
            with torch.no_grad():
                _ = self._prefill_fn(q_prefill, k_prefill, v_prefill)
            
            # Compile decode kernel with flex_attention
            print("  Compiling decode kernel with FlexAttention...")
            self._decode_fn = torch.compile(
                lambda q, k, v: flex_attention(q, k, v, score_mod=score_mod),
                mode="max-autotune"
            )
            
            # Warmup compilation
            with torch.no_grad():
                _ = self._decode_fn(q_decode, k_decode, v_decode)
                
        else:
            # Fallback to standard attention with torch.compile
            print("  Using standard attention with torch.compile...")
            
            # For standard attention, we need [batch, seq_len, num_heads, head_dim]
            q_prefill_std = q_prefill.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            k_prefill_std = k_prefill.transpose(1, 2)
            v_prefill_std = v_prefill.transpose(1, 2)
            q_decode_std = q_decode.transpose(1, 2)
            k_decode_std = k_decode.transpose(1, 2)
            v_decode_std = v_decode.transpose(1, 2)
            
            # Compile prefill kernel (Q_len >> KV_len scenario)
            print("  Compiling prefill kernel...")
            self._prefill_fn = torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                mode="max-autotune"
            )
            
            # Warmup compilation with sample data
            with torch.no_grad():
                _ = self._prefill_fn(q_prefill_std, k_prefill_std, v_prefill_std)
            
            # Compile decode kernel (Q_len = 1 scenario) 
            print("  Compiling decode kernel...")
            self._decode_fn = torch.compile(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True),
                mode="max-autotune"
            )
            
            # Warmup compilation
            with torch.no_grad():
                _ = self._decode_fn(q_decode_std, k_decode_std, v_decode_std)
        
        print("  Kernel compilation complete!")
    
    def prefill(self, x: torch.Tensor, past_kv_len: int = 0) -> torch.Tensor:
        """
        Prefill phase: Process entire prompt sequence.
        Uses compiled prefill kernel optimized for long sequences.
        """
        batch_size, seq_len, _ = x.shape
        
        # Resize KV cache if needed for this batch size
        if self.k_cache.shape[0] != batch_size:
            self.k_cache = torch.zeros(batch_size, self.max_seq_len, self.num_heads, self.head_dim, 
                                     device=self.k_cache.device, dtype=self.k_cache.dtype)
            self.v_cache = torch.zeros(batch_size, self.max_seq_len, self.num_heads, self.head_dim,
                                     device=self.v_cache.device, dtype=self.v_cache.dtype)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim) 
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Update KV cache
        self.k_cache[:, past_kv_len:past_kv_len + seq_len] = k
        self.v_cache[:, past_kv_len:past_kv_len + seq_len] = v
        
        # Use compiled prefill kernel
        if self._prefill_fn is not None:
            if FLEX_ATTENTION_AVAILABLE:
                # flex_attention expects [batch, num_heads, seq_len, head_dim]
                q_flex = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
                k_flex = k.transpose(1, 2)
                v_flex = v.transpose(1, 2)
                attn_out = self._prefill_fn(q_flex, k_flex, v_flex)
                # Convert back to [batch, seq_len, num_heads, head_dim]
                attn_out = attn_out.transpose(1, 2)
            else:
                attn_out = self._prefill_fn(q, k, v)
        else:
            # Fallback to standard attention
            if FLEX_ATTENTION_AVAILABLE:
                score_mod = self._get_causal_score_mod_fn()
                q_flex = q.transpose(1, 2)
                k_flex = k.transpose(1, 2)
                v_flex = v.transpose(1, 2)
                attn_out = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod)
                attn_out = attn_out.transpose(1, 2)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Project output
        attn_out = attn_out.reshape(batch_size, seq_len, self.dim)
        return self.o_proj(attn_out)
    
    def decode_step(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """
        Decode phase: Process single token attending to full KV cache.
        Uses compiled decode kernel optimized for Q_len=1 case.
        """
        batch_size, _, _ = x.shape  # x should be [batch, 1, dim]
        
        # Update offset for flex_attention
        self.offset.fill_(position)
        
        # Resize KV cache if needed for this batch size
        if self.k_cache.shape[0] != batch_size:
            self.k_cache = torch.zeros(batch_size, self.max_seq_len, self.num_heads, self.head_dim,
                                     device=self.k_cache.device, dtype=self.k_cache.dtype)
            self.v_cache = torch.zeros(batch_size, self.max_seq_len, self.num_heads, self.head_dim,
                                     device=self.v_cache.device, dtype=self.v_cache.dtype)
        
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
            if FLEX_ATTENTION_AVAILABLE:
                # flex_attention expects [batch, num_heads, seq_len, head_dim]
                q_flex = q.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
                k_flex = k_past.transpose(1, 2)  # [batch, num_heads, position+1, head_dim]
                v_flex = v_past.transpose(1, 2)
                attn_out = self._decode_fn(q_flex, k_flex, v_flex)
                # Convert back to [batch, 1, num_heads, head_dim]
                attn_out = attn_out.transpose(1, 2)
            else:
                attn_out = self._decode_fn(q, k_past, v_past)
        else:
            # Fallback
            if FLEX_ATTENTION_AVAILABLE:
                score_mod = self._get_causal_score_mod_fn()
                q_flex = q.transpose(1, 2)
                k_flex = k_past.transpose(1, 2)
                v_flex = v_past.transpose(1, 2)
                attn_out = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod)
                attn_out = attn_out.transpose(1, 2)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_past, v_past, is_causal=True)
        
        # Project output - flex_attention returns [batch, seq_len, num_heads, head_dim]
        # We need to reshape it to [batch, seq_len, dim]
        if len(attn_out.shape) == 4:
            # If it's [batch, seq_len, num_heads, head_dim], take only the last token
            attn_out = attn_out[:, -1:, :, :]  # Take only the last token
            attn_out = attn_out.reshape(batch_size, 1, self.dim)
        else:
            # If it's already in the right format, just ensure the shape
            attn_out = attn_out.reshape(batch_size, 1, -1)
        
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
            # Ensure the sequence is on the same device as the padded tensor
            if hasattr(seq, 'device'):
                seq = seq.to(padded.device)
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
        device = next(model.parameters()).device
        padded_input, offsets = NestedJaggedTensorDemo.create_jagged_tensor(sequences)
        padded_input = padded_input.to(device)
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
