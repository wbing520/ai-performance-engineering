import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
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
FlexDecoding example from Chapter 18.
This demonstrates nested tensor operations for variable-length sequences.
"""

import torch
import torch.nn as nn
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FlexDecodingConfig:
    """Configuration for FlexDecoding"""
    max_batch_size: int = 32
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    use_nested_tensors: bool = True

class FlexDecodingAttention(nn.Module):
    """FlexDecoding attention with nested tensor support"""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        # Scaling factor
        self.scale = head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, kv_cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with FlexDecoding optimizations
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim] or nested tensor
            kv_cache: Optional KV cache [batch_size, cache_len, 2, num_heads, head_dim]
        
        Returns:
            output: Attention output
            new_kv_cache: Updated KV cache
        """
        batch_size = x.size(0) if not x.is_nested else x.size(0)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if kv_cache is not None:
            # Concatenate with cache
            cache_k = kv_cache[:, :, 0]  # [batch_size, cache_len, num_heads, head_dim]
            cache_v = kv_cache[:, :, 1]  # [batch_size, cache_len, num_heads, head_dim]
            
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        seq_len = k.size(1)
        if scores.size(-1) > 1:  # Not first token
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Project output
        output = self.o_proj(attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim))
        
        # Update KV cache
        new_kv_cache = torch.stack([k, v], dim=2)  # [batch_size, seq_len, 2, num_heads, head_dim]
        
        return output, new_kv_cache

class FlexDecodingModel(nn.Module):
    """Model with FlexDecoding optimizations"""
    
    def __init__(self, config: FlexDecodingConfig):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(32000, config.hidden_dim)
        
        # FlexDecoding attention layers
        self.attention_layers = nn.ModuleList([
            FlexDecodingAttention(config.hidden_dim, config.num_heads, config.head_dim)
            for _ in range(4)  # Simplified: 4 layers
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim)
            )
            for _ in range(4)
        ])
        
        # Layer norms
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(4)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(4)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, 32000)
        
    def forward(self, input_ids: torch.Tensor, kv_cache: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with FlexDecoding
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len] or nested tensor
            kv_cache: Optional list of KV caches for each layer
        
        Returns:
            logits: Output logits
            new_kv_cache: Updated KV caches for each layer
        """
        # Embed input
        x = self.embedding(input_ids)
        x = self.input_norm(x)
        
        # Initialize KV cache if not provided
        if kv_cache is None:
            kv_cache = [None] * len(self.attention_layers)
        
        new_kv_cache = []
        
        # Process through layers
        for i, (attn_layer, ff_layer, attn_norm, ff_norm) in enumerate(
            zip(self.attention_layers, self.ff_layers, self.attention_norms, self.ff_norms)
        ):
            # Attention layer
            attn_input = attn_norm(x)
            attn_output, new_cache = attn_layer(attn_input, kv_cache[i])
            x = x + attn_output
            new_kv_cache.append(new_cache)
            
            # Feed-forward layer
            ff_input = ff_norm(x)
            ff_output = ff_layer(ff_input)
            x = x + ff_output
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits, new_kv_cache

class FlexDecodingInference:
    """FlexDecoding inference engine"""
    
    def __init__(self, config: FlexDecodingConfig):
        self.config = config
        self.model = FlexDecodingModel(config)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Warm up
        self._warmup()
    
    def _warmup(self):
        """Warm up the model for better performance"""
        print("Warming up FlexDecoding model...")
        
        # Create dummy input
        dummy_input = torch.randint(0, 32000, (1, 1), device=self.device)
        
        # Run a few forward passes
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        print("Warmup completed!")
    
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 100) -> List[int]:
        """
        Generate tokens using FlexDecoding
        
        Args:
            prompt_tokens: Input prompt tokens
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated token sequence
        """
        # Convert to tensor
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        
        # Prefill phase (if prompt is longer than 1 token)
        kv_cache = None
        if len(prompt_tokens) > 1:
            print(f"Prefill phase: processing {len(prompt_tokens)} tokens...")
            start_time = time.time()
            
            with torch.no_grad():
                _, kv_cache = self.model(input_ids)
            
            prefill_time = time.time() - start_time
            print(f"Prefill completed in {prefill_time:.3f}s")
        
        # Decode phase
        print(f"Decode phase: generating {max_new_tokens} tokens...")
        generated_tokens = prompt_tokens.copy()
        
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # Get last token
                last_token = torch.tensor([[generated_tokens[-1]]], device=self.device)
                
                # Forward pass
                logits, kv_cache = self.model(last_token, kv_cache)
                
                # Sample next token (greedy for simplicity)
                next_token = torch.argmax(logits[0, -1]).item()
                generated_tokens.append(next_token)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{max_new_tokens} tokens...")
        
        decode_time = time.time() - start_time
        print(f"Decode completed in {decode_time:.3f}s")
        print(f"Average decode time per token: {decode_time/max_new_tokens:.3f}s")
        
        return generated_tokens

def create_nested_tensor_example():
    """Example of using nested tensors with FlexDecoding"""
    print("\n=== Nested Tensor Example ===")
    
    # Create nested tensor with variable lengths
    lengths = [5, 3, 7, 2]
    max_len = max(lengths)
    
    # Create padded tensor
    padded_tensor = torch.randn(len(lengths), max_len, 4096)
    
    # Convert to nested tensor
    nested_tensor = torch.nested.nested_tensor([
        padded_tensor[i, :lengths[i]] for i in range(len(lengths))
    ])
    
    print(f"Nested tensor shape: {nested_tensor.shape}")
    print(f"Nested tensor sizes: {nested_tensor.size(1)}")
    
    # Create model
    config = FlexDecodingConfig(use_nested_tensors=True)
    model = FlexDecodingModel(config)
    
    # Forward pass with nested tensor
    with torch.no_grad():
        logits, kv_cache = model(nested_tensor)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"KV cache length: {len(kv_cache)}")

def main():
    """Main function demonstrating FlexDecoding"""
    print("FlexDecoding Example from Chapter 18")
    print("====================================")
    
    # Configuration
    config = FlexDecodingConfig(
        max_batch_size=8,
        max_seq_len=2048,
        hidden_dim=4096,
        num_heads=32,
        head_dim=128,
        use_nested_tensors=True
    )
    
    print(f"Configuration:")
    print(f"  Max Batch Size: {config.max_batch_size}")
    print(f"  Max Seq Len: {config.max_seq_len}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Num Heads: {config.num_heads}")
    print(f"  Head Dim: {config.head_dim}")
    print(f"  Use Nested Tensors: {config.use_nested_tensors}")
    
    # Create inference engine
    inference = FlexDecodingInference(config)
    
    # Example prompt
    prompt = [1, 2, 3, 4, 5]  # Dummy token IDs
    print(f"\nInput prompt: {prompt}")
    
    # Generate tokens
    generated = inference.generate(prompt, max_new_tokens=20)
    print(f"Generated sequence: {generated}")
    
    # Demonstrate nested tensor usage
    create_nested_tensor_example()
    
    print("\nFlexDecoding example completed successfully!")

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == "10.0":  # Blackwell B200/B300
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.8 features
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.autotune_mode = "max-autotune"
    torch._dynamo.config.automatic_dynamic_shapes = True
