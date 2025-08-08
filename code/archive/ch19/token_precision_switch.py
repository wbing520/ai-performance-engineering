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
Token-level precision switching during generation.
This demonstrates dynamic precision adaptation based on model confidence.
"""

import torch
import torch.nn as nn
import time
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PrecisionMode(Enum):
    """Available precision modes"""
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"

@dataclass
class TokenConfidence:
    """Token confidence metrics"""
    max_probability: float
    entropy: float
    top1_top2_diff: float
    logit_variance: float

class TokenPrecisionSwitcher:
    """Manages token-level precision switching"""
    
    def __init__(self, confidence_threshold: float = 2.0):
        self.confidence_threshold = confidence_threshold
        self.current_precision = PrecisionMode.FP16
        self.precision_history = []
        self.switch_count = 0
        
        # Precision usage statistics
        self.precision_usage = {
            PrecisionMode.FP16: 0,
            PrecisionMode.FP8: 0,
            PrecisionMode.FP4: 0
        }
    
    def evaluate_confidence(self, logits: torch.Tensor) -> TokenConfidence:
        """
        Evaluate confidence based on logits
        
        Args:
            logits: Model output logits [batch_size, vocab_size]
            
        Returns:
            Confidence metrics
        """
        # Get top-2 probabilities
        top_probs, top_indices = torch.softmax(logits, dim=-1).topk(2, dim=-1)
        top1_prob = top_probs[:, 0]
        top2_prob = top_probs[:, 1]
        
        # Calculate confidence metrics
        max_probability = top1_prob.mean().item()
        top1_top2_diff = (top1_prob - top2_prob).mean().item()
        
        # Calculate entropy
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
        
        # Calculate logit variance
        logit_variance = logits.var(dim=-1).mean().item()
        
        return TokenConfidence(
            max_probability=max_probability,
            entropy=entropy,
            top1_top2_diff=top1_top2_diff,
            logit_variance=logit_variance
        )
    
    def should_switch_precision(self, confidence: TokenConfidence) -> Optional[PrecisionMode]:
        """
        Determine if precision should be switched based on confidence
        
        Args:
            confidence: Token confidence metrics
            
        Returns:
            New precision mode if switch is needed, None otherwise
        """
        # Use top1-top2 difference as primary confidence metric
        confidence_score = confidence.top1_top2_diff
        
        if self.current_precision == PrecisionMode.FP16:
            # Switch to FP8 if confidence is high
            if confidence_score > self.confidence_threshold:
                return PrecisionMode.FP8
        elif self.current_precision == PrecisionMode.FP8:
            # Switch to FP4 if confidence is very high
            if confidence_score > self.confidence_threshold * 1.5:
                return PrecisionMode.FP4
            # Switch back to FP16 if confidence drops
            elif confidence_score < self.confidence_threshold * 0.5:
                return PrecisionMode.FP16
        elif self.current_precision == PrecisionMode.FP4:
            # Switch back to FP8 if confidence drops
            if confidence_score < self.confidence_threshold * 1.2:
                return PrecisionMode.FP8
            # Switch back to FP16 if confidence drops significantly
            elif confidence_score < self.confidence_threshold * 0.3:
                return PrecisionMode.FP16
        
        return None
    
    def switch_precision(self, new_precision: PrecisionMode):
        """Switch to new precision mode"""
        if new_precision != self.current_precision:
            print(f"Switching precision: {self.current_precision.value} → {new_precision.value}")
            
            # Record the switch
            self.precision_history.append({
                "from": self.current_precision.value,
                "to": new_precision.value,
                "timestamp": time.time()
            })
            
            self.current_precision = new_precision
            self.switch_count += 1
        
        # Update usage statistics
        self.precision_usage[new_precision] += 1
    
    def get_precision_stats(self) -> dict:
        """Get precision switching statistics"""
        total_tokens = sum(self.precision_usage.values())
        
        return {
            "total_switches": self.switch_count,
            "current_precision": self.current_precision.value,
            "precision_usage": {
                mode.value: count for mode, count in self.precision_usage.items()
            },
            "precision_percentages": {
                mode.value: (count / total_tokens * 100) if total_tokens > 0 else 0
                for mode, count in self.precision_usage.items()
            },
            "switch_history": self.precision_history
        }

class MockLLMModel(nn.Module):
    """Mock LLM model for demonstration"""
    
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple transformer-like architecture
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=32,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(4)  # Simplified: 4 layers
        ])
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with precision switching"""
        x = self.embedding(input_ids)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        logits = self.output_proj(x)
        return logits

class AdaptiveInferenceEngine:
    """Inference engine with token-level precision switching"""
    
    def __init__(self, confidence_threshold: float = 2.0):
        self.model = MockLLMModel()
        self.precision_switcher = TokenPrecisionSwitcher(confidence_threshold)
        
        # Performance tracking
        self.total_tokens = 0
        self.start_time = time.time()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
    
    def generate_with_precision_switching(self, prompt_tokens: List[int], 
                                        max_new_tokens: int = 50) -> List[int]:
        """
        Generate tokens with dynamic precision switching
        
        Args:
            prompt_tokens: Input prompt tokens
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated token sequence
        """
        generated_tokens = prompt_tokens.copy()
        
        print(f"Generating {max_new_tokens} tokens with precision switching...")
        print(f"Initial precision: {self.precision_switcher.current_precision.value}")
        
        for i in range(max_new_tokens):
            # Prepare input
            input_ids = torch.tensor([generated_tokens], device=self.device)
            
            # Forward pass with current precision
            with torch.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu",
                               dtype=self._get_torch_dtype()):
                logits = self.model(input_ids)
            
            # Evaluate confidence
            confidence = self.precision_switcher.evaluate_confidence(logits)
            
            # Check if precision should be switched
            new_precision = self.precision_switcher.should_switch_precision(confidence)
            if new_precision:
                self.precision_switcher.switch_precision(new_precision)
            
            # Sample next token (greedy for simplicity)
            next_token = torch.argmax(logits[0, -1]).item()
            generated_tokens.append(next_token)
            
            # Print progress and confidence
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{max_new_tokens} tokens "
                      f"(precision: {self.precision_switcher.current_precision.value}, "
                      f"confidence: {confidence.top1_top2_diff:.3f})")
            
            self.total_tokens += 1
        
        return generated_tokens
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype for current precision mode"""
        if self.precision_switcher.current_precision == PrecisionMode.FP16:
            return torch.float16
        elif self.precision_switcher.current_precision == PrecisionMode.FP8:
            return torch.float8_e4m3fn  # Use FP8 E4M3 format
        elif self.precision_switcher.current_precision == PrecisionMode.FP4:
            # Note: PyTorch doesn't have native FP4, so we'll use FP8 as approximation
            return torch.float8_e4m3fn
        else:
            return torch.float16
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        runtime = time.time() - self.start_time
        tokens_per_sec = self.total_tokens / runtime if runtime > 0 else 0
        
        return {
            "total_tokens": self.total_tokens,
            "runtime_seconds": runtime,
            "tokens_per_sec": tokens_per_sec,
            "precision_stats": self.precision_switcher.get_precision_stats()
        }

def create_sample_prompts() -> List[List[int]]:
    """Create sample prompts with different characteristics"""
    prompts = []
    
    # High-confidence prompt (likely to trigger precision switching)
    high_confidence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Sequential tokens
    prompts.append(high_confidence)
    
    # Medium-confidence prompt
    medium_confidence = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    prompts.append(medium_confidence)
    
    # Low-confidence prompt (ambiguous)
    low_confidence = [random.randint(0, 1000) for _ in range(10)]
    prompts.append(low_confidence)
    
    return prompts

def main():
    """Main function demonstrating token-level precision switching"""
    print("Token-Level Precision Switching Example")
    print("======================================")
    
    # Create adaptive inference engine
    engine = AdaptiveInferenceEngine(confidence_threshold=2.0)
    
    # Create sample prompts
    prompts = create_sample_prompts()
    
    # Generate tokens for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i + 1} ---")
        print(f"Input tokens: {prompt[:5]}...")  # Show first 5 tokens
        
        # Generate tokens with precision switching
        generated = engine.generate_with_precision_switching(prompt, max_new_tokens=30)
        
        print(f"Generated {len(generated) - len(prompt)} new tokens")
        print(f"Final sequence: {generated[:10]}...")  # Show first 10 tokens
    
    # Print final statistics
    stats = engine.get_performance_stats()
    print("\n=== Final Performance Statistics ===")
    print(f"Total Tokens Generated: {stats['total_tokens']}")
    print(f"Runtime: {stats['runtime_seconds']:.2f} seconds")
    print(f"Throughput: {stats['tokens_per_sec']:.1f} tokens/sec")
    
    precision_stats = stats['precision_stats']
    print(f"\nPrecision Switching Statistics:")
    print(f"Total Switches: {precision_stats['total_switches']}")
    print(f"Final Precision: {precision_stats['current_precision']}")
    
    print(f"\nPrecision Usage:")
    for precision, count in precision_stats['precision_usage'].items():
        percentage = precision_stats['precision_percentages'][precision]
        print(f"  {precision.upper()}: {count} tokens ({percentage:.1f}%)")
    
    print(f"\nPrecision Switch History:")
    for i, switch in enumerate(precision_stats['switch_history']):
        print(f"  {i+1}. {switch['from'].upper()} → {switch['to'].upper()}")
    
    print("\nToken-level precision switching example completed successfully!")

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
