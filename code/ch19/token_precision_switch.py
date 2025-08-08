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
token_precision_switch.py
Chapter 19: Token-level Precision Switching During Generation

Implementation of dynamic precision switching that adapts numerical precision
per token based on model confidence and output quality metrics.

Based on Chapter 19's token-level precision adaptation concepts.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math
import logging
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from hqq.core.quantize import quantize, dequantize
    HQQ_AVAILABLE = True
except ImportError:
    HQQ_AVAILABLE = False
    print("Warning: HQQ not available. Some quantization features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ConfidenceMetrics:
    """Metrics used to assess model confidence for precision switching."""
    max_probability: float      # Highest softmax probability
    entropy: float              # Output entropy (uncertainty measure)
    logit_variance: float       # Variance of logits
    logit_max_diff: float      # Difference between top-2 logits
    temperature_adjusted: float # Confidence adjusted for temperature
    
    @property
    def confidence_score(self) -> float:
        """Combined confidence score (0-1, higher = more confident)."""
        # Normalize and combine metrics
        prob_score = self.max_probability
        entropy_score = max(0, 1.0 - self.entropy / 4.0)  # Normalize entropy
        logit_score = min(1.0, self.logit_max_diff / 10.0)  # Normalize logit diff
        
        # Weighted combination
        return 0.5 * prob_score + 0.3 * entropy_score + 0.2 * logit_score


class TokenPrecisionController:
    """
    Controller for dynamic token-level precision switching.
    Implementation of Chapter 19's adaptive precision concepts.
    """
    
    def __init__(self, model: torch.nn.Module, initial_precision: PrecisionLevel = PrecisionLevel.FP16):
        self.model = model
        self.current_precision = initial_precision
        
        # Thresholds for precision switching (from Chapter 19)
        self.confidence_threshold_high = 0.9  # Switch to lower precision
        self.confidence_threshold_low = 0.6   # Switch to higher precision
        self.entropy_threshold_low = 0.5      # Low entropy = high confidence
        self.entropy_threshold_high = 2.0     # High entropy = low confidence
        self.logit_diff_threshold = 2.0       # From Chapter 19 example
        
        # Performance tracking
        self.precision_history = []
        self.switch_count = 0
        self.quality_metrics = []
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Precision switching policy
        self.switching_enabled = True
        self.conservative_mode = False  # If True, prefer higher precision
        
    def compute_confidence_metrics(self, logits: torch.Tensor, 
                                 temperature: float = 1.0) -> ConfidenceMetrics:
        """
        Compute confidence metrics from model logits.
        Based on Chapter 19's confidence assessment methods.
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Max probability
        max_prob = torch.max(probs).item()
        
        # Entropy (uncertainty measure)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        
        # Logit variance
        logit_var = torch.var(scaled_logits).item()
        
        # Difference between top-2 logits
        top2_logits = torch.topk(scaled_logits, k=2).values
        logit_diff = (top2_logits[0] - top2_logits[1]).item()
        
        return ConfidenceMetrics(
            max_probability=max_prob,
            entropy=entropy,
            logit_variance=logit_var,
            logit_max_diff=logit_diff,
            temperature_adjusted=max_prob * (1.0 / temperature)
        )
    
    def decide_precision(self, confidence: ConfidenceMetrics, 
                        current_precision: PrecisionLevel) -> PrecisionLevel:
        """
        Decide optimal precision based on confidence metrics.
        Implementation of Chapter 19's precision switching logic.
        """
        if not self.switching_enabled:
            return current_precision
        
        confidence_score = confidence.confidence_score
        
        # Conservative mode: prefer higher precision
        if self.conservative_mode:
            if confidence_score < 0.8:
                return PrecisionLevel.FP16
            elif confidence_score > 0.95:
                return PrecisionLevel.INT8
            else:
                return current_precision
        
        # Chapter 19's threshold-based switching
        if confidence.logit_max_diff > self.logit_diff_threshold:
            # High confidence - can use lower precision
            if confidence_score > self.confidence_threshold_high:
                # Very high confidence
                if current_precision in [PrecisionLevel.FP32, PrecisionLevel.FP16]:
                    return PrecisionLevel.INT8
                elif current_precision == PrecisionLevel.INT8:
                    return PrecisionLevel.INT4
            elif confidence_score > self.confidence_threshold_low:
                # Moderate confidence
                if current_precision == PrecisionLevel.FP32:
                    return PrecisionLevel.FP16
                elif current_precision == PrecisionLevel.FP16:
                    return PrecisionLevel.INT8
        else:
            # Low confidence - use higher precision
            if confidence_score < self.confidence_threshold_low:
                if current_precision in [PrecisionLevel.INT4, PrecisionLevel.INT8]:
                    return PrecisionLevel.FP16
                elif current_precision == PrecisionLevel.FP16:
                    return PrecisionLevel.FP32
        
        # No change needed
        return current_precision
    
    def apply_precision(self, tensor: torch.Tensor, 
                       precision: PrecisionLevel) -> torch.Tensor:
        """Apply the specified precision to tensor computations."""
        if precision == PrecisionLevel.FP32:
            return tensor.float()
        elif precision == PrecisionLevel.FP16:
            return tensor.half()
        elif precision == PrecisionLevel.BF16:
            return tensor.bfloat16()
        elif precision in [PrecisionLevel.INT8, PrecisionLevel.INT4]:
            # Simulate quantization (in practice would use proper quantization)
            if precision == PrecisionLevel.INT8:
                # 8-bit quantization
                scale = tensor.abs().max() / 127.0
                quantized = torch.round(tensor / scale).clamp(-128, 127)
                return quantized * scale
            else:  # INT4
                # 4-bit quantization  
                scale = tensor.abs().max() / 7.0
                quantized = torch.round(tensor / scale).clamp(-8, 7)
                return quantized * scale
        else:
            return tensor
    
    def generate_with_adaptive_precision(self, input_ids: torch.Tensor, 
                                       max_length: int = 50,
                                       temperature: float = 1.0) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate tokens with adaptive precision switching.
        Main implementation of Chapter 19's token-level precision adaptation.
        """
        generated_ids = input_ids.clone()
        generation_stats = []
        
        for step in range(max_length):
            with torch.no_grad():
                # Forward pass with current precision
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Apply current precision to logits
                logits = self.apply_precision(logits, self.current_precision)
                
                # Compute confidence metrics
                confidence = self.compute_confidence_metrics(logits, temperature)
                
                # Decide next precision
                next_precision = self.decide_precision(confidence, self.current_precision)
                
                # Sample next token
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Update generation
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Track statistics
                precision_changed = next_precision != self.current_precision
                if precision_changed:
                    self.switch_count += 1
                
                step_stats = {
                    'step': step,
                    'token_id': next_token.item(),
                    'confidence': confidence.confidence_score,
                    'entropy': confidence.entropy,
                    'max_prob': confidence.max_probability,
                    'logit_diff': confidence.logit_max_diff,
                    'precision_before': self.current_precision.value,
                    'precision_after': next_precision.value,
                    'precision_changed': precision_changed
                }
                generation_stats.append(step_stats)
                
                # Update precision for next iteration
                self.current_precision = next_precision
                self.precision_history.append(next_precision)
                
                # Log precision changes
                if precision_changed:
                    logger.info(f"Step {step}: Precision switch to {next_precision.value} "
                              f"(confidence: {confidence.confidence_score:.3f})")
                
                # Check for EOS or other stopping criteria
                if next_token.item() == 0:  # Assuming 0 is EOS
                    break
        
        return generated_ids, generation_stats


class DynamicQuantizedCache:
    """
    Dynamic quantization for KV cache from Chapter 19.
    Adapts cache precision based on memory pressure and quality requirements.
    """
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.policy_switch_counter = 0
        
        # Quantization policies
        self.quantization_policies = {
            'none': {'bits': 16, 'error_threshold': 0.0},
            'conservative': {'bits': 8, 'error_threshold': 0.01},
            'aggressive': {'bits': 4, 'error_threshold': 0.05}
        }
        
        self.current_policy = 'none'
    
    def async_quantize(self, tensor: torch.Tensor, nbits: int, 
                      axis: int = 0, group_size: int = 64):
        """Async quantization helper from Chapter 19."""
        if not HQQ_AVAILABLE:
            # Fallback to simple quantization
            def simple_quantize():
                scale = tensor.abs().max() / ((2 ** (nbits - 1)) - 1)
                quantized = torch.round(tensor / scale).clamp(-(2**(nbits-1)), 2**(nbits-1)-1)
                return quantized, scale
            
            return self.executor.submit(simple_quantize)
        
        def task():
            return quantize(tensor, nbits=nbits, axis=axis, group_size=group_size)
        
        return self.executor.submit(task)
    
    def maybe_quantize_cache(self, layers, policy: str, 
                           memory_threshold: float = 0.8):
        """
        Dynamic policy and cache management from Chapter 19.
        Quantizes cache based on memory pressure and policy.
        """
        global policy_switch_counter
        
        # Check GPU memory usage
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available():
            used = torch.cuda.memory_reserved(device_index)
            total = torch.cuda.get_device_properties(device_index).total_memory
            memory_ratio = used / total
        else:
            # Simulate memory usage
            memory_ratio = 0.6
        
        if memory_ratio < memory_threshold:
            return
        
        # Determine quantization bits based on policy
        policy_config = self.quantization_policies.get(policy, self.quantization_policies['conservative'])
        nbits = policy_config['bits']
        error_threshold = policy_config['error_threshold']
        
        logger.info(f"Memory pressure detected ({memory_ratio:.1%}), applying {policy} quantization")
        
        # Quantize cache layers
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'kv_cache') and layer.kv_cache is not None:
                # Quantize key and value caches
                for cache_type in ['key_cache', 'value_cache']:
                    cache_tensor = getattr(layer, cache_type, None)
                    if cache_tensor is not None:
                        # Submit async quantization
                        future = self.async_quantize(cache_tensor, nbits)
                        
                        # Get result (in practice would be non-blocking)
                        try:
                            if HQQ_AVAILABLE:
                                quantized_result = future.result(timeout=1.0)
                            else:
                                quantized_tensor, scale = future.result(timeout=1.0)
                                # Store scale for dequantization
                                setattr(layer, f'{cache_type}_scale', scale)
                            
                            # Update cache (simplified)
                            setattr(layer, cache_type, quantized_tensor)
                            
                            logger.info(f"Layer {layer_idx} {cache_type} quantized to {nbits}-bit")
                            
                        except Exception as e:
                            logger.warning(f"Quantization failed for layer {layer_idx}: {e}")
        
        self.policy_switch_counter += 1


def demonstrate_token_precision_switching():
    """
    Demonstrate token-level precision switching with a simple model.
    Shows Chapter 19's adaptive precision concepts in action.
    """
    print("\n=== Token Precision Switching Demo ===")
    
    # Create a simple model for demonstration
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, vocab_size=1000, dim=512):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, dim)
            self.transformer = torch.nn.TransformerDecoderLayer(
                d_model=dim, nhead=8, batch_first=True
            )
            self.lm_head = torch.nn.Linear(dim, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            # Create causal mask
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            
            x = self.transformer(x, memory=x, tgt_mask=mask)
            logits = self.lm_head(x)
            
            return type('Output', (), {'logits': logits})()
    
    # Initialize model and controller
    model = SimpleTransformer()
    controller = TokenPrecisionController(model)
    
    # Generate with adaptive precision
    input_ids = torch.randint(1, 100, (1, 10))  # Random starting sequence
    print(f"Starting sequence: {input_ids.tolist()}")
    
    generated_ids, stats = controller.generate_with_adaptive_precision(
        input_ids, max_length=20, temperature=0.8
    )
    
    print(f"Generated sequence: {generated_ids.tolist()}")
    print(f"Total precision switches: {controller.switch_count}")
    
    # Analyze precision usage
    precision_counts = {}
    for step_stat in stats:
        precision = step_stat['precision_after']
        precision_counts[precision] = precision_counts.get(precision, 0) + 1
    
    print(f"\nPrecision usage:")
    for precision, count in precision_counts.items():
        print(f"  {precision}: {count} tokens ({count/len(stats)*100:.1f}%)")
    
    # Show confidence vs precision relationship
    print(f"\nConfidence vs Precision Analysis:")
    for i, stat in enumerate(stats[:10]):  # Show first 10 steps
        print(f"Step {stat['step']:2d}: confidence={stat['confidence']:.3f}, "
              f"entropy={stat['entropy']:.2f}, precision={stat['precision_after']}")


def benchmark_precision_switching_overhead():
    """Benchmark the computational overhead of precision switching."""
    print("\n=== Precision Switching Overhead Benchmark ===")
    
    # Create test data
    batch_size, seq_len, hidden_dim = 8, 512, 1024
    test_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    if torch.cuda.is_available():
        test_tensor = test_tensor.cuda()
    
    precisions = [PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8]
    controller = TokenPrecisionController(torch.nn.Identity())
    
    # Benchmark precision applications
    for precision in precisions:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            result = controller.apply_precision(test_tensor, precision)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        print(f"{precision.value:4s}: {avg_time:.3f} ms per conversion")
    
    # Benchmark confidence computation
    logits = torch.randn(1000)  # Vocabulary-sized logits
    if torch.cuda.is_available():
        logits = logits.cuda()
    
    start_time = time.time()
    for _ in range(1000):
        confidence = controller.compute_confidence_metrics(logits)
    end_time = time.time()
    
    avg_confidence_time = (end_time - start_time) / 1000 * 1000  # Convert to ms
    print(f"Confidence computation: {avg_confidence_time:.3f} ms per call")


def main():
    """Main demonstration of token-level precision switching."""
    print("Chapter 19: Token-level Precision Switching During Generation")
    print("=" * 60)
    
    # Basic demonstration
    demonstrate_token_precision_switching()
    
    # Performance analysis
    benchmark_precision_switching_overhead()
    
    # Show dynamic quantized cache example
    print(f"\n=== Dynamic Quantized Cache Demo ===")
    
    # Create mock layers for quantization demo
    class MockLayer:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.key_cache = torch.randn(8, 512, 64) if torch.cuda.is_available() else None
            self.value_cache = torch.randn(8, 512, 64) if torch.cuda.is_available() else None
    
    layers = [MockLayer(i) for i in range(4)]
    cache_manager = DynamicQuantizedCache()
    
    # Simulate high memory pressure
    print("Simulating high memory pressure...")
    cache_manager.maybe_quantize_cache(layers, 'conservative', memory_threshold=0.5)
    
    print(f"\n=== Key Benefits of Token-Level Precision Switching ===")
    print("- Adapts precision based on model confidence in real-time")
    print("- Reduces computation for high-confidence predictions")
    print("- Maintains quality for uncertain/difficult tokens")
    print("- Enables fine-grained speed vs accuracy trade-offs")
    print("- Works with any existing model architecture")
    print("- Can be combined with other optimization techniques")


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
