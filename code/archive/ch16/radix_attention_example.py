#!/usr/bin/env python3
"""
RadixAttention KV-cache example from Chapter 16.
This demonstrates prefix caching using a radix tree structure.
"""

import time
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelState:
    """Simplified model state for KV cache"""
    kv_cache: torch.Tensor
    is_finished: bool = False
    
    @classmethod
    def from_cache(cls, cache: torch.Tensor):
        """Create model state from cached KV cache"""
        return cls(kv_cache=cache.clone())

class RadixTree:
    """Simplified RadixAttention KV-cache implementation"""
    
    def __init__(self):
        self.nodes = {}  # token_sequence -> node
        self.cache = {}  # node -> kv_cache
        self.access_times = {}  # node -> last access time
        
    def longest_prefix(self, prompt_tokens: List[int]) -> Tuple[str, int]:
        """Find longest cached prefix"""
        current_prefix = ""
        longest_match = ("", 0)
        
        for i, token in enumerate(prompt_tokens):
            current_prefix += f"{token}_"
            if current_prefix in self.nodes:
                longest_match = (current_prefix, i + 1)
                self.access_times[current_prefix] = time.time()
            else:
                break
                
        return longest_match
    
    def insert(self, token_sequence: List[int], cache: torch.Tensor) -> str:
        """Insert or update cache for token sequence"""
        sequence_key = "_".join(map(str, token_sequence))
        self.nodes[sequence_key] = sequence_key
        self.cache[sequence_key] = cache
        self.access_times[sequence_key] = time.time()
        return sequence_key
    
    def evict_lru(self):
        """Evict least recently used cache entry"""
        if not self.access_times:
            return
            
        lru_node = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.nodes[lru_node]
        del self.cache[lru_node]
        del self.access_times[lru_node]
        print(f"Evicted LRU node: {lru_node}")

class MockModel:
    """Mock model for demonstration"""
    
    def __init__(self, hidden_size=768, num_layers=12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, token: int, state: Optional[ModelState] = None) -> ModelState:
        """Forward pass with optional cached state"""
        if state is None:
            # Initialize new state
            kv_cache = torch.randn(self.num_layers, 2, 1, self.hidden_size)
        else:
            # Use cached state and extend
            kv_cache = torch.cat([state.kv_cache, 
                                 torch.randn(self.num_layers, 2, 1, self.hidden_size)], dim=2)
        
        return ModelState(kv_cache=kv_cache)
    
    def generate_next(self, state: ModelState) -> Tuple[int, ModelState]:
        """Generate next token"""
        # Simulate token generation
        next_token = random.randint(100, 1000)
        
        # Extend KV cache
        new_kv = torch.randn(self.num_layers, 2, 1, self.hidden_size)
        extended_kv = torch.cat([state.kv_cache, new_kv], dim=2)
        
        return next_token, ModelState(kv_cache=extended_kv)

def generate_with_radix(prompt_tokens: List[int], model: MockModel, radix: RadixTree) -> List[int]:
    """Generate tokens using RadixAttention KV-cache"""
    
    print(f"Processing prompt: {prompt_tokens}")
    
    # 1) Find longest cached prefix
    node, prefix_len = radix.longest_prefix(prompt_tokens)
    
    if prefix_len > 0:
        print(f"Cache hit! Found prefix of length {prefix_len}")
        # shallow-clone the KV cache for that prefix
        model_state = ModelState.from_cache(radix.cache[node])
    else:
        print("Cache miss - starting from scratch")
        model_state = None
    
    # 2) Process remaining prompt suffix
    for i, token in enumerate(prompt_tokens[prefix_len:], prefix_len):
        model_state = model.forward(token, state=model_state)
        
        # 3) As we go, insert or split edges in the radix tree
        matched = prompt_tokens[:i + 1]
        # insert returns the node for this full prefix
        node = radix.insert(matched, cache=model_state.kv_cache)
        prefix_len += 1
    
    # 4) Now generate new tokens autoregressively
    output_tokens = []
    max_tokens = 10  # Limit for demo
    
    while not model_state.is_finished and len(output_tokens) < max_tokens:
        token, model_state = model.generate_next(model_state)
        output_tokens.append(token)
        
        # cache each generated prefix as well
        matched = prompt_tokens + output_tokens
        node = radix.insert(matched, cache=model_state.kv_cache)
        
        # Simulate completion
        if len(output_tokens) >= max_tokens:
            model_state.is_finished = True
    
    return output_tokens

def main():
    """Main demonstration function"""
    print("RadixAttention KV-cache Example")
    print("================================")
    
    # Initialize components
    radix = RadixTree()
    model = MockModel()
    
    # Test prompts
    test_prompts = [
        [1, 2, 3, 4, 5],  # First prompt
        [1, 2, 3, 4, 5, 6, 7],  # Extends first prompt
        [1, 2, 3, 8, 9],  # Shares prefix with first
        [10, 11, 12, 13],  # Completely different
        [1, 2, 3, 4, 5, 8, 9],  # Extends shared prefix
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Processing Prompt {i+1} ---")
        
        # Simulate memory pressure
        if i == 3:  # Force eviction
            print("Simulating memory pressure...")
            radix.evict_lru()
        
        start_time = time.time()
        output = generate_with_radix(prompt, model, radix)
        end_time = time.time()
        
        print(f"Generated {len(output)} tokens in {end_time - start_time:.3f}s")
        print(f"Output tokens: {output[:5]}...")  # Show first 5 tokens
        
        # Show cache stats
        print(f"Cache size: {len(radix.nodes)} nodes")
        print(f"Memory usage: {sum(cache.numel() * cache.element_size() for cache in radix.cache.values()) / 1e6:.2f} MB")

if __name__ == "__main__":
    main()
