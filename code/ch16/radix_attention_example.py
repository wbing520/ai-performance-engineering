#!/usr/bin/env python3
"""
radix_attention_example.py
Chapter 16: Radix Attention KV Cache Implementation Example

This example demonstrates SGLang's RadixAttention approach using a compressed trie 
(radix tree) for efficient KV cache prefix sharing in LLM inference.

Based on Chapter 16 content and SGLang's prefix caching design.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np


@dataclass
class KVCache:
    """Represents a KV cache tensor pair for a specific prefix."""
    keys: torch.Tensor      # [seq_len, num_heads, head_dim]
    values: torch.Tensor    # [seq_len, num_heads, head_dim]
    seq_len: int
    ref_count: int = 1
    
    def clone_ref(self):
        """Shallow clone with reference counting."""
        self.ref_count += 1
        return KVCache(
            keys=self.keys,
            values=self.values,
            seq_len=self.seq_len,
            ref_count=self.ref_count
        )
    
    def release(self):
        """Decrease reference count."""
        self.ref_count -= 1
        return self.ref_count <= 0


class RadixTreeNode:
    """Node in the radix tree for prefix caching."""
    
    def __init__(self, token_sequence: List[int] = None, cache: KVCache = None):
        self.token_sequence = token_sequence or []  # Edge label (sequence of tokens)
        self.cache = cache                          # KV cache for this prefix
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.last_accessed = 0                      # For LRU eviction
        self.is_leaf = True
    
    def __repr__(self):
        return f"RadixNode(tokens={self.token_sequence}, children={len(self.children)})"


class RadixTree:
    """
    Radix tree (compressed trie) for efficient prefix caching.
    Based on SGLang's RadixAttention design.
    """
    
    def __init__(self):
        self.root = RadixTreeNode()
        self.access_counter = 0
        self.max_cache_size = 1000  # Maximum number of cached prefixes
        self.current_cache_count = 0
    
    def longest_prefix(self, tokens: List[int]) -> Tuple[RadixTreeNode, int]:
        """
        Find the longest cached prefix for the given token sequence.
        Returns (node, prefix_length)
        """
        current = self.root
        matched_len = 0
        
        i = 0
        while i < len(tokens):
            # Check if any child edge starts with tokens[i]
            found_child = None
            for first_token, child in current.children.items():
                if first_token == tokens[i]:
                    found_child = child
                    break
            
            if found_child is None:
                break
            
            # Check how many tokens match this edge
            edge_tokens = found_child.token_sequence
            edge_match_len = 0
            
            for j, edge_token in enumerate(edge_tokens):
                if i + j >= len(tokens) or tokens[i + j] != edge_token:
                    break
                edge_match_len += 1
            
            if edge_match_len == 0:
                break
            
            # If we matched the entire edge, move to child and continue
            if edge_match_len == len(edge_tokens):
                current = found_child
                matched_len += edge_match_len
                i += edge_match_len
                # Update access time for LRU
                current.last_accessed = self.access_counter
                self.access_counter += 1
            else:
                # Partial match - we found the longest prefix
                break
        
        return current, matched_len
    
    def insert(self, tokens: List[int], cache: KVCache) -> RadixTreeNode:
        """
        Insert token sequence and associated KV cache into the radix tree.
        May split existing edges if needed.
        """
        if not tokens:
            return self.root
        
        current = self.root
        i = 0
        
        while i < len(tokens):
            # Check if any child edge starts with tokens[i]
            found_child = None
            first_token = tokens[i]
            
            for child_first_token, child in current.children.items():
                if child_first_token == first_token:
                    found_child = child
                    break
            
            if found_child is None:
                # No matching child - create new edge with remaining tokens
                remaining_tokens = tokens[i:]
                new_node = RadixTreeNode(remaining_tokens, cache)
                new_node.last_accessed = self.access_counter
                self.access_counter += 1
                
                current.children[first_token] = new_node
                current.is_leaf = False
                self._maybe_evict()
                return new_node
            
            # Found a child - check how much of the edge matches
            edge_tokens = found_child.token_sequence
            edge_match_len = 0
            
            for j, edge_token in enumerate(edge_tokens):
                if i + j >= len(tokens) or tokens[i + j] != edge_token:
                    break
                edge_match_len += 1
            
            if edge_match_len == len(edge_tokens):
                # Matched entire edge - continue to child
                current = found_child
                i += edge_match_len
                
                if i == len(tokens):
                    # Exact match - update cache
                    if current.cache:
                        current.cache.release()
                    current.cache = cache
                    current.last_accessed = self.access_counter
                    self.access_counter += 1
                    return current
            else:
                # Partial match - need to split the edge
                return self._split_edge(found_child, edge_match_len, tokens, i, cache)
        
        return current
    
    def _split_edge(self, node: RadixTreeNode, split_pos: int, 
                   new_tokens: List[int], new_token_pos: int, cache: KVCache) -> RadixTreeNode:
        """Split an edge at the given position."""
        # Split the existing edge
        common_prefix = node.token_sequence[:split_pos]
        remaining_old = node.token_sequence[split_pos:]
        remaining_new = new_tokens[new_token_pos + split_pos:]
        
        # Create intermediate node for common prefix
        intermediate = RadixTreeNode(common_prefix)
        intermediate.last_accessed = self.access_counter
        self.access_counter += 1
        
        # Update the existing node to represent the remaining part
        node.token_sequence = remaining_old
        if remaining_old:
            intermediate.children[remaining_old[0]] = node
            intermediate.is_leaf = False
        
        # Create new node for the new branch
        if remaining_new:
            new_node = RadixTreeNode(remaining_new, cache)
            new_node.last_accessed = self.access_counter
            self.access_counter += 1
            intermediate.children[remaining_new[0]] = new_node
            intermediate.is_leaf = False
            self._maybe_evict()
            return new_node
        else:
            # New tokens end at the split point
            intermediate.cache = cache
            self._maybe_evict()
            return intermediate
    
    def _maybe_evict(self):
        """Evict least recently used nodes if cache is full."""
        self.current_cache_count += 1
        
        if self.current_cache_count > self.max_cache_size:
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict the least recently used leaf nodes."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        
        if not leaves:
            return
        
        # Sort by access time (oldest first)
        leaves.sort(key=lambda x: x.last_accessed)
        
        # Evict the oldest 10% of leaves
        evict_count = max(1, len(leaves) // 10)
        for i in range(evict_count):
            leaf = leaves[i]
            if leaf.cache:
                leaf.cache.release()
                leaf.cache = None
                self.current_cache_count -= 1
    
    def _collect_leaves(self, node: RadixTreeNode, leaves: List[RadixTreeNode]):
        """Collect all leaf nodes for LRU eviction."""
        if node.is_leaf and node.cache:
            leaves.append(node)
        
        for child in node.children.values():
            self._collect_leaves(child, leaves)


class ModelState:
    """Simplified model state for generation."""
    
    def __init__(self, kv_cache: KVCache = None, hidden_dim: int = 512, num_heads: int = 8):
        self.kv_cache = kv_cache
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.finished = False
    
    @classmethod
    def from_cache(cls, cache: KVCache) -> 'ModelState':
        """Create model state from existing KV cache (reference counting)."""
        return cls(kv_cache=cache.clone_ref() if cache else None)
    
    def is_finished(self) -> bool:
        return self.finished


class SimpleTransformerModel:
    """Simplified transformer model for demonstration."""
    
    def __init__(self, vocab_size: int = 50000, hidden_dim: int = 512, num_heads: int = 8):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Initialize dummy parameters
        self.embedding = torch.randn(vocab_size, hidden_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)
        self.w_q = torch.randn(hidden_dim, hidden_dim)
        self.w_k = torch.randn(hidden_dim, hidden_dim)
        self.w_v = torch.randn(hidden_dim, hidden_dim)
        self.w_o = torch.randn(hidden_dim, hidden_dim)
        self.lm_head = torch.randn(hidden_dim, vocab_size)
    
    def forward(self, token: int, state: ModelState) -> ModelState:
        """Forward pass for a single token, updating KV cache."""
        # Get token embedding
        x = self.embedding[token].unsqueeze(0)  # [1, hidden_dim]
        x = self.ln(x)
        
        # Compute Q, K, V
        q = torch.matmul(x, self.w_q).view(1, self.num_heads, self.head_dim)
        k = torch.matmul(x, self.w_k).view(1, self.num_heads, self.head_dim)
        v = torch.matmul(x, self.w_v).view(1, self.num_heads, self.head_dim)
        
        # Update KV cache
        if state.kv_cache is None:
            # First token - initialize cache
            new_cache = KVCache(
                keys=k,
                values=v,
                seq_len=1
            )
        else:
            # Append to existing cache
            new_keys = torch.cat([state.kv_cache.keys, k], dim=0)
            new_values = torch.cat([state.kv_cache.values, v], dim=0)
            new_cache = KVCache(
                keys=new_keys,
                values=new_values,
                seq_len=state.kv_cache.seq_len + 1
            )
            
            # Release old cache reference
            if state.kv_cache:
                state.kv_cache.release()
        
        # Create new state
        new_state = ModelState(kv_cache=new_cache, hidden_dim=self.hidden_dim, num_heads=self.num_heads)
        return new_state
    
    def generate_next(self, state: ModelState) -> Tuple[int, ModelState]:
        """Generate next token autoregressively."""
        if state.kv_cache is None:
            raise ValueError("Cannot generate without KV cache")
        
        seq_len = state.kv_cache.seq_len
        
        # Use last position for query
        last_k = state.kv_cache.keys[-1:]  # [1, num_heads, head_dim]
        last_v = state.kv_cache.values[-1:]  # [1, num_heads, head_dim]
        
        # Simplified attention (normally would use all K,V)
        # For demo, just project last value
        attn_out = last_v.view(1, self.hidden_dim)
        output = torch.matmul(attn_out, self.w_o)
        
        # Get logits and sample
        logits = torch.matmul(output, self.lm_head.T)
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
        
        # Check for end of generation (simplified)
        state.finished = (next_token == 0) or (seq_len >= 50)  # EOS or max length
        
        return next_token, state


def generate_with_radix(prompt_tokens: List[int], model: SimpleTransformerModel, 
                       radix: RadixTree) -> List[int]:
    """
    Generate tokens using RadixAttention prefix caching.
    
    This is the main function from Chapter 16 that demonstrates
    how SGLang's RadixAttention works.
    """
    print(f"Generating with prompt: {prompt_tokens}")
    
    # 1) Find longest cached prefix
    node, prefix_len = radix.longest_prefix(prompt_tokens)
    print(f"Found cached prefix of length: {prefix_len}")
    
    # Shallow-clone the KV cache for that prefix
    model_state = ModelState.from_cache(node.cache) if node.cache else ModelState()
    
    # 2) Process remaining prompt suffix
    for i, token in enumerate(prompt_tokens[prefix_len:]):
        model_state = model.forward(token, state=model_state)
        
        # 3) As we go, insert or split edges in the radix tree
        matched = prompt_tokens[:prefix_len + i + 1]
        # Insert returns the node for this full prefix
        node = radix.insert(matched, cache=model_state.kv_cache)
        
        print(f"Processed token {token}, cached prefix length: {len(matched)}")
    
    # 4) Now generate new tokens autoregressively
    output_tokens = []
    while not model_state.is_finished():
        token, model_state = model.generate_next(model_state)
        output_tokens.append(token)
        
        # Cache each generated prefix as well
        matched = prompt_tokens + output_tokens
        node = radix.insert(matched, cache=model_state.kv_cache)
        
        print(f"Generated token: {token}")
    
    print(f"Generated {len(output_tokens)} tokens")
    return output_tokens


def simulate_multi_turn_conversation():
    """
    Simulate the multi-turn conversation example from Chapter 16 Figure 16-9.
    Demonstrates how the radix tree evolves with multiple chat sessions.
    """
    print("\n=== Multi-Turn Conversation Simulation ===")
    
    # Initialize model and radix tree
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    radix = RadixTree()
    radix.max_cache_size = 10  # Small for demo
    
    # System prompt (shared across all conversations)
    system_prompt = [1, 2, 3, 4, 5]  # "You are a helpful assistant"
    
    # Chat Session 1
    print("\n--- Chat Session 1 ---")
    
    # Turn 1: User says "Hello"
    prompt1 = system_prompt + [10, 11]  # + "Hello"
    response1 = generate_with_radix(prompt1, model, radix)
    
    # Turn 2: Continue conversation
    prompt2 = prompt1 + response1 + [12, 13]  # + "How are you?"
    response2 = generate_with_radix(prompt2, model, radix)
    
    # Chat Session 2 (shares system prompt)
    print("\n--- Chat Session 2 ---")
    
    # Turn 1: Different user message
    prompt3 = system_prompt + [20, 21]  # + "Hi there"
    response3 = generate_with_radix(prompt3, model, radix)
    
    # Turn 2: Continue session 2
    prompt4 = prompt3 + response3 + [22, 23]  # + "What's new?"
    response4 = generate_with_radix(prompt4, model, radix)
    
    # Continue Chat Session 1 (should reuse cached data)
    print("\n--- Back to Chat Session 1 ---")
    prompt5 = prompt2 + response2 + [14, 15]  # + "Thanks!"
    response5 = generate_with_radix(prompt5, model, radix)
    
    print(f"\nRadix tree now has {radix.current_cache_count} cached prefixes")
    
    # New conversation with different system prompt
    print("\n--- New Conversation (Different System) ---")
    new_system = [100, 101, 102]  # Different system prompt
    prompt6 = new_system + [30, 31]  # + "Hello world"
    response6 = generate_with_radix(prompt6, model, radix)
    
    print(f"Final radix tree has {radix.current_cache_count} cached prefixes")


def benchmark_prefix_caching():
    """
    Benchmark to show the benefits of prefix caching.
    Compares generation with and without radix tree caching.
    """
    print("\n=== Prefix Caching Benchmark ===")
    
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    
    # Common system prompt (long)
    system_prompt = list(range(1, 21))  # 20 tokens
    
    # Test prompts that share the system prompt
    test_prompts = [
        system_prompt + [100, 101, 102],
        system_prompt + [200, 201, 202, 203],
        system_prompt + [300, 301],
        system_prompt + [400, 401, 402, 403, 404],
    ]
    
    import time
    
    # Test without caching (naive approach)
    print("\n--- Without Caching ---")
    start_time = time.time()
    naive_responses = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"Processing prompt {i+1}/{len(test_prompts)}")
        # Simulate processing entire prompt from scratch
        state = ModelState()
        for token in prompt:
            state = model.forward(token, state)
        
        # Generate response
        response = []
        for _ in range(5):  # Generate 5 tokens
            if state.is_finished():
                break
            token, state = model.generate_next(state)
            response.append(token)
        
        naive_responses.append(response)
    
    naive_time = time.time() - start_time
    print(f"Naive approach took: {naive_time:.3f} seconds")
    
    # Test with RadixAttention caching
    print("\n--- With RadixAttention Caching ---")
    radix = RadixTree()
    start_time = time.time()
    cached_responses = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"Processing prompt {i+1}/{len(test_prompts)} with caching")
        
        # Use radix caching
        node, prefix_len = radix.longest_prefix(prompt)
        print(f"  Reused {prefix_len}/{len(prompt)} tokens from cache")
        
        # Start from cached state
        state = ModelState.from_cache(node.cache) if node.cache else ModelState()
        
        # Only process uncached suffix
        for token in prompt[prefix_len:]:
            state = model.forward(token, state)
        
        # Cache the full prompt
        radix.insert(prompt, state.kv_cache)
        
        # Generate response
        response = []
        for _ in range(5):  # Generate 5 tokens
            if state.is_finished():
                break
            token, state = model.generate_next(state)
            response.append(token)
            
            # Cache intermediate states
            full_sequence = prompt + response
            radix.insert(full_sequence, state.kv_cache)
        
        cached_responses.append(response)
    
    cached_time = time.time() - start_time
    print(f"RadixAttention approach took: {cached_time:.3f} seconds")
    
    speedup = naive_time / cached_time if cached_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify responses are similar (they won't be identical due to randomness)
    print(f"Radix tree cached {radix.current_cache_count} prefixes")


def main():
    """Main demonstration of RadixAttention concepts."""
    print("Chapter 16: RadixAttention Prefix Caching Example")
    print("=" * 50)
    
    # Basic example
    print("\n=== Basic RadixAttention Example ===")
    model = SimpleTransformerModel(vocab_size=1000, hidden_dim=256, num_heads=4)
    radix = RadixTree()
    
    # Simple prompt
    prompt = [1, 2, 3, 4, 5, 6, 7]
    response = generate_with_radix(prompt, model, radix)
    print(f"Response: {response}")
    
    # Second prompt with shared prefix
    prompt2 = [1, 2, 3, 4, 5, 8, 9]  # Shares first 5 tokens
    response2 = generate_with_radix(prompt2, model, radix)
    print(f"Response 2: {response2}")
    
    # Run more comprehensive demos
    simulate_multi_turn_conversation()
    benchmark_prefix_caching()
    
    print("\n" + "=" * 50)
    print("RadixAttention Demo Complete!")
    print("Key benefits demonstrated:")
    print("- Efficient prefix sharing across requests")
    print("- Automatic KV cache management")
    print("- LRU eviction for memory management")
    print("- Significant speedup for repeated prefixes")


if __name__ == "__main__":
    main()
