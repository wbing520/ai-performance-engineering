#!/usr/bin/env python3
"""
Chapter 13: PyTorch Compiler Comparison
Demonstrates different torch.compile modes and their trade-offs
"""

import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def create_simple_model():
    """Create a simple model for testing compiler modes"""
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024)
    ).cuda()
    return model

def benchmark_compiler_modes():
    """Benchmark different torch.compile modes"""
    print("=== PyTorch Compiler Mode Comparison ===")
    
    # Create model and data
    model = create_simple_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(32, 1024, device='cuda')
    
    # Make runs deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Test different compiler modes
    modes = [
        ("default", "default"),
        ("reduce-overhead", "reduce-overhead"),
        ("max-autotune", "max-autotune"),
        ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs")
    ]
    
    results = {}
    
    for mode_name, mode in modes:
        print(f"\n--- Testing {mode_name} mode ---")
        
        # Compile model with this mode
        compiled_model = torch.compile(model, mode=mode)
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            optimizer.zero_grad()
            output = compiled_model(x)
            loss = output.mean()
            loss.backward()
            optimizer.step()
        
        # Benchmark
        print("Benchmarking...")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for i in range(10):
            optimizer.zero_grad()
            output = compiled_model(x)
            loss = output.mean()
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        results[mode_name] = avg_time
        print(f"Average iteration time: {avg_time:.4f} s")
    
    # Print comparison
    print("\n=== Results Summary ===")
    baseline = results["default"]
    for mode_name, time_taken in results.items():
        speedup = baseline / time_taken
        print(f"{mode_name:25s}: {time_taken:.4f}s ({speedup:.2f}x vs default)")
    
    return results

def demonstrate_compiler_debugging():
    """Demonstrate debugging compiler issues"""
    print("\n=== Compiler Debugging ===")
    
    # Create a model with potential graph breaks
    class DynamicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
            self.counter = 0
        
        def forward(self, x):
            # This will cause graph breaks due to dynamic control flow
            if self.counter % 2 == 0:
                x = self.linear(x)
            else:
                x = x * 2
            self.counter += 1
            return x
    
    model = DynamicModel().cuda()
    
    # Try to compile and explain graph breaks
    print("Attempting to compile model with dynamic control flow...")
    
    try:
        compiled_model = torch.compile(model)
        
        # Use torch._dynamo.explain to see what happened
        import torch._dynamo
        explanation = torch._dynamo.explain(model, torch.randn(1, 512, device='cuda'))
        print("\nGraph break explanation:")
        print(explanation)
        
    except Exception as e:
        print(f"Compilation failed: {e}")

def demonstrate_memory_profiling():
    """Demonstrate memory profiling with compiler"""
    print("\n=== Memory Profiling with Compiler ===")
    
    model = create_simple_model()
    x = torch.randn(64, 1024, device='cuda')
    
    # Profile memory usage with and without compilation
    print("Memory usage comparison:")
    
    # Without compilation
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    eager_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Eager mode peak memory: {eager_memory:.2f} GB")
    
    # With compilation
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    compiled_model = torch.compile(model, mode="max-autotune")
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-4)
    
    # Warm up
    for _ in range(3):
        optimizer.zero_grad()
        output = compiled_model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    # Benchmark
    for _ in range(5):
        optimizer.zero_grad()
        output = compiled_model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    compiled_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Compiled mode peak memory: {compiled_memory:.2f} GB")
    print(f"Memory difference: {compiled_memory - eager_memory:.2f} GB")

def demonstrate_compilation_modes():
    """Demonstrate different compilation modes and their characteristics"""
    print("\n=== Compilation Modes Deep Dive ===")
    
    modes_info = {
        "default": {
            "description": "Balanced optimizations",
            "compile_time": "Low-Medium",
            "extra_memory": "No",
            "features": "General fusion, basic autotuning"
        },
        "reduce-overhead": {
            "description": "Reduces per-iteration overhead",
            "compile_time": "Medium",
            "extra_memory": "Yes (workspace caching)",
            "features": "Uses CUDA Graphs (if possible) to eliminate launch overhead"
        },
        "max-autotune": {
            "description": "Maximizes runtime performance",
            "compile_time": "High (slow compile)",
            "extra_memory": "Maybe (if graphs used)",
            "features": "Aggressive Triton autotuning; enables CUDA Graphs on GPU"
        },
        "max-autotune-no-cudagraphs": {
            "description": "Same as max-autotune but without CUDA graph capture",
            "compile_time": "High",
            "extra_memory": "No",
            "features": "Same as above but disables graphs for flexibility"
        }
    }
    
    print("Compilation Mode Characteristics:")
    print("-" * 80)
    for mode, info in modes_info.items():
        print(f"{mode:25s}: {info['description']}")
        print(f"{'':25s}  Compile time: {info['compile_time']}")
        print(f"{'':25s}  Extra memory: {info['extra_memory']}")
        print(f"{'':25s}  Features: {info['features']}")
        print()

def main():
    """Main function demonstrating compiler comparison"""
    print("Chapter 13: PyTorch Compiler Comparison")
    print("=" * 60)
    
    # Run all demonstrations
    benchmark_compiler_modes()
    demonstrate_compiler_debugging()
    demonstrate_memory_profiling()
    demonstrate_compilation_modes()
    
    print("\n" + "=" * 60)
    print("Compiler comparison completed!")
    print("\nKey takeaways:")
    print("- Use 'default' mode for general use cases")
    print("- Use 'reduce-overhead' for small models or inference")
    print("- Use 'max-autotune' for long-running training jobs")
    print("- Use 'max-autotune-no-cudagraphs' for dynamic shapes")
    print("- Always profile to find the best mode for your workload")

if __name__ == "__main__":
    main()
