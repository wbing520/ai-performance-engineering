# torch_compiler_examples.py
import torch
import torch._dynamo as dynamo
import time
from torch.nn import functional as F

def toy_example_with_graph_breaks(a, b):
    """
    Example showing code patterns that cause graph breaks.
    """
    x = a / (torch.abs(a) + 1)
    print("woo")  # This causes a graph break - side effect
    
    if b.sum() < 0:  # Dynamic control flow - causes graph break
        b = -b
    
    return x * b

def optimized_toy_example(a, b):
    """
    Optimized version that avoids unnecessary graph breaks.
    """
    x = a / (torch.abs(a) + 1)
    
    # Avoid side effects during compilation
    if not torch._dynamo.is_compiling():
        print("do not print during tracing/compiling")
    
    # Use torch.cond for data-dependent branches
    def true_fn(b):
        return -b
    
    def false_fn(b):
        return b
        
    b = torch.cond(b.sum() < 0, true_fn, false_fn, (b,))
    
    return x * b

def model_with_torch_cond(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Example using torch.cond for data-dependent control flow.
    """
    # Compute x as before
    x = a / (torch.abs(a) + 1)
    
    # Use torch.cond to capture both branches
    def true_branch(b):
        return -b
        
    def false_branch(b):
        return b
    
    predicate = b.sum() < 0
    b = torch.cond(predicate, true_branch, false_branch, (b,))
    
    return x * b

def model_with_torch_where(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Alternative using torch.where for element-wise conditional operations.
    """
    x = a / (torch.abs(a) + 1)
    
    # Element-wise conditional: if each element of b < 0, negate it
    b = torch.where(b < 0, -b, b)
    
    return x * b

class SimpleModel(torch.nn.Module):
    """
    Simple model for demonstrating torch.compile optimizations.
    """
    def __init__(self, input_size=1024, hidden_size=512, output_size=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def benchmark_compilation_modes():
    """
    Benchmark different torch.compile modes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and sample data
    model = SimpleModel().to(device)
    x = torch.randn(32, 1024, device=device)
    
    # Test different compilation modes
    modes = ['default', 'reduce-overhead', 'max-autotune']
    
    results = {}
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        
        # Compile model
        if mode == 'default':
            compiled_model = torch.compile(model)
        else:
            compiled_model = torch.compile(model, mode=mode)
        
        # Warmup
        for _ in range(10):
            _ = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            output = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        results[mode] = avg_time
        print(f"Average time per forward pass: {avg_time:.4f}s")
    
    # Test uncompiled baseline
    print(f"\nTesting uncompiled baseline")
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(100):
        output = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    baseline_time = (end_time - start_time) / 100
    
    results['uncompiled'] = baseline_time
    print(f"Average time per forward pass: {baseline_time:.4f}s")
    
    # Print speedup comparison
    print(f"\nSpeedup comparison:")
    for mode, time_taken in results.items():
        if mode != 'uncompiled':
            speedup = baseline_time / time_taken
            print(f"{mode}: {speedup:.2f}x speedup")
    
    return results

def analyze_graph_breaks():
    """
    Analyze and debug graph breaks using torch._dynamo.explain().
    """
    print("Analyzing graph breaks in toy_example_with_graph_breaks:")
    
    # Generate sample inputs
    a = torch.randn(10)
    b = torch.randn(10)
    
    # Analyze the function with graph breaks
    explanation = dynamo.explain(toy_example_with_graph_breaks)(a, b)
    print(explanation)
    
    print("\n" + "="*50)
    print("Analyzing optimized version:")
    
    # Analyze the optimized version
    explanation_opt = dynamo.explain(optimized_toy_example)(a, b)
    print(explanation_opt)

def dynamic_shape_example():
    """
    Example showing dynamic shape handling.
    """
    def dynamic_model(x):
        # This creates dynamic shapes based on input
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Dynamically sized operations
        if seq_len > 100:
            # Process in chunks for long sequences
            chunks = torch.chunk(x, chunks=seq_len//50, dim=1)
            processed = [F.relu(chunk) for chunk in chunks]
            return torch.cat(processed, dim=1)
        else:
            # Direct processing for short sequences  
            return F.relu(x)
    
    # Test with different input sizes
    sizes = [(32, 50, 128), (32, 150, 128), (16, 75, 128)]
    
    compiled_model = torch.compile(dynamic_model, dynamic=True)
    
    for batch_size, seq_len, hidden_size in sizes:
        x = torch.randn(batch_size, seq_len, hidden_size)
        print(f"Processing shape {x.shape}")
        
        # First run may trigger recompilation
        output = compiled_model(x)
        print(f"Output shape: {output.shape}")

def custom_operator_example():
    """
    Example of registering a custom operator that works with torch.compile.
    """
    @torch.compile
    def model_with_custom_op(x):
        # Use built-in operations that compile well
        y = x * 2
        z = torch.sin(y)
        return z.sum()
    
    x = torch.randn(1000, 1000, requires_grad=True)
    
    # Forward pass
    result = model_with_custom_op(x)
    
    # Backward pass also gets compiled
    result.backward()
    
    print(f"Custom operator result: {result.item()}")
    print(f"Gradient shape: {x.grad.shape}")

def memory_efficient_compilation():
    """
    Example showing memory-efficient patterns with torch.compile.
    """
    class MemoryEfficientModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(1024, 1024) for _ in range(10)
            ])
        
        def forward(self, x):
            # Use gradient checkpointing for memory efficiency
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    lambda x, layer=layer: F.relu(layer(x)), x
                )
            return x
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemoryEfficientModel().to(device)
    
    # Compile with memory-efficient mode
    compiled_model = torch.compile(model, mode='reduce-overhead')
    
    x = torch.randn(32, 1024, device=device, requires_grad=True)
    
    # Measure memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    output = compiled_model(x)
    loss = output.sum()
    loss.backward()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory usage: {peak_memory:.2f} GB")

def main():
    """
    Run all torch.compile examples.
    """
    print("PyTorch Compiler Examples")
    print("=" * 40)
    
    print("\n1. Analyzing Graph Breaks:")
    analyze_graph_breaks()
    
    print("\n2. Benchmarking Compilation Modes:")
    benchmark_compilation_modes()
    
    print("\n3. Dynamic Shape Handling:")
    dynamic_shape_example()
    
    print("\n4. Custom Operator Example:")
    custom_operator_example()
    
    print("\n5. Memory-Efficient Compilation:")
    memory_efficient_compilation()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()
