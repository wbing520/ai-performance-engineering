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
Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends
Comprehensive examples demonstrating torch.compile, graph breaks, debugging, and optimization
"""

import torch
import torch.nn as nn
import torch._dynamo as dynamo
import os

def demonstrate_basic_compilation():
    """Demonstrate basic torch.compile usage"""
    print("=== Basic PyTorch Compiler Usage ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).cuda()
    
    # Basic compilation
    compiled_model = torch.compile(model, mode="max-autotune")
    
    # Test input
    x = torch.randn(32, 1024, device='cuda')
    
    # Warm up
    print("Warming up compiled model...")
    for _ in range(3):
        output = compiled_model(x)
    
    print("Basic compilation completed successfully!")

def demonstrate_compilation_modes():
    """Demonstrate different torch.compile modes"""
    print("\n=== Compilation Modes ===")
    
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512)
    ).cuda()
    
    x = torch.randn(16, 512, device='cuda')
    
    modes = [
        ("default", "default"),
        ("reduce-overhead", "reduce-overhead"),
        ("max-autotune", "max-autotune"),
        ("max-autotune-no-cudagraphs", "max-autotune-no-cudagraphs")
    ]
    
    for mode_name, mode in modes:
        print(f"\nTesting {mode_name} mode...")
        try:
            compiled_model = torch.compile(model, mode=mode)
            
            # Warm up
            for _ in range(2):
                output = compiled_model(x)
            
            print(f"✓ {mode_name} mode works")
        except Exception as e:
            print(f"✗ {mode_name} mode failed: {e}")

def demonstrate_graph_breaks():
    """Demonstrate graph breaks and how to fix them"""
    print("\n=== Graph Breaks Analysis ===")
    
    def problematic_model(a, b):
        x = a / (torch.abs(a) + 1)
        print("woo")  # This will cause a graph break
        if b.sum() < 0:  # This will also cause a graph break
            b = -b
        return x * b
    
    def fixed_model(a, b):
        x = a / (torch.abs(a) + 1)
        # Avoid print during compilation
        if not torch._dynamo.is_compiling():
            print("woo")
        # Use torch.where instead of Python if
        b = torch.where(b.sum() < 0, -b, b)
        return x * b
    
    # Test problematic model
    print("Testing problematic model...")
    a = torch.randn(10, device='cuda')
    b = torch.randn(10, device='cuda')
    
    try:
        explanation = dynamo.explain(problematic_model)(a, b)
        print(f"Problematic model graph breaks: {explanation.graph_break_count}")
    except Exception as e:
        print(f"Problematic model failed: {e}")
    
    # Test fixed model
    print("\nTesting fixed model...")
    try:
        explanation = dynamo.explain(fixed_model)(a, b)
        print(f"Fixed model graph breaks: {explanation.graph_break_count}")
    except Exception as e:
        print(f"Fixed model failed: {e}")

def demonstrate_dynamic_shapes():
    """Demonstrate dynamic shape handling"""
    print("\n=== Dynamic Shapes ===")
    
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).cuda()
    
    # Compile with dynamic shapes
    compiled_model = torch.compile(model, dynamic=True)
    
    # Test with different input sizes
    sizes = [16, 32, 64, 128]
    
    for size in sizes:
        x = torch.randn(size, 512, device='cuda')
        try:
            output = compiled_model(x)
            print(f"✓ Dynamic shape {size}x512 works")
        except Exception as e:
            print(f"✗ Dynamic shape {size}x512 failed: {e}")

def demonstrate_mark_dynamic():
    """Demonstrate marking specific dimensions as dynamic"""
    print("\n=== Mark Dynamic Dimensions ===")
    
    def model_with_variable_sequence(x):
        # This model has variable sequence length
        return x.mean(dim=1)
    
    # Mark the sequence dimension as dynamic
    x = torch.randn(4, 100, 512, device='cuda')
    torch._dynamo.mark_dynamic(x, 1)  # Mark sequence dimension as dynamic
    
    compiled_model = torch.compile(model_with_variable_sequence)
    
    # Test with different sequence lengths
    sequence_lengths = [50, 100, 150, 200]
    
    for seq_len in sequence_lengths:
        x = torch.randn(4, seq_len, 512, device='cuda')
        torch._dynamo.mark_dynamic(x, 1)
        try:
            output = compiled_model(x)
            print(f"✓ Sequence length {seq_len} works")
        except Exception as e:
            print(f"✗ Sequence length {seq_len} failed: {e}")

def demonstrate_allow_in_graph():
    """Demonstrate allow_in_graph for safe functions"""
    print("\n=== Allow in Graph ===")
    
    @torch._dynamo.allow_in_graph
    def safe_custom_function(x):
        # This function is safe to include in the graph
        return x * 2 + 1
    
    def model_with_custom_function(x):
        x = nn.Linear(512, 512).cuda()(x)
        x = safe_custom_function(x)  # This will be included in the graph
        return x
    
    x = torch.randn(16, 512, device='cuda')
    
    try:
        compiled_model = torch.compile(model_with_custom_function)
        output = compiled_model(x)
        print("✓ Custom function included in graph successfully")
    except Exception as e:
        print(f"✗ Custom function failed: {e}")

def demonstrate_compiler_stances():
    """Demonstrate different compiler stances"""
    print("\n=== Compiler Stances ===")
    
    stances = [
        ("default", "default"),
        ("fail_on_recompile", "fail_on_recompile"),
        ("eager_on_recompile", "eager_on_recompile"),
        ("force_eager", "force_eager")
    ]
    
    model = nn.Linear(512, 512).cuda()
    x = torch.randn(16, 512, device='cuda')
    
    for stance_name, stance in stances:
        print(f"\nTesting {stance_name} stance...")
        try:
            torch.compiler.set_stance(stance)
            compiled_model = torch.compile(model)
            output = compiled_model(x)
            print(f"✓ {stance_name} stance works")
        except Exception as e:
            print(f"✗ {stance_name} stance failed: {e}")

def demonstrate_debugging_tools():
    """Demonstrate debugging tools for torch.compile"""
    print("\n=== Debugging Tools ===")
    
    # Set up debugging environment variables
    os.environ['TORCH_LOGS'] = 'graph_breaks,dynamo'
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
    ).cuda()
    
    x = torch.randn(8, 256, device='cuda')
    
    print("Compiling with debug logging...")
    try:
        compiled_model = torch.compile(model, mode="max-autotune")
        output = compiled_model(x)
        print("✓ Compilation with debugging completed")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")

def demonstrate_fullgraph_mode():
    """Demonstrate fullgraph=True for strict compilation"""
    print("\n=== Full Graph Mode ===")
    
    def clean_model(x):
        # This model should compile without breaks
        x = x * 2
        x = x + 1
        return x
    
    def problematic_model(x):
        # This model will cause breaks
        x = x * 2
        print("This will cause a break")
        x = x + 1
        return x
    
    x = torch.randn(10, device='cuda')
    
    # Test clean model with fullgraph=True
    print("Testing clean model with fullgraph=True...")
    try:
        compiled_model = torch.compile(clean_model, fullgraph=True)
        output = compiled_model(x)
        print("✓ Clean model compiled successfully with fullgraph=True")
    except Exception as e:
        print(f"✗ Clean model failed: {e}")
    
    # Test problematic model with fullgraph=True
    print("\nTesting problematic model with fullgraph=True...")
    try:
        compiled_model = torch.compile(problematic_model, fullgraph=True)
        output = compiled_model(x)
        print("✓ Problematic model compiled successfully")
    except Exception as e:
        print(f"✗ Problematic model failed (expected): {e}")

def demonstrate_performance_hints():
    """Demonstrate performance hints logging"""
    print("\n=== Performance Hints ===")
    
    # Enable performance hints
    os.environ['TORCH_LOGS'] = 'perf_hints'
    
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512)
    ).cuda()
    
    x = torch.randn(32, 512, device='cuda')
    
    print("Compiling with performance hints...")
    try:
        compiled_model = torch.compile(model, mode="max-autotune")
        output = compiled_model(x)
        print("✓ Compilation with performance hints completed")
    except Exception as e:
        print(f"✗ Compilation failed: {e}")

def demonstrate_xla_backend():
    """Demonstrate XLA backend usage"""
    print("\n=== XLA Backend ===")
    
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
    ).cuda()
    
    x = torch.randn(16, 256, device='cuda')
    
    print("Testing XLA backend...")
    try:
        # Note: XLA backend may not be available on all systems
        compiled_model = torch.compile(model, backend="openxla")
        output = compiled_model(x)
        print("✓ XLA backend compilation successful")
    except Exception as e:
        print(f"✗ XLA backend failed (may not be available): {e}")

def demonstrate_eager_backend():
    """Demonstrate eager backend for comparison"""
    print("\n=== Eager Backend ===")
    
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
    ).cuda()
    
    x = torch.randn(16, 256, device='cuda')
    
    print("Testing eager backend...")
    try:
        compiled_model = torch.compile(model, backend="eager")
        output = compiled_model(x)
        print("✓ Eager backend compilation successful")
    except Exception as e:
        print(f"✗ Eager backend failed: {e}")

def demonstrate_disable_compiler():
    """Demonstrate disabling the compiler"""
    print("\n=== Disable Compiler ===")
    
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
    ).cuda()
    
    x = torch.randn(16, 256, device='cuda')
    
    # Disable compiler temporarily
    with torch._dynamo.disable():
        print("Compiler disabled - running in eager mode")
        output = model(x)
        print("✓ Model ran successfully in eager mode")

def main():
    """Main function demonstrating all PyTorch compiler techniques"""
    print("Chapter 14: PyTorch Compiler Deep Dive")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_basic_compilation()
    demonstrate_compilation_modes()
    demonstrate_graph_breaks()
    demonstrate_dynamic_shapes()
    demonstrate_mark_dynamic()
    demonstrate_allow_in_graph()
    demonstrate_compiler_stances()
    demonstrate_debugging_tools()
    demonstrate_fullgraph_mode()
    demonstrate_performance_hints()
    demonstrate_xla_backend()
    demonstrate_eager_backend()
    demonstrate_disable_compiler()
    
    print("\n" + "=" * 60)
    print("PyTorch compiler examples completed!")
    print("\nKey takeaways:")
    print("- Use torch.compile for automatic optimizations")
    print("- Minimize graph breaks for better performance")
    print("- Use dynamic shapes for variable input sizes")
    print("- Debug with torch._dynamo.explain()")
    print("- Monitor performance with TORCH_LOGS")
    print("- Choose appropriate compilation modes")
    print("- Use fullgraph=True to catch breaks early")

if __name__ == "__main__":
    main()
