import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    return "blackwell" if compute_capability == "10.0" else "other"


def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": []
    }

"""torch_compiler_examples.py
Chapter 14: Torch Compile Examples

Torch compile usage patterns optimized for Blackwell GPUs."""

import torch._dynamo as dynamo
import time
from torch.nn import functional as F
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import psutil
import GPUtil

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
    
    # Use torch.where for element-wise conditional operations instead of torch.cond
    # This avoids aliasing issues with torch.cond
    b = torch.where(b.sum() < 0, -b, b)
    
    return x * b

def model_with_torch_cond(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Example using torch.where for data-dependent control flow.
    """
    # Compute x as before
    x = a / (torch.abs(a) + 1)
    
    # Use torch.where for element-wise conditional operations
    # This is more appropriate than torch.cond for this use case
    b = torch.where(b.sum() < 0, -b, b)
    
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
    Updated for PyTorch 2.9 nightly features.
    """
    def __init__(self, input_size=256, hidden_size=256, output_size=10):  # compact to keep profiling under a few seconds
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

def configure_blackwell_optimizations():
    """
    Configure PyTorch 2.9 nightly optimizations for Blackwell B200/B300.
    """
    if torch.cuda.is_available():
        # Enable Blackwell B200/B300 specific optimizations (only if they exist)
        if hasattr(torch._inductor.config.triton, 'unique_kernel_names'):
            torch._inductor.config.triton.unique_kernel_names = True
        
        # Note: The following options are not available in the current PyTorch version
        # They are commented out to avoid AttributeError
        # 
        # if hasattr(torch._inductor.config.triton, 'use_blackwell_optimizations'):
        #     torch._inductor.config.triton.use_blackwell_optimizations = True
        # if hasattr(torch._inductor.config.triton, 'hbm3e_optimizations'):
        #     torch._inductor.config.triton.hbm3e_optimizations = True
        # if hasattr(torch._inductor.config.triton, 'use_blackwell_tensor_cores'):
        #     torch._inductor.config.triton.use_blackwell_tensor_cores = True
        # if hasattr(torch._inductor.config.triton, 'hbm3e_memory_optimizations'):
        #     torch._inductor.config.triton.hbm3e_memory_optimizations = True
        # if hasattr(torch._inductor.config.triton, 'profiler_mode'):
        #     torch._inductor.config.triton.profiler_mode = "max-autotune"
        # if hasattr(torch._inductor.config.triton, 'enable_blackwell_features'):
        #     torch._inductor.config.triton.enable_blackwell_features = True
        
        print("Blackwell B200/B300 optimizations enabled (compatible features only)")

def benchmark_compilation_modes():
    """
    Benchmark different torch.compile modes with PyTorch 2.9 nightly features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure Blackwell optimizations
    configure_blackwell_optimizations()
    
    # Create model and sample data
    model = SimpleModel().to(device)
    x = torch.randn(16, 256, device=device)
    
    # Test different compilation modes
    modes = ['default', 'reduce-overhead', 'max-autotune']
    
    results = {}
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        
        # Compile model with enhanced options
        if mode == 'default':
            compiled_model = torch.compile(
                model,
                mode='default',
                fullgraph=True,
                dynamic=True
            )
        else:
            compiled_model = torch.compile(
                model,
                mode=mode,
                fullgraph=True,
                dynamic=True
            )
        
        # Warmup with enhanced profiling
        with torch.no_grad():
            for _ in range(3):
                with nvtx.range(f"warmup_{mode}"):
                    _ = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark with enhanced timing and profiling
        start_time = time.time()
        
        # Enhanced profiler configuration for PyTorch 2.9
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            profile_memory=True,
            schedule=schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            )
        ) as prof:
            with torch.no_grad():
                for _ in range(20):
                    with nvtx.range(f"benchmark_{mode}"):
                        output = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 20
        
        results[mode] = avg_time
        print(f"Average time per forward pass: {avg_time:.4f}s")
        
        # Print profiling insights
        try:
            if hasattr(prof, 'key_averages') and prof.key_averages() is not None:
                print(f"Top operations by CUDA time:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=3))
            else:
                print("Profiler data not available")
        except Exception as e:
            print(f"Profiler data not available: {e}")
    
    # Test uncompiled baseline
    print(f"\nTesting uncompiled baseline")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            with nvtx.range("warmup_baseline"):
                _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(20):
            with nvtx.range("benchmark_baseline"):
                output = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    baseline_time = (end_time - start_time) / 20
    
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
    Example showing dynamic shape handling with PyTorch 2.9 enhancements.
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
    sizes = [(16, 40, 128), (16, 120, 128), (8, 60, 128)]
    
    compiled_model = torch.compile(
        dynamic_model, 
        dynamic=True,
        fullgraph=True
    )
    
    for batch_size, seq_len, hidden_size in sizes:
        x = torch.randn(batch_size, seq_len, hidden_size)
        print(f"Processing shape {x.shape}")
        
        # First run may trigger recompilation
        with nvtx.range(f"dynamic_shape_{batch_size}_{seq_len}"):
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
    
    x = torch.randn(256, 256, requires_grad=True)
    
    # Forward pass
    with nvtx.range("custom_operator_forward"):
        result = model_with_custom_op(x)
    
    # Backward pass also gets compiled
    with nvtx.range("custom_operator_backward"):
        result.backward()
    
    print(f"Custom operator result: {result.item()}")
    print(f"Gradient shape: {x.grad.shape}")

def memory_efficient_compilation():
    """
    Example showing memory-efficient patterns with torch.compile.
    Updated for PyTorch 2.9 memory optimizations.
    """
    class MemoryEfficientModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(256, 256) for _ in range(5)
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
    compiled_model = torch.compile(
        model, 
        mode='reduce-overhead',
        fullgraph=True,
        dynamic=True
    )
    
    x = torch.randn(16, 256, device=device, requires_grad=True)
    
    # Measure memory usage with enhanced monitoring
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    with nvtx.range("memory_efficient_forward"):
        output = compiled_model(x)
    
    with nvtx.range("memory_efficient_backward"):
        loss = output.sum()
        loss.backward()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Peak memory usage: {peak_memory:.2f} GB")
        print(f"Current memory usage: {current_memory:.2f} GB")

def demonstrate_blackwell_features():
    """
    Demonstrate Blackwell B200/B300 specific features and optimizations.
    """
    print("\nBlackwell B200/B300 Features:")
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        print(f"GPU: {device_props.name}")
        print(f"Compute Capability: {compute_capability}")
        print(f"Memory: {device_props.total_memory / 1e9:.1f} GB")
        print(f"Multi-Processors: {device_props.multi_processor_count}")
        print(f"Max Threads per MP: {device_props.max_threads_per_multi_processor}")
        print(f"Shared Memory per Block: {device_props.shared_memory_per_block / 1024:.0f} KB")
        print(f"Warp Size: {device_props.warp_size}")
        
        if compute_capability == "10.0":  # Blackwell B200/B300
            print("✓ Blackwell B200/B300 Architecture Detected")
            print("• HBM3e Memory (8.0 TB/s bandwidth)")
            print("• 5th Gen Tensor Cores")
            print("• TMA (Tensor Memory Accelerator)")
            print("• NVLink-C2C Communication")
            print("• Stream-ordered Memory Allocation")
        else:
            print("• Standard CUDA Architecture")
            print("• Compatible with PyTorch 2.9 optimizations")

def enhanced_profiling_example():
    """
    Demonstrate enhanced profiling with PyTorch 2.9 nightly features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    x = torch.randn(32, 256, device=device)
    
    # Enhanced profiler configuration
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )
    ) as prof:
        with torch.no_grad():
            for _ in range(20):
                with nvtx.range("enhanced_profiling"):
                    output = model(x)
    
    print("\nEnhanced Profiling Results:")
    try:
        print("Top operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        
        print("\nMemory profiling:")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
        
        print("\nFLOP analysis:")
        print(prof.key_averages().table(sort_by="flops", row_limit=5))
    except Exception as e:
        print(f"Profiler data not available: {e}")

def system_monitoring():
    """
    Demonstrate system-level monitoring with latest tools.
    """
    print("\nSystem Monitoring:")
    
    # CPU monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / 1e9:.2f} GB")
    
    # GPU monitoring
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            print(f"  Utilization: {gpu.load * 100:.1f}%")
            print(f"  Memory Used: {gpu.memoryUsed} MB")
            print(f"  Memory Total: {gpu.memoryTotal} MB")
            print(f"  Temperature: {gpu.temperature}°C")
    except:
        print("GPU monitoring not available")
    
    # PyTorch GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"PyTorch GPU Memory: {allocated:.2f} GB allocated, {cached:.2f} GB cached")

def main():
    """
    Run all torch.compile examples with PyTorch 2.9 nightly features.
    """
    print("PyTorch 2.9 Compiler Examples (Blackwell B200/B300 Optimized)")
    print("=" * 60)
    
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
    
    print("\n6. Blackwell B200/B300 Features:")
    demonstrate_blackwell_features()
    
    print("\n7. Enhanced Profiling:")
    enhanced_profiling_example()
    
    print("\n8. System Monitoring:")
    system_monitoring()
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main()
