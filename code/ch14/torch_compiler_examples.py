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
# torch_compiler_examples.py
# Updated for PyTorch 2.8 nightly and Blackwell B200/B300 optimizations

import torch
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
    Updated for PyTorch 2.8 nightly features.
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

def configure_blackwell_optimizations():
    """
    Configure PyTorch 2.8 nightly optimizations for Blackwell B200/B300.
    """
    if torch.cuda.is_available():
        # Enable Blackwell B200/B300 specific optimizations
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.triton.autotune_mode = "max-autotune"
        
        # Enable advanced optimizations
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.triton.use_blackwell_tensor_cores = True
        
        # Memory optimizations for HBM3e
        torch._inductor.config.triton.hbm3e_memory_optimizations = True
        
        # Enhanced profiling configuration
        torch._inductor.config.triton.profiler_mode = "max-autotune"
        torch._inductor.config.triton.enable_blackwell_features = True
        
        print("Blackwell B200/B300 optimizations enabled")

def benchmark_compilation_modes():
    """
    Benchmark different torch.compile modes with PyTorch 2.8 nightly features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure Blackwell optimizations
    configure_blackwell_optimizations()
    
    # Create model and sample data
    model = SimpleModel().to(device)
    x = torch.randn(32, 1024, device=device)
    
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
            for _ in range(10):
                with nvtx.annotate(f"warmup_{mode}"):
                    _ = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark with enhanced timing and profiling
        start_time = time.time()
        
        # Enhanced profiler configuration for PyTorch 2.8
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
                for _ in range(100):
                    with nvtx.annotate(f"benchmark_{mode}"):
                        output = compiled_model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        results[mode] = avg_time
        print(f"Average time per forward pass: {avg_time:.4f}s")
        
        # Print profiling insights
        print(f"Top operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=3))
    
    # Test uncompiled baseline
    print(f"\nTesting uncompiled baseline")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            with nvtx.annotate("warmup_baseline"):
                _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            with nvtx.annotate("benchmark_baseline"):
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
    Example showing dynamic shape handling with PyTorch 2.8 enhancements.
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
    
    compiled_model = torch.compile(
        dynamic_model, 
        dynamic=True,
        fullgraph=True
    )
    
    for batch_size, seq_len, hidden_size in sizes:
        x = torch.randn(batch_size, seq_len, hidden_size)
        print(f"Processing shape {x.shape}")
        
        # First run may trigger recompilation
        with nvtx.annotate(f"dynamic_shape_{batch_size}_{seq_len}"):
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
    with nvtx.annotate("custom_operator_forward"):
        result = model_with_custom_op(x)
    
    # Backward pass also gets compiled
    with nvtx.annotate("custom_operator_backward"):
        result.backward()
    
    print(f"Custom operator result: {result.item()}")
    print(f"Gradient shape: {x.grad.shape}")

def memory_efficient_compilation():
    """
    Example showing memory-efficient patterns with torch.compile.
    Updated for PyTorch 2.8 memory optimizations.
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
    compiled_model = torch.compile(
        model, 
        mode='reduce-overhead',
        fullgraph=True,
        dynamic=True
    )
    
    x = torch.randn(32, 1024, device=device, requires_grad=True)
    
    # Measure memory usage with enhanced monitoring
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    with nvtx.annotate("memory_efficient_forward"):
        output = compiled_model(x)
    
    with nvtx.annotate("memory_efficient_backward"):
        loss = output.sum()
        loss.backward()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        print(f"Peak memory usage: {peak_memory:.2f} GB")
        print(f"Current memory usage: {current_memory:.2f} GB")

def demonstrate_blackwell_features():
    """
    Demonstrate Blackwell B200/B300 specific features.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell features")
        return
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"\nBlackwell B200/B300 Features:")
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Memory: {device_props.total_memory / 1e9:.1f} GB")
    memory_bandwidth = (2.0 * device_props.memoryClockRate * 0.001 * device_props.memoryBusWidth) / 8.0
    print(f"Memory Bandwidth: {memory_bandwidth:.1f} GB/s")
    
    # Test HBM3e memory optimizations
    print("\nTesting HBM3e memory optimizations...")
    
    # Large tensor operations
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        try:
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            with nvtx.annotate(f"gemm_{size}"):
                c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = end_time - start_time
            flops = 2 * size * size * size
            tflops = flops / (avg_time * 1e12)
            
            print(f"Size {size}x{size}: {avg_time:.4f}s, {tflops:.2f} TFLOPS")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Size {size}x{size}: OOM")
            else:
                print(f"Size {size}x{size}: Error")

def enhanced_profiling_example():
    """
    Demonstrate enhanced profiling with PyTorch 2.8 nightly features.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
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
            for _ in range(50):
                with nvtx.annotate("enhanced_profiling"):
                    output = model(x)
    
    print("\nEnhanced Profiling Results:")
    print("Top operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\nMemory profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\nFLOP analysis:")
    print(prof.key_averages().table(sort_by="flops", row_limit=5))

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
    Run all torch.compile examples with PyTorch 2.8 nightly features.
    """
    print("PyTorch 2.8 Compiler Examples (Blackwell B200/B300 Optimized)")
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
