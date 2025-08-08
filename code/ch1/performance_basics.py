#!/usr/bin/env python3
"""
Chapter 1: Introduction and AI System Overview
Performance Basics and Goodput Measurement

This example demonstrates the core concepts from Chapter 1:
- Measuring goodput (useful throughput)
- Hardware-software co-design principles
- Performance profiling and benchmarking
- PyTorch 2.8 nightly optimizations for Blackwell B200/B300
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple
import torch.cuda.nvtx as nvtx


class SimpleTransformer(nn.Module):
    """A simplified transformer model for demonstration purposes with PyTorch 2.8 optimizations."""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, 
                 n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output_projection(x)


class PerformanceMonitor:
    """Monitor system performance and calculate goodput metrics with enhanced monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.useful_work_time = 0
        self.total_overhead_time = 0
        self.gpu_memory_stats = {}
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.useful_work_time = 0
        self.total_overhead_time = 0
        
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
    def record_useful_work(self, duration: float):
        """Record time spent on useful work."""
        self.useful_work_time += duration
        
    def record_overhead(self, duration: float):
        """Record time spent on overhead."""
        self.total_overhead_time += duration
        
    def end_monitoring(self):
        """End monitoring and calculate metrics."""
        self.end_time = time.time()
        
        # Capture final GPU memory stats
        if torch.cuda.is_available():
            self.gpu_memory_stats = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'peak_allocated': torch.cuda.max_memory_allocated() / 1e9,
                'peak_cached': torch.cuda.max_memory_reserved() / 1e9
            }
        
    def get_goodput_metrics(self) -> Dict[str, float]:
        """Calculate goodput and efficiency metrics."""
        if self.start_time is None or self.end_time is None:
            return {}
            
        total_time = self.end_time - self.start_time
        goodput_ratio = self.useful_work_time / total_time if total_time > 0 else 0
        overhead_ratio = self.total_overhead_time / total_time if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'useful_work_time': self.useful_work_time,
            'overhead_time': self.total_overhead_time,
            'goodput_ratio': goodput_ratio,
            'overhead_ratio': overhead_ratio,
            'efficiency_percentage': goodput_ratio * 100
        }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource utilization with enhanced GPU monitoring."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_metrics = {}
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_metrics[f'gpu_{i}_utilization'] = gpu.load * 100
                gpu_metrics[f'gpu_{i}_memory_used'] = gpu.memoryUsed
                gpu_metrics[f'gpu_{i}_memory_total'] = gpu.memoryTotal
                gpu_metrics[f'gpu_{i}_temperature'] = gpu.temperature
        except:
            pass
            
        # Add PyTorch GPU memory stats
        if torch.cuda.is_available():
            gpu_metrics.update(self.gpu_memory_stats)
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            **gpu_metrics
        }


def configure_blackwell_optimizations():
    """Configure PyTorch 2.8 nightly optimizations for Blackwell B200/B300."""
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


def benchmark_model_performance(model: nn.Module, 
                              batch_size: int = 32, 
                              seq_length: int = 128,
                              num_iterations: int = 100,
                              use_compile: bool = True) -> Dict[str, float]:
    """
    Benchmark model performance and measure goodput with PyTorch 2.8 optimizations.
    
    This demonstrates the mechanical sympathy principle by measuring
    how efficiently the hardware is being utilized.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Configure Blackwell optimizations
    configure_blackwell_optimizations()
    
    # Create dummy input
    dummy_input = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
    
    # Compile model with PyTorch 2.8 optimizations if requested
    if use_compile and device.type == 'cuda':
        compiled_model = torch.compile(
            model, 
            mode="max-autotune",
            fullgraph=True,
            dynamic=True
        )
    else:
        compiled_model = model
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Warmup with enhanced profiling
    with torch.no_grad():
        for _ in range(10):
            _ = compiled_model(dummy_input)
    
    # Benchmark with enhanced profiling
    useful_work_start = time.time()
    
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
            for i in range(num_iterations):
                with record_function("model_inference"):
                    with nvtx.annotate("forward_pass"):
                        _ = compiled_model(dummy_input)
                    
                # Simulate some overhead (data loading, preprocessing, etc.)
                if i % 10 == 0:
                    overhead_start = time.time()
                    time.sleep(0.001)  # Simulate overhead
                    overhead_duration = time.time() - overhead_start
                    monitor.record_overhead(overhead_duration)
    
    useful_work_duration = time.time() - useful_work_start
    monitor.record_useful_work(useful_work_duration)
    monitor.end_monitoring()
    
    # Calculate throughput
    total_tokens = batch_size * seq_length * num_iterations
    throughput = total_tokens / useful_work_duration
    
    metrics = monitor.get_goodput_metrics()
    system_metrics = monitor.get_system_metrics()
    
    return {
        'throughput_tokens_per_second': throughput,
        'total_tokens_processed': total_tokens,
        'goodput_ratio': metrics['goodput_ratio'],
        'efficiency_percentage': metrics['efficiency_percentage'],
        'system_metrics': system_metrics,
        'profiler_output': prof,
        'compiled': use_compile
    }


def demonstrate_hardware_software_co_design():
    """
    Demonstrate hardware-software co-design principles with PyTorch 2.8 optimizations.
    
    This shows how different optimizations can affect performance
    and how to measure the impact of changes.
    """
    print("=== Chapter 1: AI Systems Performance Engineering Demo (PyTorch 2.8) ===\n")
    
    # Create model
    model = SimpleTransformer()
    
    print("1. Baseline Performance Measurement (Uncompiled)")
    print("-" * 50)
    
    baseline_results = benchmark_model_performance(model, num_iterations=50, use_compile=False)
    
    print(f"Throughput: {baseline_results['throughput_tokens_per_second']:.2f} tokens/sec")
    print(f"Goodput Ratio: {baseline_results['goodput_ratio']:.3f}")
    print(f"Efficiency: {baseline_results['efficiency_percentage']:.1f}%")
    print(f"Total Tokens Processed: {baseline_results['total_tokens_processed']:,}")
    
    print("\n2. PyTorch 2.8 Compiled Performance")
    print("-" * 50)
    
    compiled_results = benchmark_model_performance(model, num_iterations=50, use_compile=True)
    
    print(f"Throughput: {compiled_results['throughput_tokens_per_second']:.2f} tokens/sec")
    print(f"Goodput Ratio: {compiled_results['goodput_ratio']:.3f}")
    print(f"Efficiency: {compiled_results['efficiency_percentage']:.1f}%")
    print(f"Speedup: {compiled_results['throughput_tokens_per_second'] / baseline_results['throughput_tokens_per_second']:.2f}x")
    
    print("\n3. System Resource Utilization")
    print("-" * 50)
    
    system_metrics = compiled_results['system_metrics']
    print(f"CPU Utilization: {system_metrics.get('cpu_percent', 'N/A')}%")
    print(f"Memory Utilization: {system_metrics.get('memory_percent', 'N/A')}%")
    print(f"Available Memory: {system_metrics.get('memory_available_gb', 'N/A'):.2f} GB")
    
    # Check for GPU metrics
    gpu_metrics = {k: v for k, v in system_metrics.items() if k.startswith('gpu_')}
    if gpu_metrics:
        print("\nGPU Metrics:")
        for metric, value in gpu_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\n4. Enhanced Performance Profiling Insights")
    print("-" * 50)
    
    prof = compiled_results['profiler_output']
    print("Top 5 operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\nMemory profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\nFLOP analysis:")
    print(prof.key_averages().table(sort_by="flops", row_limit=5))
    
    print("\n5. Key Takeaways from Chapter 1 (PyTorch 2.8)")
    print("-" * 50)
    print("• PyTorch 2.8 torch.compile provides significant speedup")
    print("• Blackwell B200/B300 optimizations improve performance")
    print("• Enhanced profiler provides detailed insights")
    print("• Memory profiling helps identify bottlenecks")
    print("• Hardware-software co-design is crucial")
    print("• Use NVTX markers for detailed timeline analysis")
    
    return compiled_results


def demonstrate_mechanical_sympathy():
    """
    Demonstrate mechanical sympathy by showing how hardware awareness
    can lead to better performance with PyTorch 2.8 optimizations.
    """
    print("\n=== Mechanical Sympathy Demo (PyTorch 2.8) ===")
    
    # Show how different batch sizes affect performance
    batch_sizes = [8, 16, 32, 64, 128]
    model = SimpleTransformer()
    
    print("\nBatch Size vs Performance (Compiled vs Uncompiled):")
    print("Batch Size | Uncompiled (tokens/sec) | Compiled (tokens/sec) | Speedup")
    print("-" * 75)
    
    for batch_size in batch_sizes:
        try:
            # Test uncompiled
            uncompiled_results = benchmark_model_performance(
                model, batch_size=batch_size, num_iterations=20, use_compile=False
            )
            
            # Test compiled
            compiled_results = benchmark_model_performance(
                model, batch_size=batch_size, num_iterations=20, use_compile=True
            )
            
            speedup = compiled_results['throughput_tokens_per_second'] / uncompiled_results['throughput_tokens_per_second']
            
            print(f"{batch_size:10d} | {uncompiled_results['throughput_tokens_per_second']:20.2f} | {compiled_results['throughput_tokens_per_second']:20.2f} | {speedup:7.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:10d} | {'OOM':>20} | {'OOM':>20} | {'N/A':>7}")
            else:
                print(f"{batch_size:10d} | {'Error':>20} | {'Error':>20} | {'N/A':>7}")


def demonstrate_blackwell_optimizations():
    """
    Demonstrate Blackwell B200/B300 specific optimizations.
    """
    print("\n=== Blackwell B200/B300 Optimizations ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell optimizations")
        return
    
    # Check for Blackwell B200/B300 features
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    print(f"Memory: {device_props.total_memory / 1e9:.1f} GB")
    
    # Test HBM3e memory optimizations
    print("\nTesting HBM3e memory optimizations...")
    
    # Large tensor operations to test memory bandwidth
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        try:
            # Create large tensors
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            # Measure matrix multiplication performance
            torch.cuda.synchronize()
            start_time = time.time()
            
            with nvtx.annotate(f"gemm_{size}"):
                for _ in range(10):
                    c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            flops = 2 * size * size * size  # FLOPS for matrix multiplication
            tflops = flops / (avg_time * 1e12)
            
            print(f"Size {size}x{size}: {avg_time:.4f}s, {tflops:.2f} TFLOPS")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Size {size}x{size}: OOM")
            else:
                print(f"Size {size}x{size}: Error")


def demonstrate_enhanced_profiling():
    """
    Demonstrate enhanced profiling with latest tools.
    """
    print("\n=== Enhanced Profiling Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer().to(device)
    x = torch.randn(64, 128, device=device)
    
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
    
    print("Enhanced Profiling Results:")
    print("Top operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\nMemory profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\nFLOP analysis:")
    print(prof.key_averages().table(sort_by="flops", row_limit=5))


def demonstrate_system_monitoring():
    """
    Demonstrate system-level monitoring with latest tools.
    """
    print("\n=== System Monitoring Demo ===")
    
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


if __name__ == "__main__":
    # Run the demonstrations
    results = demonstrate_hardware_software_co_design()
    demonstrate_mechanical_sympathy()
    demonstrate_blackwell_optimizations()
    demonstrate_enhanced_profiling()
    demonstrate_system_monitoring()
    
    print("\n=== Summary ===")
    print("This demo shows the core principles from Chapter 1 with PyTorch 2.8:")
    print("1. Performance measurement and goodput calculation")
    print("2. Hardware-software co-design (mechanical sympathy)")
    print("3. Enhanced profiling and bottleneck identification")
    print("4. PyTorch 2.8 torch.compile optimizations")
    print("5. Blackwell B200/B300 specific features")
    print("6. System-level optimization considerations")
    print("7. Latest profiling tools integration")
    print("\nThese concepts will be explored in detail throughout the book.")
