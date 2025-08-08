#!/usr/bin/env python3
"""
Chapter 1: Introduction and AI System Overview
Performance Basics and Goodput Measurement

This example demonstrates the core concepts from Chapter 1:
- Measuring goodput (useful throughput)
- Hardware-software co-design principles
- Performance profiling and benchmarking
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple


class SimpleTransformer(nn.Module):
    """A simplified transformer model for demonstration purposes."""
    
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
    """Monitor system performance and calculate goodput metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.useful_work_time = 0
        self.total_overhead_time = 0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.useful_work_time = 0
        self.total_overhead_time = 0
        
    def record_useful_work(self, duration: float):
        """Record time spent on useful work."""
        self.useful_work_time += duration
        
    def record_overhead(self, duration: float):
        """Record time spent on overhead."""
        self.total_overhead_time += duration
        
    def end_monitoring(self):
        """End monitoring and calculate metrics."""
        self.end_time = time.time()
        
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
        """Get current system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        gpu_metrics = {}
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_metrics[f'gpu_{i}_utilization'] = gpu.load * 100
                gpu_metrics[f'gpu_{i}_memory_used'] = gpu.memoryUsed
                gpu_metrics[f'gpu_{i}_memory_total'] = gpu.memoryTotal
        except:
            pass
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            **gpu_metrics
        }


def benchmark_model_performance(model: nn.Module, 
                              batch_size: int = 32, 
                              seq_length: int = 128,
                              num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark model performance and measure goodput.
    
    This demonstrates the mechanical sympathy principle by measuring
    how efficiently the hardware is being utilized.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark with profiling
    useful_work_start = time.time()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for i in range(num_iterations):
                with record_function("model_inference"):
                    _ = model(dummy_input)
                    
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
        'profiler_output': prof
    }


def demonstrate_hardware_software_co_design():
    """
    Demonstrate hardware-software co-design principles.
    
    This shows how different optimizations can affect performance
    and how to measure the impact of changes.
    """
    print("=== Chapter 1: AI Systems Performance Engineering Demo ===\n")
    
    # Create model
    model = SimpleTransformer()
    
    print("1. Baseline Performance Measurement")
    print("-" * 40)
    
    baseline_results = benchmark_model_performance(model, num_iterations=50)
    
    print(f"Throughput: {baseline_results['throughput_tokens_per_second']:.2f} tokens/sec")
    print(f"Goodput Ratio: {baseline_results['goodput_ratio']:.3f}")
    print(f"Efficiency: {baseline_results['efficiency_percentage']:.1f}%")
    print(f"Total Tokens Processed: {baseline_results['total_tokens_processed']:,}")
    
    print("\n2. System Resource Utilization")
    print("-" * 40)
    
    system_metrics = baseline_results['system_metrics']
    print(f"CPU Utilization: {system_metrics.get('cpu_percent', 'N/A')}%")
    print(f"Memory Utilization: {system_metrics.get('memory_percent', 'N/A')}%")
    print(f"Available Memory: {system_metrics.get('memory_available_gb', 'N/A'):.2f} GB")
    
    # Check for GPU metrics
    gpu_metrics = {k: v for k, v in system_metrics.items() if k.startswith('gpu_')}
    if gpu_metrics:
        print("\nGPU Metrics:")
        for metric, value in gpu_metrics.items():
            print(f"  {metric}: {value}")
    
    print("\n3. Performance Profiling Insights")
    print("-" * 40)
    
    prof = baseline_results['profiler_output']
    print("Top 5 operations by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\n4. Key Takeaways from Chapter 1")
    print("-" * 40)
    print("• Measure goodput, not just raw throughput")
    print("• Hardware-software co-design is crucial")
    print("• Small optimizations can have large impacts at scale")
    print("• Use profiling to identify bottlenecks")
    print("• Consider the entire system stack")
    
    return baseline_results


def demonstrate_mechanical_sympathy():
    """
    Demonstrate mechanical sympathy by showing how hardware awareness
    can lead to better performance.
    """
    print("\n=== Mechanical Sympathy Demo ===")
    
    # Show how different batch sizes affect performance
    batch_sizes = [8, 16, 32, 64, 128]
    model = SimpleTransformer()
    
    print("\nBatch Size vs Performance:")
    print("Batch Size | Throughput (tokens/sec) | Goodput Ratio")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        try:
            results = benchmark_model_performance(model, batch_size=batch_size, num_iterations=20)
            print(f"{batch_size:10d} | {results['throughput_tokens_per_second']:20.2f} | {results['goodput_ratio']:12.3f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:10d} | {'OOM':>20} | {'N/A':>12}")
            else:
                print(f"{batch_size:10d} | {'Error':>20} | {'N/A':>12}")


if __name__ == "__main__":
    # Run the demonstrations
    results = demonstrate_hardware_software_co_design()
    demonstrate_mechanical_sympathy()
    
    print("\n=== Summary ===")
    print("This demo shows the core principles from Chapter 1:")
    print("1. Performance measurement and goodput calculation")
    print("2. Hardware-software co-design (mechanical sympathy)")
    print("3. Profiling and bottleneck identification")
    print("4. System-level optimization considerations")
    print("\nThese concepts will be explored in detail throughout the book.")
