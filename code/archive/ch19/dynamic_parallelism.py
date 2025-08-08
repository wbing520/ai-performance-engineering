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
Dynamic parallelism switching for inference optimization.
This demonstrates runtime adaptation between TP, PP, and hybrid strategies.
"""

import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn

class ParallelismStrategy(Enum):
    """Available parallelism strategies"""
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"
    DATA_PARALLEL = "data_parallel"

@dataclass
class SystemMetrics:
    """Current system metrics for decision making"""
    gpu_memory_utilization: float  # percentage
    gpu_compute_utilization: float  # percentage
    nvlink_utilization: float  # percentage
    queue_length: int
    avg_latency_ms: float
    throughput_tokens_per_sec: float

@dataclass
class Request:
    """Inference request with metadata"""
    request_id: str
    prompt_length: int
    expected_output_length: int
    priority: str  # "high", "medium", "low"
    timestamp: float

class DynamicParallelismManager:
    """Manages dynamic parallelism strategy switching"""
    
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.current_strategy = ParallelismStrategy.TENSOR_PARALLEL
        self.strategy_history = []
        self.adaptation_count = 0
        
        # Pre-loaded model replicas for different strategies
        self.model_replicas = {
            ParallelismStrategy.TENSOR_PARALLEL: self._create_tp_model(),
            ParallelismStrategy.PIPELINE_PARALLEL: self._create_pp_model(),
            ParallelismStrategy.HYBRID: self._create_hybrid_model(),
            ParallelismStrategy.DATA_PARALLEL: self._create_dp_model()
        }
        
        # Strategy-specific configurations
        self.strategy_configs = {
            ParallelismStrategy.TENSOR_PARALLEL: {
                "tp_size": 4,
                "pp_size": 1,
                "dp_size": 2
            },
            ParallelismStrategy.PIPELINE_PARALLEL: {
                "tp_size": 1,
                "pp_size": 4,
                "dp_size": 2
            },
            ParallelismStrategy.HYBRID: {
                "tp_size": 2,
                "pp_size": 2,
                "dp_size": 2
            },
            ParallelismStrategy.DATA_PARALLEL: {
                "tp_size": 1,
                "pp_size": 1,
                "dp_size": 8
            }
        }
    
    def _create_tp_model(self):
        """Create tensor parallel model replica"""
        print("Creating tensor parallel model replica...")
        return {"type": "tensor_parallel", "tp_size": 4}
    
    def _create_pp_model(self):
        """Create pipeline parallel model replica"""
        print("Creating pipeline parallel model replica...")
        return {"type": "pipeline_parallel", "pp_size": 4}
    
    def _create_hybrid_model(self):
        """Create hybrid TP+PP model replica"""
        print("Creating hybrid TP+PP model replica...")
        return {"type": "hybrid", "tp_size": 2, "pp_size": 2}
    
    def _create_dp_model(self):
        """Create data parallel model replica"""
        print("Creating data parallel model replica...")
        return {"type": "data_parallel", "dp_size": 8}
    
    def choose_worker_pool(self, seq_len: int, gpu_mem_util: float, 
                          concurrent_reqs: int) -> ParallelismStrategy:
        """
        Choose the best parallelism strategy based on current conditions
        
        Args:
            seq_len: Input sequence length
            gpu_mem_util: GPU memory utilization (0-1)
            concurrent_reqs: Number of concurrent requests
            
        Returns:
            Optimal parallelism strategy
        """
        # For long contexts or high memory pressure, use hybrid pipeline + tensor parallelism
        if seq_len > 4096 or gpu_mem_util > 0.8:
            return ParallelismStrategy.HYBRID
        
        # For many simultaneous small requests, stick with tensor parallelism
        if concurrent_reqs > 4:
            return ParallelismStrategy.TENSOR_PARALLEL
        
        # For extremely large models, use hybrid
        if seq_len > 8192:
            return ParallelismStrategy.HYBRID
        
        # For latency-sensitive requests, use tensor parallelism
        if concurrent_reqs == 1 and seq_len < 1024:
            return ParallelismStrategy.TENSOR_PARALLEL
        
        # Fallback to tensor-parallel for typical workloads
        return ParallelismStrategy.TENSOR_PARALLEL
    
    def adapt_strategy(self, metrics: SystemMetrics, 
                      current_requests: List[Request]) -> bool:
        """
        Adapt parallelism strategy based on current metrics
        
        Args:
            metrics: Current system metrics
            current_requests: List of current requests
            
        Returns:
            True if strategy was changed, False otherwise
        """
        # Calculate average sequence length
        if not current_requests:
            return False
        
        avg_seq_len = sum(req.prompt_length for req in current_requests) / len(current_requests)
        
        # Choose optimal strategy
        optimal_strategy = self.choose_worker_pool(
            seq_len=avg_seq_len,
            gpu_mem_util=metrics.gpu_memory_utilization,
            concurrent_reqs=len(current_requests)
        )
        
        # Check if we need to switch strategies
        if optimal_strategy != self.current_strategy:
            print(f"Switching from {self.current_strategy.value} to {optimal_strategy.value}")
            print(f"  Avg seq len: {avg_seq_len:.0f}")
            print(f"  GPU mem util: {metrics.gpu_memory_utilization:.1%}")
            print(f"  Concurrent reqs: {len(current_requests)}")
            
            # Record strategy change
            self.strategy_history.append({
                "from": self.current_strategy.value,
                "to": optimal_strategy.value,
                "timestamp": time.time(),
                "metrics": metrics,
                "avg_seq_len": avg_seq_len
            })
            
            self.current_strategy = optimal_strategy
            self.adaptation_count += 1
            return True
        
        return False
    
    def get_current_model(self):
        """Get the current model replica"""
        return self.model_replicas[self.current_strategy]
    
    def get_strategy_config(self):
        """Get configuration for current strategy"""
        return self.strategy_configs[self.current_strategy]
    
    def get_adaptation_stats(self) -> Dict:
        """Get adaptation statistics"""
        return {
            "total_adaptations": self.adaptation_count,
            "current_strategy": self.current_strategy.value,
            "strategy_history": self.strategy_history
        }

class AdaptiveInferenceEngine:
    """Adaptive inference engine with dynamic parallelism"""
    
    def __init__(self, num_gpus: int = 8):
        self.parallelism_manager = DynamicParallelismManager(num_gpus)
        self.request_queue = []
        self.processing_requests = []
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()
    
    def submit_request(self, request: Request):
        """Submit a new inference request"""
        self.request_queue.append(request)
        self.total_requests += 1
        print(f"Submitted request {request.request_id} (length: {request.prompt_length})")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics (simulated)"""
        # Simulate realistic metrics
        gpu_mem_util = 0.6 + 0.3 * random.random()  # 60-90%
        gpu_compute_util = 0.7 + 0.25 * random.random()  # 70-95%
        nvlink_util = 0.4 + 0.4 * random.random()  # 40-80%
        
        return SystemMetrics(
            gpu_memory_utilization=gpu_mem_util,
            gpu_compute_utilization=gpu_compute_util,
            nvlink_utilization=nvlink_util,
            queue_length=len(self.request_queue),
            avg_latency_ms=120.0 + 50.0 * random.random(),
            throughput_tokens_per_sec=800.0 + 200.0 * random.random()
        )
    
    def process_requests(self, num_iterations: int = 10):
        """Process requests with dynamic adaptation"""
        print(f"Starting adaptive inference with {len(self.request_queue)} requests")
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get current metrics
            metrics = self.get_system_metrics()
            
            # Adapt strategy if needed
            strategy_changed = self.parallelism_manager.adapt_strategy(
                metrics, self.processing_requests
            )
            
            # Process some requests
            self._process_batch(metrics)
            
            # Update processing requests
            if self.request_queue:
                # Move some requests from queue to processing
                batch_size = min(4, len(self.request_queue))
                for _ in range(batch_size):
                    if self.request_queue:
                        req = self.request_queue.pop(0)
                        self.processing_requests.append(req)
            
            # Simulate request completion
            if self.processing_requests:
                completed_req = self.processing_requests.pop(0)
                tokens_generated = completed_req.expected_output_length
                self.total_tokens_generated += tokens_generated
                print(f"Completed request {completed_req.request_id} ({tokens_generated} tokens)")
            
            # Print current state
            self._print_status(metrics, strategy_changed)
            
            time.sleep(0.5)  # Simulate processing time
    
    def _process_batch(self, metrics: SystemMetrics):
        """Process a batch of requests"""
        current_model = self.parallelism_manager.get_current_model()
        config = self.parallelism_manager.get_strategy_config()
        
        print(f"Processing with {current_model['type']} (config: {config})")
        
        # Simulate processing time based on strategy
        if self.parallelism_manager.current_strategy == ParallelismStrategy.TENSOR_PARALLEL:
            processing_time = 0.1  # Fast for TP
        elif self.parallelism_manager.current_strategy == ParallelismStrategy.PIPELINE_PARALLEL:
            processing_time = 0.2  # Slower for PP due to bubbles
        elif self.parallelism_manager.current_strategy == ParallelismStrategy.HYBRID:
            processing_time = 0.15  # Medium for hybrid
        else:
            processing_time = 0.08  # Fastest for DP
        
        time.sleep(processing_time)
    
    def _print_status(self, metrics: SystemMetrics, strategy_changed: bool):
        """Print current system status"""
        print(f"Current Strategy: {self.parallelism_manager.current_strategy.value}")
        print(f"GPU Memory: {metrics.gpu_memory_utilization:.1%}")
        print(f"GPU Compute: {metrics.gpu_compute_utilization:.1%}")
        print(f"NVLink: {metrics.nvlink_utilization:.1%}")
        print(f"Queue: {len(self.request_queue)}, Processing: {len(self.processing_requests)}")
        print(f"Strategy Changed: {strategy_changed}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        runtime = time.time() - self.start_time
        throughput = self.total_tokens_generated / runtime if runtime > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_generated,
            "runtime_seconds": runtime,
            "throughput_tokens_per_sec": throughput,
            "adaptation_stats": self.parallelism_manager.get_adaptation_stats()
        }

def create_sample_requests() -> List[Request]:
    """Create sample requests with different characteristics"""
    requests = []
    
    # Short, latency-sensitive requests
    for i in range(5):
        requests.append(Request(
            request_id=f"short_{i}",
            prompt_length=random.randint(50, 200),
            expected_output_length=random.randint(20, 100),
            priority="high",
            timestamp=time.time()
        ))
    
    # Medium requests
    for i in range(3):
        requests.append(Request(
            request_id=f"medium_{i}",
            prompt_length=random.randint(500, 1500),
            expected_output_length=random.randint(100, 300),
            priority="medium",
            timestamp=time.time()
        ))
    
    # Long, memory-intensive requests
    for i in range(2):
        requests.append(Request(
            request_id=f"long_{i}",
            prompt_length=random.randint(3000, 8000),
            expected_output_length=random.randint(500, 1000),
            priority="low",
            timestamp=time.time()
        ))
    
    return requests

def main():
    """Main function demonstrating dynamic parallelism"""
    print("Dynamic Parallelism Switching Example")
    print("=====================================")
    
    # Create adaptive inference engine
    engine = AdaptiveInferenceEngine(num_gpus=8)
    
    # Create sample requests
    requests = create_sample_requests()
    
    # Submit requests
    for request in requests:
        engine.submit_request(request)
    
    # Process requests with dynamic adaptation
    engine.process_requests(num_iterations=15)
    
    # Print final statistics
    stats = engine.get_performance_stats()
    print("\n=== Final Performance Statistics ===")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Total Tokens Generated: {stats['total_tokens']}")
    print(f"Runtime: {stats['runtime_seconds']:.2f} seconds")
    print(f"Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")
    
    adaptation_stats = stats['adaptation_stats']
    print(f"Total Adaptations: {adaptation_stats['total_adaptations']}")
    print(f"Final Strategy: {adaptation_stats['current_strategy']}")
    
    print("\nStrategy History:")
    for i, change in enumerate(adaptation_stats['strategy_history']):
        print(f"  {i+1}. {change['from']} → {change['to']} "
              f"(seq_len: {change['avg_seq_len']:.0f}, "
              f"mem: {change['metrics'].gpu_memory_utilization:.1%})")
    
    print("\nDynamic parallelism example completed successfully!")

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
