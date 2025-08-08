#!/usr/bin/env python3
"""
dynamic_parallelism.py
Chapter 19: Dynamic Parallelism for Adaptive Inference

Implementation of dynamic parallelism strategies that adaptively choose
between tensor, pipeline, and hybrid parallelism based on workload characteristics.

Based on Chapter 19 content about real-time parallelism adaptation.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import time
import psutil
import threading
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelismStrategy(Enum):
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"
    DATA_PARALLEL = "data_parallel"


@dataclass
class WorkloadMetrics:
    """Runtime metrics used for parallelism decisions."""
    seq_len: int
    batch_size: int
    gpu_memory_util: float
    concurrent_requests: int
    avg_latency_ms: float
    throughput_tokens_per_sec: float
    memory_bandwidth_util: float
    compute_utilization: float


@dataclass
class ParallelismConfig:
    """Configuration for a specific parallelism strategy."""
    strategy: ParallelismStrategy
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    estimated_latency_ms: float
    estimated_throughput: float
    memory_efficiency: float


class DynamicParallelismRouter:
    """
    Router that dynamically selects parallelism strategy based on workload.
    Implementation of Chapter 19's adaptive parallelism concepts.
    """
    
    def __init__(self, available_gpus: int = 8):
        self.available_gpus = available_gpus
        self.current_strategy = ParallelismStrategy.TENSOR_PARALLEL
        
        # Performance profiles for different strategies
        self.strategy_profiles = self._initialize_strategy_profiles()
        
        # Runtime metrics tracking
        self.metrics_history: List[WorkloadMetrics] = []
        self.strategy_performance: Dict[ParallelismStrategy, List[float]] = {
            strategy: [] for strategy in ParallelismStrategy
        }
        
        # Switching thresholds
        self.seq_len_threshold_long = 1024
        self.seq_len_threshold_short = 256
        self.memory_threshold_high = 0.8
        self.memory_threshold_low = 0.4
        self.latency_threshold_ms = 100
        
        # Cooldown to prevent thrashing
        self.last_switch_time = 0
        self.switch_cooldown_sec = 5.0
    
    def _initialize_strategy_profiles(self) -> Dict[ParallelismStrategy, ParallelismConfig]:
        """Initialize performance profiles for different strategies."""
        profiles = {}
        
        # Tensor Parallel: Best for latency-sensitive, moderate sequences
        profiles[ParallelismStrategy.TENSOR_PARALLEL] = ParallelismConfig(
            strategy=ParallelismStrategy.TENSOR_PARALLEL,
            tensor_parallel_size=self.available_gpus,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            estimated_latency_ms=50.0,
            estimated_throughput=1000.0,
            memory_efficiency=0.7
        )
        
        # Pipeline Parallel: Best for very long sequences
        profiles[ParallelismStrategy.PIPELINE_PARALLEL] = ParallelismConfig(
            strategy=ParallelismStrategy.PIPELINE_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=self.available_gpus,
            data_parallel_size=1,
            estimated_latency_ms=80.0,
            estimated_throughput=800.0,
            memory_efficiency=0.9
        )
        
        # Hybrid: Best for very long sequences with high memory usage
        profiles[ParallelismStrategy.HYBRID] = ParallelismConfig(
            strategy=ParallelismStrategy.HYBRID,
            tensor_parallel_size=self.available_gpus // 2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            estimated_latency_ms=65.0,
            estimated_throughput=900.0,
            memory_efficiency=0.85
        )
        
        # Data Parallel: Best for high throughput, many short sequences
        profiles[ParallelismStrategy.DATA_PARALLEL] = ParallelismConfig(
            strategy=ParallelismStrategy.DATA_PARALLEL,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=self.available_gpus,
            estimated_latency_ms=120.0,
            estimated_throughput=1500.0,
            memory_efficiency=0.6
        )
        
        return profiles
    
    def choose_worker_pool(self, seq_len: int, gpu_mem_util: float, 
                          concurrent_reqs: int) -> ParallelismStrategy:
        """
        Core decision function from Chapter 19.
        Chooses optimal parallelism strategy based on runtime conditions.
        """
        # For long contexts or high memory pressure,
        # use hybrid pipeline + tensor parallelism
        if seq_len > self.seq_len_threshold_long or gpu_mem_util > self.memory_threshold_high:
            if gpu_mem_util > 0.9:
                # Extreme memory pressure - use pipeline for memory efficiency
                return ParallelismStrategy.PIPELINE_PARALLEL
            else:
                # High memory but manageable - use hybrid
                return ParallelismStrategy.HYBRID
        
        # For many concurrent short requests, use data parallelism
        elif concurrent_reqs > 32 and seq_len < self.seq_len_threshold_short:
            return ParallelismStrategy.DATA_PARALLEL
        
        # For latency-sensitive requests, use tensor parallelism
        else:
            return ParallelismStrategy.TENSOR_PARALLEL
    
    def evaluate_strategy_performance(self, metrics: WorkloadMetrics, 
                                    strategy: ParallelismStrategy) -> float:
        """
        Evaluate how well a strategy performs for given metrics.
        Returns performance score (higher is better).
        """
        config = self.strategy_profiles[strategy]
        
        # Weighted scoring based on multiple factors
        latency_score = max(0, 1.0 - (metrics.avg_latency_ms / 200.0))  # Normalize to 200ms max
        throughput_score = min(1.0, metrics.throughput_tokens_per_sec / 2000.0)  # Normalize to 2000 tokens/s
        memory_score = 1.0 - metrics.gpu_memory_util  # Lower memory usage is better
        
        # Different strategies prioritize different aspects
        if strategy == ParallelismStrategy.TENSOR_PARALLEL:
            # Prioritize latency
            score = 0.6 * latency_score + 0.3 * throughput_score + 0.1 * memory_score
        elif strategy == ParallelismStrategy.PIPELINE_PARALLEL:
            # Prioritize memory efficiency
            score = 0.2 * latency_score + 0.3 * throughput_score + 0.5 * memory_score
        elif strategy == ParallelismStrategy.HYBRID:
            # Balanced approach
            score = 0.4 * latency_score + 0.4 * throughput_score + 0.2 * memory_score
        else:  # DATA_PARALLEL
            # Prioritize throughput
            score = 0.1 * latency_score + 0.7 * throughput_score + 0.2 * memory_score
        
        return score
    
    def should_switch_strategy(self, current_metrics: WorkloadMetrics) -> Optional[ParallelismStrategy]:
        """
        Determine if we should switch to a different parallelism strategy.
        Includes cooldown logic to prevent thrashing.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_switch_time < self.switch_cooldown_sec:
            return None
        
        # Get recommendation from decision function
        recommended = self.choose_worker_pool(
            current_metrics.seq_len,
            current_metrics.gpu_memory_util,
            current_metrics.concurrent_requests
        )
        
        # If already using recommended strategy, no switch needed
        if recommended == self.current_strategy:
            return None
        
        # Evaluate if switch would provide significant benefit
        current_score = self.evaluate_strategy_performance(current_metrics, self.current_strategy)
        recommended_score = self.evaluate_strategy_performance(current_metrics, recommended)
        
        # Only switch if significant improvement (10% threshold)
        if recommended_score > current_score * 1.1:
            logger.info(f"Strategy switch recommended: {self.current_strategy.value} -> {recommended.value}")
            logger.info(f"Score improvement: {current_score:.3f} -> {recommended_score:.3f}")
            return recommended
        
        return None
    
    def switch_strategy(self, new_strategy: ParallelismStrategy):
        """Execute strategy switch with proper coordination."""
        logger.info(f"Switching from {self.current_strategy.value} to {new_strategy.value}")
        
        # Record switch time
        self.last_switch_time = time.time()
        
        # Update current strategy
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        
        # Log configuration details
        config = self.strategy_profiles[new_strategy]
        logger.info(f"New configuration: TP={config.tensor_parallel_size}, "
                   f"PP={config.pipeline_parallel_size}, DP={config.data_parallel_size}")
        
        # In a real implementation, this would trigger:
        # 1. Graceful shutdown of current model instances
        # 2. Reinitialization with new parallelism configuration
        # 3. Load balancer updates to route to new instances
        
        return config


class SystemMetricsCollector:
    """Collects runtime metrics for parallelism decisions."""
    
    def __init__(self):
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background metrics collection."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop metrics collection."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            metrics = self.collect_current_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only recent history
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            time.sleep(1.0)  # Collect metrics every second
    
    def collect_current_metrics(self) -> WorkloadMetrics:
        """Collect current system metrics."""
        # Simulate metrics collection
        # In practice, these would come from:
        # - CUDA metrics (nvidia-ml-py)
        # - Application metrics (request queue lengths, latencies)
        # - System metrics (psutil, /proc/stat)
        
        import random
        
        # Simulate varying workload patterns
        base_seq_len = random.choice([128, 512, 1024, 2048, 4096])
        load_factor = random.uniform(0.3, 0.9)
        
        return WorkloadMetrics(
            seq_len=base_seq_len + random.randint(-50, 50),
            batch_size=random.randint(4, 32),
            gpu_memory_util=0.4 + load_factor * 0.5,
            concurrent_requests=int(load_factor * 50),
            avg_latency_ms=50 + load_factor * 100,
            throughput_tokens_per_sec=1000 * (2.0 - load_factor),
            memory_bandwidth_util=0.6 + load_factor * 0.3,
            compute_utilization=0.5 + load_factor * 0.4
        )
    
    def get_recent_metrics(self, window_size: int = 10) -> List[WorkloadMetrics]:
        """Get recent metrics for analysis."""
        return self.metrics_history[-window_size:]


class DynamicInferenceServer:
    """
    Simulation of an inference server with dynamic parallelism adaptation.
    Demonstrates Chapter 19's adaptive parallelism in practice.
    """
    
    def __init__(self, available_gpus: int = 8):
        self.router = DynamicParallelismRouter(available_gpus)
        self.metrics_collector = SystemMetricsCollector()
        self.request_queue = []
        self.running = False
        
        # Performance tracking
        self.processed_requests = 0
        self.total_latency = 0.0
        self.strategy_switches = 0
    
    def start(self):
        """Start the inference server with adaptive parallelism."""
        logger.info("Starting dynamic inference server...")
        self.running = True
        self.metrics_collector.start_monitoring()
        
        # Start main processing loop
        self._processing_loop()
    
    def stop(self):
        """Stop the server."""
        logger.info("Stopping dynamic inference server...")
        self.running = False
        self.metrics_collector.stop_monitoring()
    
    def _processing_loop(self):
        """Main request processing loop with adaptive parallelism."""
        while self.running:
            try:
                # Get current metrics
                current_metrics = self.metrics_collector.collect_current_metrics()
                
                # Check if we should switch strategy
                new_strategy = self.router.should_switch_strategy(current_metrics)
                if new_strategy:
                    config = self.router.switch_strategy(new_strategy)
                    self.strategy_switches += 1
                
                # Process requests (simulate)
                self._process_requests(current_metrics)
                
                # Brief pause
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def _process_requests(self, metrics: WorkloadMetrics):
        """Simulate request processing with current strategy."""
        # Simulate processing multiple requests
        for _ in range(min(5, metrics.concurrent_requests)):
            start_time = time.time()
            
            # Simulate inference work
            self._simulate_inference(metrics)
            
            # Track performance
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.total_latency += latency
            self.processed_requests += 1
            
            # Log occasionally
            if self.processed_requests % 100 == 0:
                avg_latency = self.total_latency / self.processed_requests
                logger.info(f"Processed {self.processed_requests} requests, "
                           f"avg latency: {avg_latency:.1f}ms, "
                           f"strategy: {self.router.current_strategy.value}, "
                           f"switches: {self.strategy_switches}")
    
    def _simulate_inference(self, metrics: WorkloadMetrics):
        """Simulate inference work based on current strategy and metrics."""
        config = self.router.strategy_profiles[self.router.current_strategy]
        
        # Simulate processing time based on strategy
        base_time = config.estimated_latency_ms / 1000.0
        
        # Adjust for sequence length
        seq_factor = metrics.seq_len / 1024.0
        
        # Adjust for memory pressure
        memory_factor = 1.0 + metrics.gpu_memory_util * 0.5
        
        processing_time = base_time * seq_factor * memory_factor
        time.sleep(processing_time)


def simulate_workload_patterns():
    """
    Simulate different workload patterns to test adaptive parallelism.
    Demonstrates various scenarios from Chapter 19.
    """
    print("=== Workload Pattern Simulation ===")
    
    router = DynamicParallelismRouter(available_gpus=8)
    
    # Test different workload scenarios
    scenarios = [
        {
            "name": "Latency-sensitive short requests",
            "seq_len": 128,
            "memory_util": 0.3,
            "concurrent_reqs": 10,
            "expected_strategy": ParallelismStrategy.TENSOR_PARALLEL
        },
        {
            "name": "Long context documents",
            "seq_len": 4096,
            "memory_util": 0.7,
            "concurrent_reqs": 5,
            "expected_strategy": ParallelismStrategy.PIPELINE_PARALLEL
        },
        {
            "name": "High throughput batch processing",
            "seq_len": 256,
            "memory_util": 0.4,
            "concurrent_reqs": 50,
            "expected_strategy": ParallelismStrategy.DATA_PARALLEL
        },
        {
            "name": "Mixed workload under memory pressure",
            "seq_len": 1024,
            "memory_util": 0.85,
            "concurrent_reqs": 20,
            "expected_strategy": ParallelismStrategy.HYBRID
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        chosen_strategy = router.choose_worker_pool(
            scenario['seq_len'],
            scenario['memory_util'],
            scenario['concurrent_reqs']
        )
        
        config = router.strategy_profiles[chosen_strategy]
        
        print(f"Sequence length: {scenario['seq_len']}")
        print(f"Memory utilization: {scenario['memory_util']:.1%}")
        print(f"Concurrent requests: {scenario['concurrent_reqs']}")
        print(f"Chosen strategy: {chosen_strategy.value}")
        print(f"Configuration: TP={config.tensor_parallel_size}, "
              f"PP={config.pipeline_parallel_size}, DP={config.data_parallel_size}")
        
        expected = scenario['expected_strategy']
        status = "✓ CORRECT" if chosen_strategy == expected else f"✗ Expected {expected.value}"
        print(f"Result: {status}")


def benchmark_strategy_switching():
    """Benchmark the overhead of strategy switching decisions."""
    print("\n=== Strategy Switching Benchmark ===")
    
    router = DynamicParallelismRouter(available_gpus=8)
    metrics_collector = SystemMetricsCollector()
    
    # Benchmark decision making
    num_decisions = 10000
    start_time = time.time()
    
    for _ in range(num_decisions):
        metrics = metrics_collector.collect_current_metrics()
        strategy = router.choose_worker_pool(
            metrics.seq_len,
            metrics.gpu_memory_util,
            metrics.concurrent_requests
        )
    
    decision_time = time.time() - start_time
    
    print(f"Made {num_decisions} routing decisions in {decision_time:.3f} seconds")
    print(f"Average decision time: {decision_time/num_decisions*1000:.3f} ms")
    print(f"Decisions per second: {num_decisions/decision_time:.0f}")
    
    # Test strategy evaluation
    start_time = time.time()
    for _ in range(1000):
        metrics = metrics_collector.collect_current_metrics()
        for strategy in ParallelismStrategy:
            score = router.evaluate_strategy_performance(metrics, strategy)
    
    eval_time = time.time() - start_time
    print(f"Strategy evaluation time: {eval_time/4000*1000:.3f} ms per evaluation")


def main():
    """Main demonstration of dynamic parallelism concepts."""
    print("Chapter 19: Dynamic Parallelism for Adaptive Inference")
    print("=" * 55)
    
    # Simulate different workload patterns
    simulate_workload_patterns()
    
    # Benchmark decision making performance
    benchmark_strategy_switching()
    
    # Interactive demo
    print(f"\n=== Interactive Dynamic Inference Server ===")
    print("Starting server with adaptive parallelism...")
    print("Press Ctrl+C to stop")
    
    try:
        server = DynamicInferenceServer(available_gpus=8)
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()
    
    print(f"\n=== Key Dynamic Parallelism Benefits ===")
    print("- Adapts to changing workload characteristics in real-time")
    print("- Optimizes for different objectives (latency vs throughput)")
    print("- Maximizes GPU utilization across different scenarios")
    print("- Prevents resource waste from fixed parallelism strategies")
    print("- Includes cooldown mechanisms to prevent strategy thrashing")


if __name__ == "__main__":
    main()
