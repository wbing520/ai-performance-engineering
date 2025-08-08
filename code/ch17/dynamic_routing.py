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
dynamic_routing.py
Chapter 17: Dynamic Routing for Disaggregated Prefill-Decode

Implementation of dynamic routing algorithms that decide whether to offload
prefill computation to dedicated prefill workers or handle it locally on 
decode workers.

Based on Chapter 17 content about disaggregated inference systems.
"""

import time
import random
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import yaml


class Priority(Enum):
    LOW = "low"
    STANDARD = "standard" 
    HIGH = "high"


@dataclass
class Request:
    id: str
    prompt_tokens: List[int]
    priority: Priority
    timestamp: float
    prefix_cached_length: int = 0
    expected_output_length: int = 50


@dataclass
class WorkerMetrics:
    """Metrics for a prefill or decode worker."""
    queue_length: int
    gpu_utilization: float
    memory_usage: float
    kv_cache_usage: float
    active_requests: int
    last_updated: float


class DisaggregatedRouter:
    """
    Router that implements dynamic routing policies for disaggregated inference.
    Based on Chapter 17's routing algorithms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Configuration parameters from Chapter 17
        self.PREFILL_LENGTH_THRESHOLD = 100  # tokens
        self.PREFILL_QUEUE_MAX = 10  # max queue length
        self.TTFT_SLO_MAX = 500  # milliseconds
        
        # Latency cost weights (from Chapter 17 example)
        self.occupancy_weight = 0.7
        self.active_req_weight = 0.3
        
        # Worker pools
        self.prefill_workers: Dict[str, WorkerMetrics] = {}
        self.decode_workers: Dict[str, WorkerMetrics] = {}
        
        # Metrics tracking
        self.avg_prefill_time_per_req = 50.0  # ms
        self.avg_decode_time_per_req = 10.0   # ms
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load routing configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Update parameters from config
            split_policy = config.get('split_policy', {})
            self.PREFILL_LENGTH_THRESHOLD = split_policy.get('prompt_length_threshold', 100)
            
            print(f"Loaded configuration from {config_path}")
            print(f"Prefill threshold: {self.PREFILL_LENGTH_THRESHOLD} tokens")
            
        except Exception as e:
            print(f"Failed to load config: {e}, using defaults")
    
    def should_offload_prefill(self, prompt_length: int, prefix_cached_length: int, 
                             prefill_queue_size: int) -> bool:
        """
        Core routing decision from Chapter 17.
        
        Decides whether to offload prefill to dedicated prefill workers
        or handle it locally on the decode worker.
        """
        # Condition 1: Long effective prefill
        # (prompt minus cached part exceeds threshold)
        long_prefill = (prompt_length - prefix_cached_length) > self.PREFILL_LENGTH_THRESHOLD
        
        # Condition 2: Prefill workers have capacity
        # (queue not too long)
        prefill_available = prefill_queue_size < self.PREFILL_QUEUE_MAX
        
        if long_prefill and prefill_available:
            # offload to prefill worker
            return True
        else:
            # do prefill locally (async)
            return False
    
    def admit_request(self, request: Request) -> bool:
        """
        Early rejection based on estimated latency and priority.
        Implementation from Chapter 17.
        """
        # Estimate current TTFT when new request is added
        est_ttft = (self.get_current_prefill_queue_length() * 
                   self.avg_prefill_time_per_req)
        
        # Consider decode backlog as well
        est_ttft += (self.get_current_decode_queue_length() * 
                    self.avg_decode_time_per_req)
        
        if est_ttft > self.TTFT_SLO_MAX:
            if request.priority == Priority.LOW:
                # reject low priority request
                print(f"REJECTING low priority request {request.id} "
                      f"(estimated TTFT: {est_ttft:.1f}ms > {self.TTFT_SLO_MAX}ms)")
                return False
            else:
                # high priority: admit high priority request
                print(f"ADMITTING high priority request {request.id} "
                      f"despite high load (estimated TTFT: {est_ttft:.1f}ms)")
                return True
        else:
            return True
    
    def calculate_latency_cost(self, worker_metrics: WorkerMetrics) -> float:
        """
        Calculate latency cost for a worker based on Chapter 17 formula.
        Lower cost is preferable.
        """
        occupancy_percent = worker_metrics.memory_usage / 100.0
        active_req_count = worker_metrics.active_requests
        
        latency_cost = (self.occupancy_weight * occupancy_percent + 
                       self.active_req_weight * active_req_count)
        
        return latency_cost
    
    def select_best_worker(self, workers: Dict[str, WorkerMetrics]) -> str:
        """Select worker with lowest latency cost."""
        if not workers:
            return None
        
        best_worker = None
        best_cost = float('inf')
        
        for worker_id, metrics in workers.items():
            cost = self.calculate_latency_cost(metrics)
            if cost < best_cost:
                best_cost = cost
                best_worker = worker_id
        
        return best_worker
    
    def route_request(self, request: Request) -> Tuple[str, str]:
        """
        Main routing function that decides how to handle a request.
        Returns (worker_type, worker_id) or (None, None) if rejected.
        """
        # Step 1: Admission control
        if not self.admit_request(request):
            return None, None
        
        prompt_length = len(request.prompt_tokens)
        prefill_queue_size = self.get_current_prefill_queue_length()
        
        # Step 2: Routing decision
        should_offload = self.should_offload_prefill(
            prompt_length, request.prefix_cached_length, prefill_queue_size
        )
        
        if should_offload:
            # Route to prefill worker
            worker_id = self.select_best_worker(self.prefill_workers)
            if worker_id:
                print(f"ROUTING request {request.id} to prefill worker {worker_id} "
                      f"(prompt: {prompt_length} tokens, effective: {prompt_length - request.prefix_cached_length})")
                return "prefill", worker_id
            else:
                print(f"No prefill workers available, falling back to decode worker")
        
        # Route to decode worker (either by choice or fallback)
        worker_id = self.select_best_worker(self.decode_workers)
        if worker_id:
            action = "local prefill" if not should_offload else "fallback"
            print(f"ROUTING request {request.id} to decode worker {worker_id} ({action})")
            return "decode", worker_id
        
        print(f"No workers available for request {request.id}")
        return None, None
    
    def get_current_prefill_queue_length(self) -> int:
        """Get total queue length across all prefill workers."""
        return sum(worker.queue_length for worker in self.prefill_workers.values())
    
    def get_current_decode_queue_length(self) -> int:
        """Get total queue length across all decode workers."""
        return sum(worker.queue_length for worker in self.decode_workers.values())
    
    def update_worker_metrics(self, worker_type: str, worker_id: str, 
                            metrics: WorkerMetrics):
        """Update metrics for a specific worker."""
        if worker_type == "prefill":
            self.prefill_workers[worker_id] = metrics
        elif worker_type == "decode":
            self.decode_workers[worker_id] = metrics
    
    def simulate_cluster_state(self):
        """Simulate a cluster with varying load patterns."""
        # Initialize some workers
        for i in range(2):
            self.prefill_workers[f"prefill-{i}"] = WorkerMetrics(
                queue_length=random.randint(0, 5),
                gpu_utilization=random.uniform(20, 80),
                memory_usage=random.uniform(30, 90),
                kv_cache_usage=random.uniform(40, 80),
                active_requests=random.randint(1, 8),
                last_updated=time.time()
            )
        
        for i in range(4):
            self.decode_workers[f"decode-{i}"] = WorkerMetrics(
                queue_length=random.randint(0, 3),
                gpu_utilization=random.uniform(30, 70),
                memory_usage=random.uniform(40, 85),
                kv_cache_usage=random.uniform(50, 90),
                active_requests=random.randint(2, 12),
                last_updated=time.time()
            )


def create_dynamo_config():
    """Create a sample Dynamo configuration from Chapter 17."""
    config = {
        "model": "llama-70b",
        "split_policy": {
            "prompt_length_threshold": 256,
            "prefix_cache_weight": 10.0,
            "queue_length_weight": 1.5,
            "decode_load_weight": 0.5,
            "enable_hotspot_prevention": True
        },
        "cache": {
            "reuse_prefix": True,
            "min_cache_hit_ratio": 0.75
        },
        "autoscale": {
            "prefill": {
                "min_replicas": 4,
                "max_replicas": 12,
                "scale_up": {"queue_length": 8, "gpu_utilization": 80},
                "scale_down": {"queue_length": 2, "gpu_utilization": 40}
            },
            "decode": {
                "min_replicas": 8,
                "max_replicas": 24,
                "scale_up": {"queue_length": 16, "kv_cache_usage": 75},
                "scale_down": {"queue_length": 4, "kv_cache_usage": 30}
            }
        },
        "qos": {
            "enable_early_rejection": True,
            "low_priority_threshold_ms": 500,
            "reject_on_slo_violation": True
        }
    }
    
    # Save configuration
    with open('dynamo_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Created dynamo_config.yaml with sample configuration")
    return config


def simulate_request_stream():
    """Simulate a stream of requests with different characteristics."""
    requests = []
    
    # Different types of requests
    request_types = [
        # (prompt_length, priority, prefix_cached, description)
        (50, Priority.HIGH, 0, "short_urgent"),
        (200, Priority.STANDARD, 100, "medium_cached"),
        (500, Priority.LOW, 0, "long_batch"),
        (300, Priority.HIGH, 200, "medium_urgent_cached"),
        (800, Priority.LOW, 0, "very_long_batch"),
        (80, Priority.STANDARD, 50, "short_cached"),
    ]
    
    for i in range(20):
        prompt_len, priority, cached, desc = random.choice(request_types)
        
        # Add some randomness
        prompt_len += random.randint(-20, 20)
        cached = min(cached, prompt_len - 10)  # cached can't exceed prompt
        
        request = Request(
            id=f"req-{i:03d}-{desc}",
            prompt_tokens=list(range(prompt_len)),  # dummy tokens
            priority=priority,
            timestamp=time.time() + i * 0.1,  # stagger requests
            prefix_cached_length=max(0, cached),
            expected_output_length=random.randint(20, 100)
        )
        requests.append(request)
    
    return requests


def main():
    """Demonstrate dynamic routing for disaggregated inference."""
    print("Chapter 17: Dynamic Routing for Disaggregated Prefill-Decode")
    print("=" * 60)
    
    # Create sample configuration
    config = create_dynamo_config()
    
    # Initialize router with configuration
    router = DisaggregatedRouter('dynamo_config.yaml')
    
    # Setup simulated cluster
    router.simulate_cluster_state()
    
    print("\n=== Initial Cluster State ===")
    print("Prefill Workers:")
    for worker_id, metrics in router.prefill_workers.items():
        cost = router.calculate_latency_cost(metrics)
        print(f"  {worker_id}: queue={metrics.queue_length}, "
              f"mem={metrics.memory_usage:.1f}%, active={metrics.active_requests}, "
              f"cost={cost:.3f}")
    
    print("\nDecode Workers:")
    for worker_id, metrics in router.decode_workers.items():
        cost = router.calculate_latency_cost(metrics)
        print(f"  {worker_id}: queue={metrics.queue_length}, "
              f"mem={metrics.memory_usage:.1f}%, active={metrics.active_requests}, "
              f"cost={cost:.3f}")
    
    # Generate request stream
    requests = simulate_request_stream()
    
    print(f"\n=== Processing {len(requests)} Requests ===")
    
    routing_stats = {
        "prefill_routed": 0,
        "decode_routed": 0, 
        "rejected": 0,
        "by_priority": {Priority.LOW: 0, Priority.STANDARD: 0, Priority.HIGH: 0}
    }
    
    for request in requests:
        print(f"\n--- Request {request.id} ---")
        print(f"Prompt: {len(request.prompt_tokens)} tokens, "
              f"Cached: {request.prefix_cached_length}, "
              f"Priority: {request.priority.value}")
        
        worker_type, worker_id = router.route_request(request)
        
        if worker_type is None:
            routing_stats["rejected"] += 1
        elif worker_type == "prefill":
            routing_stats["prefill_routed"] += 1
        else:
            routing_stats["decode_routed"] += 1
        
        routing_stats["by_priority"][request.priority] += 1
        
        # Simulate some processing time and update metrics
        time.sleep(0.05)
        
        # Occasionally update worker metrics to simulate changing load
        if random.random() < 0.3:
            worker_to_update = random.choice(list(router.prefill_workers.keys()) + 
                                           list(router.decode_workers.keys()))
            if worker_to_update.startswith("prefill"):
                current = router.prefill_workers[worker_to_update]
                current.queue_length = max(0, current.queue_length + random.randint(-1, 2))
                current.gpu_utilization += random.uniform(-10, 10)
                current.gpu_utilization = max(0, min(100, current.gpu_utilization))
            else:
                current = router.decode_workers[worker_to_update]
                current.queue_length = max(0, current.queue_length + random.randint(-1, 2))
                current.memory_usage += random.uniform(-5, 5)
                current.memory_usage = max(0, min(100, current.memory_usage))
    
    # Print routing statistics
    print(f"\n=== Routing Statistics ===")
    print(f"Total requests: {len(requests)}")
    print(f"Routed to prefill workers: {routing_stats['prefill_routed']}")
    print(f"Routed to decode workers: {routing_stats['decode_routed']}")
    print(f"Rejected: {routing_stats['rejected']}")
    print(f"Acceptance rate: {(len(requests) - routing_stats['rejected']) / len(requests) * 100:.1f}%")
    
    print(f"\nBy priority:")
    for priority in Priority:
        count = routing_stats["by_priority"][priority]
        print(f"  {priority.value}: {count} requests")
    
    # Demonstrate latency cost calculation
    print(f"\n=== Latency Cost Analysis ===")
    print("Worker latency costs (lower is better):")
    
    all_workers = [(f"prefill-{k}", v) for k, v in router.prefill_workers.items()] + \
                  [(f"decode-{k}", v) for k, v in router.decode_workers.items()]
    
    all_workers.sort(key=lambda x: router.calculate_latency_cost(x[1]))
    
    for worker_id, metrics in all_workers[:5]:  # Show top 5
        cost = router.calculate_latency_cost(metrics)
        print(f"  {worker_id}: cost={cost:.3f} "
              f"(mem={metrics.memory_usage:.1f}%, active={metrics.active_requests})")
    
    print(f"\n=== Configuration Summary ===")
    print(f"Prefill threshold: {router.PREFILL_LENGTH_THRESHOLD} tokens")
    print(f"Max prefill queue: {router.PREFILL_QUEUE_MAX}")
    print(f"TTFT SLO: {router.TTFT_SLO_MAX}ms")
    print(f"Latency cost weights: memory={router.occupancy_weight}, active={router.active_req_weight}")


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
