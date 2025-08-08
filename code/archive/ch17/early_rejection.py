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
Early rejection policy for QoS in disaggregated inference.
This demonstrates admission control to maintain SLOs under load.
"""

import time
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

class RequestPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Request:
    """Inference request with metadata"""
    request_id: str
    prompt_length: int
    priority: RequestPriority
    timestamp: float
    estimated_ttft: float = 0.0

@dataclass
class SystemMetrics:
    """Current system metrics"""
    prefill_queue_length: int
    decode_queue_length: int
    avg_prefill_time_per_req: float  # ms
    avg_decode_time_per_req: float   # ms
    gpu_utilization: float  # percentage
    kv_cache_usage: float   # percentage

class EarlyRejectionPolicy:
    """Early rejection policy for maintaining SLOs"""
    
    def __init__(self):
        # SLO thresholds
        self.TTFT_SLO_MAX = 200.0  # ms
        self.TPOT_SLO_MAX = 50.0   # ms
        
        # Priority thresholds
        self.LOW_PRIORITY_THRESHOLD = 150.0  # ms
        self.MEDIUM_PRIORITY_THRESHOLD = 180.0  # ms
        self.HIGH_PRIORITY_THRESHOLD = 200.0  # ms
        
        # System state
        self.prefill_queue = deque()
        self.decode_queue = deque()
        self.active_requests = {}
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=100)
        self.rejection_stats = {
            "total_requests": 0,
            "accepted_requests": 0,
            "rejected_requests": 0,
            "rejection_by_priority": {p.value: 0 for p in RequestPriority},
        }
    
    def admit_request(self, request: Request, current_metrics: SystemMetrics) -> bool:
        """Determine if a request should be admitted"""
        
        self.rejection_stats["total_requests"] += 1
        
        # Estimate current TTFT when new request is added
        estimated_ttft = self._estimate_ttft(request, current_metrics)
        request.estimated_ttft = estimated_ttft
        
        # Get priority threshold
        priority_threshold = self._get_priority_threshold(request.priority)
        
        # Check if request should be rejected
        should_reject = estimated_ttft > priority_threshold
        
        if should_reject:
            self.rejection_stats["rejected_requests"] += 1
            self.rejection_stats["rejection_by_priority"][request.priority.value] += 1
            return False
        else:
            self.rejection_stats["accepted_requests"] += 1
            # Add to appropriate queue
            if request.prompt_length > 256:  # Long prompt
                self.prefill_queue.append(request)
            else:
                self.decode_queue.append(request)
            return True
    
    def _estimate_ttft(self, request: Request, metrics: SystemMetrics) -> float:
        """Estimate TTFT for the new request"""
        
        # Base prefill time based on prompt length
        base_prefill_time = request.prompt_length * metrics.avg_prefill_time_per_req / 1000.0
        
        # Queue delay for prefill
        prefill_queue_delay = len(self.prefill_queue) * metrics.avg_prefill_time_per_req
        
        # Queue delay for decode
        decode_queue_delay = len(self.decode_queue) * metrics.avg_decode_time_per_req
        
        # System load factor
        load_factor = max(1.0, metrics.gpu_utilization / 80.0)
        
        # Calculate total estimated TTFT
        estimated_ttft = (base_prefill_time + prefill_queue_delay + decode_queue_delay) * load_factor
        
        return estimated_ttft
    
    def _get_priority_threshold(self, priority: RequestPriority) -> float:
        """Get TTFT threshold for priority level"""
        thresholds = {
            RequestPriority.LOW: self.LOW_PRIORITY_THRESHOLD,
            RequestPriority.MEDIUM: self.MEDIUM_PRIORITY_THRESHOLD,
            RequestPriority.HIGH: self.HIGH_PRIORITY_THRESHOLD,
        }
        return thresholds.get(priority, self.TTFT_SLO_MAX)
    
    def update_metrics(self, metrics: SystemMetrics):
        """Update system metrics"""
        self.metrics_history.append(metrics)
    
    def get_rejection_stats(self) -> Dict:
        """Get rejection statistics"""
        stats = self.rejection_stats.copy()
        if stats["total_requests"] > 0:
            stats["rejection_rate"] = stats["rejected_requests"] / stats["total_requests"]
            stats["acceptance_rate"] = stats["accepted_requests"] / stats["total_requests"]
        return stats
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            "prefill_queue_length": len(self.prefill_queue),
            "decode_queue_length": len(self.decode_queue),
            "active_requests": len(self.active_requests),
        }

def generate_test_requests() -> List[Request]:
    """Generate test requests with varying characteristics"""
    requests = []
    
    # Generate requests with different priorities and lengths
    for i in range(20):
        prompt_length = random.randint(50, 2000)
        priority = random.choice(list(RequestPriority))
        
        request = Request(
            request_id=f"req_{i:03d}",
            prompt_length=prompt_length,
            priority=priority,
            timestamp=time.time() + i * 0.1,  # Simulate arrival times
        )
        requests.append(request)
    
    return requests

def simulate_system_load() -> SystemMetrics:
    """Simulate varying system load"""
    # Simulate different load conditions
    load_scenario = random.choice(["low", "medium", "high", "overloaded"])
    
    if load_scenario == "low":
        return SystemMetrics(
            prefill_queue_length=random.randint(0, 2),
            decode_queue_length=random.randint(0, 3),
            avg_prefill_time_per_req=random.uniform(20.0, 40.0),
            avg_decode_time_per_req=random.uniform(30.0, 50.0),
            gpu_utilization=random.uniform(30.0, 60.0),
            kv_cache_usage=random.uniform(20.0, 50.0),
        )
    elif load_scenario == "medium":
        return SystemMetrics(
            prefill_queue_length=random.randint(2, 5),
            decode_queue_length=random.randint(3, 8),
            avg_prefill_time_per_req=random.uniform(40.0, 80.0),
            avg_decode_time_per_req=random.uniform(50.0, 90.0),
            gpu_utilization=random.uniform(60.0, 85.0),
            kv_cache_usage=random.uniform(50.0, 75.0),
        )
    elif load_scenario == "high":
        return SystemMetrics(
            prefill_queue_length=random.randint(5, 10),
            decode_queue_length=random.randint(8, 15),
            avg_prefill_time_per_req=random.uniform(80.0, 150.0),
            avg_decode_time_per_req=random.uniform(90.0, 180.0),
            gpu_utilization=random.uniform(85.0, 95.0),
            kv_cache_usage=random.uniform(75.0, 90.0),
        )
    else:  # overloaded
        return SystemMetrics(
            prefill_queue_length=random.randint(10, 20),
            decode_queue_length=random.randint(15, 25),
            avg_prefill_time_per_req=random.uniform(150.0, 300.0),
            avg_decode_time_per_req=random.uniform(180.0, 350.0),
            gpu_utilization=random.uniform(95.0, 100.0),
            kv_cache_usage=random.uniform(90.0, 100.0),
        )

def main():
    """Main demonstration function"""
    print("Early Rejection Policy for QoS")
    print("==============================")
    
    # Initialize policy
    policy = EarlyRejectionPolicy()
    
    # Generate test requests
    requests = generate_test_requests()
    
    print(f"\nProcessing {len(requests)} requests...")
    print("\nRequest Admission Decisions:")
    print("-" * 80)
    
    for i, request in enumerate(requests):
        # Simulate current system metrics
        current_metrics = simulate_system_load()
        policy.update_metrics(current_metrics)
        
        # Make admission decision
        admitted = policy.admit_request(request, current_metrics)
        
        # Print decision
        status = "ACCEPTED" if admitted else "REJECTED"
        print(f"Request {i+1:2d}: {request.request_id}")
        print(f"  Prompt Length: {request.prompt_length:4d} tokens")
        print(f"  Priority: {request.priority.value}")
        print(f"  Estimated TTFT: {request.estimated_ttft:.1f} ms")
        print(f"  Status: {status}")
        print(f"  System Load: GPU={current_metrics.gpu_utilization:.1f}%, KV={current_metrics.kv_cache_usage:.1f}%")
        print(f"  Queues: Prefill={current_metrics.prefill_queue_length}, Decode={current_metrics.decode_queue_length}")
        print()
    
    # Print final statistics
    stats = policy.get_rejection_stats()
    print("Rejection Statistics:")
    print("-" * 40)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Accepted Requests: {stats['accepted_requests']}")
    print(f"Rejected Requests: {stats['rejected_requests']}")
    print(f"Acceptance Rate: {stats.get('acceptance_rate', 0):.1%}")
    print(f"Rejection Rate: {stats.get('rejection_rate', 0):.1%}")
    
    print("\nRejections by Priority:")
    for priority, count in stats["rejection_by_priority"].items():
        if count > 0:
            print(f"  {priority}: {count} rejections")
    
    # Print queue status
    queue_status = policy.get_queue_status()
    print(f"\nFinal Queue Status:")
    print(f"  Prefill Queue: {queue_status['prefill_queue_length']} requests")
    print(f"  Decode Queue: {queue_status['decode_queue_length']} requests")
    print(f"  Active Requests: {queue_status['active_requests']} requests")
    
    print(f"\nSLO Thresholds:")
    print(f"  Low Priority: {policy.LOW_PRIORITY_THRESHOLD} ms")
    print(f"  Medium Priority: {policy.MEDIUM_PRIORITY_THRESHOLD} ms")
    print(f"  High Priority: {policy.HIGH_PRIORITY_THRESHOLD} ms")

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
