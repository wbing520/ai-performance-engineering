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
early_rejection.py
Chapter 17: Early Rejection and Quality of Service for Disaggregated Inference

Implementation of early rejection (admission control) policies that prevent
system overload by rejecting requests that cannot meet SLA requirements.

Based on Chapter 17's QoS mechanisms for ultra-scale inference systems.
"""

import time
import random
import statistics
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading


class Priority(Enum):
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class Request:
    id: str
    prompt_length: int
    expected_output_length: int
    priority: Priority
    arrival_time: float
    deadline: Optional[float] = None
    
    def __post_init__(self):
        # Set deadline based on priority
        if self.deadline is None:
            if self.priority == Priority.PREMIUM:
                self.deadline = self.arrival_time + 0.2  # 200ms for premium
            elif self.priority == Priority.STANDARD:
                self.deadline = self.arrival_time + 0.5  # 500ms for standard
            else:
                self.deadline = self.arrival_time + 1.0  # 1000ms for free


@dataclass
class SystemMetrics:
    """Real-time system metrics for admission control."""
    prefill_queue_length: int = 0
    decode_queue_length: int = 0
    avg_prefill_time_per_req: float = 50.0  # ms
    avg_decode_time_per_req: float = 10.0   # ms
    current_load: float = 0.0  # 0-1
    recent_ttft_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_tpot_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_updated: float = field(default_factory=time.time)


class QoSController:
    """
    Quality of Service controller implementing early rejection policies.
    Based on Chapter 17's admission control examples.
    """
    
    def __init__(self):
        # SLO thresholds (milliseconds)
        self.TTFT_SLO_MAX = {
            Priority.PREMIUM: 200,   # 200ms for premium
            Priority.STANDARD: 500,  # 500ms for standard  
            Priority.FREE: 1000      # 1000ms for free
        }
        
        self.TPOT_SLO_MAX = {
            Priority.PREMIUM: 30,    # 30ms per token
            Priority.STANDARD: 50,   # 50ms per token
            Priority.FREE: 100       # 100ms per token
        }
        
        # Capacity limits
        self.MAX_CONCURRENT_REQUESTS = {
            Priority.PREMIUM: 50,    # Reserve capacity for premium
            Priority.STANDARD: 100,  # Standard tier capacity
            Priority.FREE: 200       # Best effort for free
        }
        
        # Current system state
        self.metrics = SystemMetrics()
        self.active_requests: Dict[Priority, int] = {
            Priority.PREMIUM: 0,
            Priority.STANDARD: 0,
            Priority.FREE: 0
        }
        
        # Request tracking
        self.admitted_requests: List[Request] = []
        self.rejected_requests: List[Request] = []
        
        # Performance tracking
        self.rejection_stats = {
            Priority.PREMIUM: {"total": 0, "rejected": 0},
            Priority.STANDARD: {"total": 0, "rejected": 0},
            Priority.FREE: {"total": 0, "rejected": 0}
        }
        
        self.lock = threading.Lock()
    
    def admit_request(self, request: Request) -> bool:
        """
        Core admission control function from Chapter 17.
        
        Early rejection based on estimated latency and priority.
        """
        with self.lock:
            self.rejection_stats[request.priority]["total"] += 1
            
            # Step 1: Check capacity limits
            if not self._check_capacity_limits(request):
                self._reject_request(request, "capacity_limit")
                return False
            
            # Step 2: Estimate TTFT for this request
            estimated_ttft = self._estimate_ttft(request)
            
            # Step 3: Check SLO compliance
            slo_limit = self.TTFT_SLO_MAX[request.priority]
            
            if estimated_ttft > slo_limit:
                if request.priority == Priority.FREE:
                    # Always reject free tier if SLO would be violated
                    self._reject_request(request, "slo_violation")
                    return False
                elif request.priority == Priority.STANDARD:
                    # Reject standard if load is very high
                    if self.metrics.current_load > 0.8:
                        self._reject_request(request, "high_load")
                        return False
                # Premium requests are rarely rejected
            
            # Step 4: Additional checks for system health
            if not self._system_health_check(request):
                self._reject_request(request, "system_health")
                return False
            
            # Request is admitted
            self._admit_request(request)
            return True
    
    def _check_capacity_limits(self, request: Request) -> bool:
        """Check if we have capacity for this priority level."""
        current_count = self.active_requests[request.priority]
        max_allowed = self.MAX_CONCURRENT_REQUESTS[request.priority]
        
        if current_count >= max_allowed:
            print(f"Capacity limit reached for {request.priority.value}: "
                  f"{current_count}/{max_allowed}")
            return False
        
        return True
    
    def _estimate_ttft(self, request: Request) -> float:
        """
        Estimate Time-To-First-Token based on current system state.
        Implementation from Chapter 17.
        """
        # Base estimation from queue lengths
        est_ttft = (self.metrics.prefill_queue_length * 
                   self.metrics.avg_prefill_time_per_req)
        
        # Consider decode backlog as well
        est_ttft += (self.metrics.decode_queue_length * 
                    self.metrics.avg_decode_time_per_req)
        
        # Adjust for request size
        # Larger prompts take longer
        size_factor = max(1.0, request.prompt_length / 100.0)
        est_ttft *= size_factor
        
        # Adjust for system load
        load_factor = 1.0 + self.metrics.current_load
        est_ttft *= load_factor
        
        # Priority gets better estimates (more accurate prediction)
        if request.priority == Priority.PREMIUM:
            est_ttft *= 0.9  # Premium gets 10% better estimates
        elif request.priority == Priority.FREE:
            est_ttft *= 1.1  # Free tier gets 10% worse estimates
        
        return est_ttft
    
    def _system_health_check(self, request: Request) -> bool:
        """Additional system health checks."""
        # Check recent performance
        if len(self.metrics.recent_ttft_samples) > 10:
            recent_p95_ttft = statistics.quantiles(
                list(self.metrics.recent_ttft_samples), n=20
            )[18]  # 95th percentile
            
            # If recent performance is bad, be more conservative
            if recent_p95_ttft > self.TTFT_SLO_MAX[Priority.STANDARD] * 1.5:
                if request.priority == Priority.FREE:
                    return False
        
        # Check memory pressure (simulated)
        memory_pressure = random.uniform(0, 1)
        if memory_pressure > 0.9:
            if request.priority != Priority.PREMIUM:
                print(f"High memory pressure, rejecting {request.priority.value} request")
                return False
        
        return True
    
    def _admit_request(self, request: Request):
        """Admit the request and update tracking."""
        self.active_requests[request.priority] += 1
        self.admitted_requests.append(request)
        
        print(f"ADMITTED {request.id} ({request.priority.value}) - "
              f"estimated TTFT: {self._estimate_ttft(request):.1f}ms")
    
    def _reject_request(self, request: Request, reason: str):
        """Reject the request and update statistics."""
        self.rejection_stats[request.priority]["rejected"] += 1
        self.rejected_requests.append(request)
        
        estimated_ttft = self._estimate_ttft(request)
        slo_limit = self.TTFT_SLO_MAX[request.priority]
        
        print(f"REJECTED {request.id} ({request.priority.value}) - "
              f"reason: {reason}, estimated TTFT: {estimated_ttft:.1f}ms "
              f"(limit: {slo_limit}ms)")
    
    def complete_request(self, request: Request, actual_ttft: float, actual_tpot: float):
        """Mark request as completed and update metrics."""
        with self.lock:
            self.active_requests[request.priority] -= 1
            
            # Update performance metrics
            self.metrics.recent_ttft_samples.append(actual_ttft)
            self.metrics.recent_tpot_samples.append(actual_tpot)
            
            # Update exponential moving averages
            alpha = 0.1
            self.metrics.avg_prefill_time_per_req = (
                alpha * actual_ttft + 
                (1 - alpha) * self.metrics.avg_prefill_time_per_req
            )
            
            self.metrics.avg_decode_time_per_req = (
                alpha * actual_tpot + 
                (1 - alpha) * self.metrics.avg_decode_time_per_req
            )
    
    def update_system_metrics(self, prefill_queue: int, decode_queue: int, load: float):
        """Update system metrics for admission control."""
        with self.lock:
            self.metrics.prefill_queue_length = prefill_queue
            self.metrics.decode_queue_length = decode_queue
            self.metrics.current_load = load
            self.metrics.last_updated = time.time()
    
    def get_rejection_rate(self, priority: Priority) -> float:
        """Get rejection rate for a priority level."""
        stats = self.rejection_stats[priority]
        if stats["total"] == 0:
            return 0.0
        return stats["rejected"] / stats["total"]
    
    def print_stats(self):
        """Print current QoS statistics."""
        print("\n=== QoS Statistics ===")
        
        total_requests = sum(stats["total"] for stats in self.rejection_stats.values())
        total_rejected = sum(stats["rejected"] for stats in self.rejection_stats.values())
        
        print(f"Total requests: {total_requests}")
        print(f"Total rejected: {total_rejected}")
        print(f"Overall rejection rate: {total_rejected/total_requests*100:.1f}%")
        
        print(f"\nBy priority:")
        for priority in Priority:
            stats = self.rejection_stats[priority]
            rate = self.get_rejection_rate(priority)
            active = self.active_requests[priority]
            capacity = self.MAX_CONCURRENT_REQUESTS[priority]
            
            print(f"  {priority.value:8}: {stats['rejected']:3}/{stats['total']:3} rejected "
                  f"({rate*100:5.1f}%), active: {active:3}/{capacity:3}")
        
        print(f"\nCurrent system state:")
        print(f"  Prefill queue: {self.metrics.prefill_queue_length}")
        print(f"  Decode queue: {self.metrics.decode_queue_length}")
        print(f"  System load: {self.metrics.current_load:.1f}")
        print(f"  Avg TTFT: {self.metrics.avg_prefill_time_per_req:.1f}ms")
        print(f"  Avg TPOT: {self.metrics.avg_decode_time_per_req:.1f}ms")


def simulate_load_spike():
    """Simulate a realistic load spike scenario."""
    qos = QoSController()
    
    print("Chapter 17: Early Rejection and Quality of Service")
    print("=" * 50)
    
    # Simulate different traffic patterns
    scenarios = [
        {
            "name": "Normal Load",
            "duration": 30,
            "request_rate": 2.0,  # requests per second
            "load_factor": 0.3,
            "priority_distribution": {Priority.PREMIUM: 0.1, Priority.STANDARD: 0.6, Priority.FREE: 0.3}
        },
        {
            "name": "Traffic Spike",
            "duration": 20, 
            "request_rate": 8.0,
            "load_factor": 0.8,
            "priority_distribution": {Priority.PREMIUM: 0.05, Priority.STANDARD: 0.3, Priority.FREE: 0.65}
        },
        {
            "name": "Heavy Premium Load",
            "duration": 15,
            "request_rate": 5.0,
            "load_factor": 0.9,
            "priority_distribution": {Priority.PREMIUM: 0.4, Priority.STANDARD: 0.4, Priority.FREE: 0.2}
        }
    ]
    
    request_id = 0
    
    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        
        # Update system state
        base_queue = int(scenario['load_factor'] * 10)
        qos.update_system_metrics(
            prefill_queue=base_queue,
            decode_queue=base_queue // 2,
            load=scenario['load_factor']
        )
        
        # Generate requests for this scenario
        scenario_start = time.time()
        while time.time() - scenario_start < scenario['duration']:
            # Randomly generate a request
            priority = random.choices(
                list(scenario['priority_distribution'].keys()),
                weights=list(scenario['priority_distribution'].values())
            )[0]
            
            request = Request(
                id=f"req-{request_id:04d}",
                prompt_length=random.randint(50, 500),
                expected_output_length=random.randint(20, 200),
                priority=priority,
                arrival_time=time.time()
            )
            request_id += 1
            
            # Try to admit the request
            admitted = qos.admit_request(request)
            
            if admitted:
                # Simulate request processing
                actual_ttft = qos._estimate_ttft(request) + random.uniform(-10, 20)
                actual_tpot = qos.metrics.avg_decode_time_per_req + random.uniform(-5, 10)
                
                # Complete request after a short delay
                def complete_later():
                    time.sleep(0.1)  # Simulate processing time
                    qos.complete_request(request, actual_ttft, actual_tpot)
                
                threading.Thread(target=complete_later, daemon=True).start()
            
            # Wait between requests based on rate
            time.sleep(1.0 / scenario['request_rate'])
            
            # Occasionally update system metrics during the scenario
            if random.random() < 0.1:
                load_variance = random.uniform(-0.1, 0.1)
                new_load = max(0, min(1, scenario['load_factor'] + load_variance))
                new_prefill_queue = max(0, base_queue + random.randint(-2, 3))
                new_decode_queue = max(0, base_queue // 2 + random.randint(-1, 2))
                
                qos.update_system_metrics(new_prefill_queue, new_decode_queue, new_load)
        
        # Print stats for this scenario
        qos.print_stats()
        
        # Brief pause between scenarios
        time.sleep(1)
    
    print(f"\n=== Final Summary ===")
    qos.print_stats()
    
    # Analyze SLO compliance
    print(f"\n=== SLO Analysis ===")
    if len(qos.metrics.recent_ttft_samples) > 0:
        ttft_samples = list(qos.metrics.recent_ttft_samples)
        ttft_p95 = statistics.quantiles(ttft_samples, n=20)[18] if len(ttft_samples) >= 20 else max(ttft_samples)
        ttft_p99 = statistics.quantiles(ttft_samples, n=100)[98] if len(ttft_samples) >= 100 else max(ttft_samples)
        
        print(f"TTFT P95: {ttft_p95:.1f}ms")
        print(f"TTFT P99: {ttft_p99:.1f}ms")
        
        # Check SLO compliance by priority
        for priority in Priority:
            slo_limit = qos.TTFT_SLO_MAX[priority]
            violations = sum(1 for ttft in ttft_samples if ttft > slo_limit)
            violation_rate = violations / len(ttft_samples) if ttft_samples else 0
            
            print(f"{priority.value} SLO ({slo_limit}ms): {violation_rate*100:.1f}% violations")


def demonstrate_qos_configuration():
    """Demonstrate QoS configuration similar to Chapter 17's YAML example."""
    
    # This simulates the configuration from Chapter 17
    qos_config = {
        "scheduler": {
            "qos_classes": [
                {
                    "name": "premium",
                    "reserved_fraction": 0.10,
                    "priority": 100,
                    "slo_ttft_ms": 200,
                    "slo_tpot_ms": 30
                },
                {
                    "name": "standard", 
                    "reserved_fraction": 0.30,
                    "priority": 50,
                    "slo_ttft_ms": 500,
                    "slo_tpot_ms": 50
                },
                {
                    "name": "free",
                    "reserved_fraction": 0.60,
                    "priority": 10,
                    "slo_ttft_ms": 1000,
                    "slo_tpot_ms": 100
                }
            ],
            "request_router": {
                "routes": [
                    {"match": {"header": "x-customer-tier: premium"}, "qos": "premium"},
                    {"match": {"header": "x-customer-tier: standard"}, "qos": "standard"},
                    {"match": {"header": "x-customer-tier: free"}, "qos": "free"},
                    {"match": {}, "qos": "free"}  # default fallback
                ]
            }
        }
    }
    
    print("=== QoS Configuration Example ===")
    print("This configuration reserves capacity for different tiers:")
    
    for qos_class in qos_config["scheduler"]["qos_classes"]:
        print(f"  {qos_class['name']:8}: {qos_class['reserved_fraction']*100:4.0f}% capacity, "
              f"TTFT≤{qos_class['slo_ttft_ms']}ms, TPOT≤{qos_class['slo_tpot_ms']}ms")
    
    return qos_config


def main():
    """Main demonstration of early rejection and QoS policies."""
    
    # First show the configuration
    demonstrate_qos_configuration()
    
    print(f"\n" + "="*60)
    
    # Run the load spike simulation
    simulate_load_spike()
    
    print(f"\n=== Key Benefits of Early Rejection ===")
    print("- Prevents system overload and cascade failures")
    print("- Maintains SLO compliance for admitted requests")
    print("- Provides fair resource allocation across priority tiers") 
    print("- Enables graceful degradation under extreme load")
    print("- Protects premium users from free tier traffic spikes")


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
