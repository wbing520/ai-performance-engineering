#!/usr/bin/env python3
"""
Dynamic routing policy for disaggregated prefill and decode.
This demonstrates the routing logic that decides when to offload prefill.
"""

import time
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class RequestPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Request:
    """Inference request with metadata"""
    prompt_tokens: List[int]
    prompt_length: int
    prefix_cached_length: int
    priority: RequestPriority
    request_id: str

@dataclass
class WorkerMetrics:
    """Metrics for a worker node"""
    waiting_queue: int
    kv_cache_usage: float  # percentage
    gpu_utilization: float  # percentage
    memory_bandwidth: float  # percentage

class DisaggregatedRouter:
    """Router for disaggregated prefill and decode"""
    
    def __init__(self):
        # Configuration thresholds
        self.PREFILL_LENGTH_THRESHOLD = 256
        self.PREFILL_QUEUE_MAX = 10
        self.DECODE_LOAD_THRESHOLD = 80.0  # percentage
        
        # Worker pools
        self.prefill_workers = {
            "prefill-1": WorkerMetrics(2, 30.0, 85.0, 75.0),
            "prefill-2": WorkerMetrics(0, 10.0, 45.0, 30.0),
        }
        
        self.decode_workers = {
            "decode-1": WorkerMetrics(1, 60.0, 92.0, 88.0),
            "decode-2": WorkerMetrics(0, 40.0, 65.0, 55.0),
        }
        
        # Statistics
        self.routing_stats = {
            "total_requests": 0,
            "offloaded_requests": 0,
            "local_requests": 0,
            "cache_hits": 0,
        }
    
    def should_offload_prefill(self, request: Request) -> bool:
        """Determine if prefill should be offloaded"""
        
        # Calculate effective prefill length (prompt minus cached part)
        effective_prefill_length = request.prompt_length - request.prefix_cached_length
        
        # Condition 1: Long effective prefill
        long_prefill = effective_prefill_length > self.PREFILL_LENGTH_THRESHOLD
        
        # Condition 2: Prefill workers have capacity
        prefill_queue_size = sum(w.waiting_queue for w in self.prefill_workers.values())
        prefill_available = prefill_queue_size < self.PREFILL_QUEUE_MAX
        
        # Condition 3: Decode workers are not overloaded
        decode_load = max(w.gpu_utilization for w in self.decode_workers.values())
        decode_available = decode_load < self.DECODE_LOAD_THRESHOLD
        
        # Condition 4: Priority-based routing
        priority_factor = self._get_priority_factor(request.priority)
        
        # Calculate routing score
        routing_score = self._calculate_routing_score(
            effective_prefill_length, prefill_queue_size, decode_load, priority_factor
        )
        
        should_offload = (long_prefill and prefill_available and 
                         decode_available and routing_score > 0.5)
        
        # Update statistics
        self.routing_stats["total_requests"] += 1
        if should_offload:
            self.routing_stats["offloaded_requests"] += 1
        else:
            self.routing_stats["local_requests"] += 1
        
        if request.prefix_cached_length > 0:
            self.routing_stats["cache_hits"] += 1
        
        return should_offload
    
    def _get_priority_factor(self, priority: RequestPriority) -> float:
        """Get priority factor for routing"""
        priority_factors = {
            RequestPriority.LOW: 0.3,
            RequestPriority.MEDIUM: 0.6,
            RequestPriority.HIGH: 1.0,
        }
        return priority_factors.get(priority, 0.5)
    
    def _calculate_routing_score(self, effective_length: int, prefill_queue: int, 
                                decode_load: float, priority_factor: float) -> float:
        """Calculate routing score (higher = more likely to offload)"""
        
        # Length factor (longer prompts prefer offload)
        length_factor = min(effective_length / self.PREFILL_LENGTH_THRESHOLD, 2.0)
        
        # Queue factor (shorter queues prefer offload)
        queue_factor = max(0, 1.0 - prefill_queue / self.PREFILL_QUEUE_MAX)
        
        # Load factor (lower decode load prefers offload)
        load_factor = max(0, 1.0 - decode_load / 100.0)
        
        # Combine factors with weights
        score = (0.4 * length_factor + 
                0.3 * queue_factor + 
                0.2 * load_factor + 
                0.1 * priority_factor)
        
        return score
    
    def select_workers(self, request: Request, should_offload: bool) -> Tuple[str, str]:
        """Select prefill and decode workers"""
        
        if should_offload:
            # Select prefill worker
            prefill_worker = self._select_prefill_worker(request)
            # Select decode worker
            decode_worker = self._select_decode_worker(request)
        else:
            # For local prefill, use a hybrid worker or decode worker
            prefill_worker = "hybrid-1"  # Simulate hybrid worker
            decode_worker = self._select_decode_worker(request)
        
        return prefill_worker, decode_worker
    
    def _select_prefill_worker(self, request: Request) -> str:
        """Select best prefill worker based on load and cache"""
        best_worker = None
        best_score = float('-inf')
        
        for worker_name, metrics in self.prefill_workers.items():
            # Calculate worker score (lower queue, lower KV usage = better)
            queue_score = max(0, 1.0 - metrics.waiting_queue / 5.0)
            kv_score = max(0, 1.0 - metrics.kv_cache_usage / 100.0)
            utilization_score = max(0, 1.0 - metrics.gpu_utilization / 100.0)
            
            worker_score = 0.4 * queue_score + 0.4 * kv_score + 0.2 * utilization_score
            
            if worker_score > best_score:
                best_score = worker_score
                best_worker = worker_name
        
        return best_worker or "prefill-1"
    
    def _select_decode_worker(self, request: Request) -> str:
        """Select best decode worker based on load and cache"""
        best_worker = None
        best_score = float('-inf')
        
        for worker_name, metrics in self.decode_workers.items():
            # Calculate worker score
            queue_score = max(0, 1.0 - metrics.waiting_queue / 3.0)
            kv_score = max(0, 1.0 - metrics.kv_cache_usage / 100.0)
            utilization_score = max(0, 1.0 - metrics.gpu_utilization / 100.0)
            
            worker_score = 0.3 * queue_score + 0.4 * kv_score + 0.3 * utilization_score
            
            if worker_score > best_score:
                best_score = worker_score
                best_worker = worker_name
        
        return best_worker or "decode-1"
    
    def update_worker_metrics(self, worker_name: str, new_metrics: WorkerMetrics):
        """Update worker metrics"""
        if worker_name in self.prefill_workers:
            self.prefill_workers[worker_name] = new_metrics
        elif worker_name in self.decode_workers:
            self.decode_workers[worker_name] = new_metrics
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        stats = self.routing_stats.copy()
        if stats["total_requests"] > 0:
            stats["offload_rate"] = stats["offloaded_requests"] / stats["total_requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        return stats

def generate_test_requests() -> List[Request]:
    """Generate test requests with varying characteristics"""
    requests = []
    
    # Short prompts (likely local prefill)
    for i in range(5):
        prompt_length = random.randint(10, 100)
        prefix_cached = random.randint(0, min(50, prompt_length))
        requests.append(Request(
            prompt_tokens=[random.randint(1, 1000) for _ in range(prompt_length)],
            prompt_length=prompt_length,
            prefix_cached_length=prefix_cached,
            priority=random.choice(list(RequestPriority)),
            request_id=f"req_{i}_short"
        ))
    
    # Long prompts (likely offload)
    for i in range(5):
        prompt_length = random.randint(500, 2000)
        prefix_cached = random.randint(0, min(200, prompt_length))
        requests.append(Request(
            prompt_tokens=[random.randint(1, 1000) for _ in range(prompt_length)],
            prompt_length=prompt_length,
            prefix_cached_length=prefix_cached,
            priority=random.choice(list(RequestPriority)),
            request_id=f"req_{i}_long"
        ))
    
    return requests

def main():
    """Main demonstration function"""
    print("Dynamic Routing Policy for Disaggregated Prefill/Decode")
    print("======================================================")
    
    # Initialize router
    router = DisaggregatedRouter()
    
    # Generate test requests
    requests = generate_test_requests()
    
    print(f"\nProcessing {len(requests)} requests...")
    print("\nRequest Routing Decisions:")
    print("-" * 80)
    
    for i, request in enumerate(requests):
        # Make routing decision
        should_offload = router.should_offload_prefill(request)
        prefill_worker, decode_worker = router.select_workers(request, should_offload)
        
        # Print decision
        decision = "OFFLOAD" if should_offload else "LOCAL"
        effective_length = request.prompt_length - request.prefix_cached_length
        
        print(f"Request {i+1:2d}: {request.request_id}")
        print(f"  Prompt Length: {request.prompt_length:4d} tokens")
        print(f"  Cached Prefix: {request.prefix_cached_length:4d} tokens")
        print(f"  Effective Length: {effective_length:4d} tokens")
        print(f"  Priority: {request.priority.value}")
        print(f"  Decision: {decision}")
        print(f"  Prefill Worker: {prefill_worker}")
        print(f"  Decode Worker: {decode_worker}")
        print()
    
    # Print final statistics
    stats = router.get_routing_stats()
    print("Routing Statistics:")
    print("-" * 40)
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Offloaded Requests: {stats['offloaded_requests']}")
    print(f"Local Requests: {stats['local_requests']}")
    print(f"Offload Rate: {stats.get('offload_rate', 0):.1%}")
    print(f"Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1%}")
    
    print("\nWorker Pool Status:")
    print("-" * 40)
    print("Prefill Workers:")
    for name, metrics in router.prefill_workers.items():
        print(f"  {name}: Queue={metrics.waiting_queue}, KV={metrics.kv_cache_usage:.1f}%, GPU={metrics.gpu_utilization:.1f}%")
    
    print("\nDecode Workers:")
    for name, metrics in router.decode_workers.items():
        print(f"  {name}: Queue={metrics.waiting_queue}, KV={metrics.kv_cache_usage:.1f}%, GPU={metrics.gpu_utilization:.1f}%")

if __name__ == "__main__":
    main()
