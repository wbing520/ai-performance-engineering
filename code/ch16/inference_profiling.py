#!/usr/bin/env python3
"""
Chapter 16: Profiling, Debugging, and Tuning Inference at Scale

This example demonstrates:
- Comprehensive monitoring and metrics collection
- Dynamic batching and scheduling strategies
- Quantization techniques (GPTQ, AWQ, SmoothQuant)
- Application-level optimizations (prefix caching, streaming, model cascading)
- Performance troubleshooting and debugging
"""

import torch
import torch.nn as nn
import torch.profiler as profiler
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum
import psutil
import GPUtil
from collections import defaultdict, deque
import hashlib


class MonitoringSystem:
    """Comprehensive monitoring and metrics collection for inference systems."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.start_time = time.time()
        
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric with timestamp."""
        if timestamp is None:
            timestamp = time.time()
        self.metrics[metric_name].append((timestamp, value))
        
    def get_gpu_metrics(self) -> Dict[str, float]:
        """Get current GPU metrics."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                return {
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_temperature': gpu.temperature,
                    'gpu_power_draw': gpu.power
                }
        except:
            return {}
            
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_utilization': cpu_percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent
        }
        
    def check_alerts(self) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # GPU utilization alerts
        gpu_metrics = self.get_gpu_metrics()
        if gpu_metrics.get('gpu_utilization', 0) < 10:
            alerts.append("GPU utilization below 10%")
        elif gpu_metrics.get('gpu_utilization', 0) > 90:
            alerts.append("GPU utilization above 90%")
            
        # Memory alerts
        if gpu_metrics.get('gpu_memory_used', 0) / gpu_metrics.get('gpu_memory_total', 1) > 0.8:
            alerts.append("GPU memory usage above 80%")
            
        # Temperature alerts
        if gpu_metrics.get('gpu_temperature', 0) > 85:
            alerts.append("GPU temperature above 85°C")
            
        return alerts
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        report = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'gpu_metrics': self.get_gpu_metrics(),
            'system_metrics': self.get_system_metrics(),
            'alerts': self.check_alerts(),
            'metric_summaries': {}
        }
        
        # Calculate metric summaries
        for metric_name, values in self.metrics.items():
            if values:
                recent_values = [v for _, v in values[-100:]]  # Last 100 values
                report['metric_summaries'][metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'average': np.mean(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values)
                }
                
        return report


class DynamicBatcher:
    """Dynamic batching with latency-aware scheduling."""
    
    def __init__(self, max_batch_size: int = 32, max_delay_ms: int = 10):
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms
        self.request_queue = deque()
        self.batch_history = []
        
    def add_request(self, request_id: str, prompt: str, priority: int = 0) -> bool:
        """Add a request to the batch queue."""
        request = {
            'id': request_id,
            'prompt': prompt,
            'priority': priority,
            'timestamp': time.time(),
            'tokens': len(prompt.split())  # Simplified token count
        }
        self.request_queue.append(request)
        return True
        
    def get_batch(self) -> List[Dict]:
        """Get the next batch of requests to process."""
        if not self.request_queue:
            return []
            
        current_time = time.time()
        batch = []
        
        # Sort by priority and timestamp
        sorted_requests = sorted(
            self.request_queue, 
            key=lambda x: (-x['priority'], x['timestamp'])
        )
        
        for request in sorted_requests:
            # Check if we should include this request
            age_ms = (current_time - request['timestamp']) * 1000
            
            if len(batch) < self.max_batch_size and age_ms <= self.max_delay_ms:
                batch.append(request)
            elif age_ms > self.max_delay_ms:
                # Force include old requests
                batch.append(request)
                
        # Remove processed requests from queue
        for request in batch:
            self.request_queue.remove(request)
            
        if batch:
            self.batch_history.append({
                'timestamp': current_time,
                'batch_size': len(batch),
                'avg_tokens': np.mean([r['tokens'] for r in batch])
            })
            
        return batch
        
    def get_batch_stats(self) -> Dict[str, float]:
        """Get batch statistics."""
        if not self.batch_history:
            return {}
            
        recent_batches = self.batch_history[-100:]  # Last 100 batches
        return {
            'avg_batch_size': np.mean([b['batch_size'] for b in recent_batches]),
            'avg_tokens_per_batch': np.mean([b['avg_tokens'] for b in recent_batches]),
            'queue_length': len(self.request_queue)
        }


class QuantizationManager:
    """Manages different quantization techniques for inference optimization."""
    
    def __init__(self):
        self.quantization_configs = {
            'fp16': {'precision': 'fp16', 'compression_ratio': 2.0},
            'fp8': {'precision': 'fp8', 'compression_ratio': 4.0},
            'int8': {'precision': 'int8', 'compression_ratio': 4.0},
            'int4': {'precision': 'int4', 'compression_ratio': 8.0}
        }
        
    def gptq_quantize(self, model: nn.Module, calibration_data: List[str]) -> nn.Module:
        """Apply GPTQ quantization to model weights."""
        print("Applying GPTQ quantization...")
        
        # Simulate GPTQ quantization process
        quantized_model = model
        
        # Apply quantization to linear layers
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Simulate weight quantization
                with torch.no_grad():
                    # Quantize weights to 4-bit
                    weights = module.weight.data
                    quantized_weights = self._quantize_weights(weights, bits=4)
                    module.weight.data = quantized_weights
                    
        print(f"GPTQ quantization completed. Model size reduced by ~4x")
        return quantized_model
        
    def awq_quantize(self, model: nn.Module, calibration_data: List[str]) -> nn.Module:
        """Apply AWQ quantization to model weights."""
        print("Applying AWQ quantization...")
        
        # Simulate AWQ quantization process
        quantized_model = model
        
        # Apply channel-specific scaling
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weights = module.weight.data
                    # Apply channel-specific scaling for salient channels
                    scaled_weights = self._apply_channel_scaling(weights)
                    quantized_weights = self._quantize_weights(scaled_weights, bits=4)
                    module.weight.data = quantized_weights
                    
        print(f"AWQ quantization completed. Model size reduced by ~4x")
        return quantized_model
        
    def smoothquant_quantize(self, model: nn.Module, calibration_data: List[str]) -> nn.Module:
        """Apply SmoothQuant for activation quantization."""
        print("Applying SmoothQuant quantization...")
        
        # Simulate SmoothQuant process
        quantized_model = model
        
        # Apply row/column scaling to shift quantization error
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    weights = module.weight.data
                    # Apply SmoothQuant scaling
                    scaled_weights = self._apply_smoothquant_scaling(weights)
                    quantized_weights = self._quantize_weights(scaled_weights, bits=8)
                    module.weight.data = quantized_weights
                    
        print(f"SmoothQuant quantization completed. Model size reduced by ~2x")
        return quantized_model
        
    def _quantize_weights(self, weights: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize weights to specified bit precision."""
        # Simulate quantization
        if bits == 4:
            # 4-bit quantization
            scale = weights.abs().max() / 7.0  # 4-bit range: -7 to 7
            quantized = torch.round(weights / scale) * scale
        elif bits == 8:
            # 8-bit quantization
            scale = weights.abs().max() / 127.0  # 8-bit range: -127 to 127
            quantized = torch.round(weights / scale) * scale
        else:
            quantized = weights
            
        return quantized
        
    def _apply_channel_scaling(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply channel-specific scaling for AWQ."""
        # Simulate channel-specific scaling
        channel_scales = torch.rand(weights.size(0)) * 0.5 + 0.75  # 0.75-1.25 scale
        scaled_weights = weights * channel_scales.unsqueeze(1)
        return scaled_weights
        
    def _apply_smoothquant_scaling(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply SmoothQuant scaling to shift quantization error."""
        # Simulate SmoothQuant scaling
        row_scales = torch.rand(weights.size(0)) * 0.5 + 0.75
        col_scales = torch.rand(weights.size(1)) * 0.5 + 0.75
        scaled_weights = weights * row_scales.unsqueeze(1) * col_scales.unsqueeze(0)
        return scaled_weights


class PrefixCache:
    """Implements prefix caching for efficient prompt reuse."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
        
    def find_longest_prefix(self, prompt: str) -> Tuple[str, int]:
        """Find the longest cached prefix for a prompt."""
        words = prompt.split()
        
        for length in range(len(words), 0, -1):
            prefix = ' '.join(words[:length])
            cache_key = self.get_cache_key(prefix)
            
            if cache_key in self.cache:
                self.hit_count += 1
                self.access_times[cache_key] = time.time()
                return prefix, length
                
        self.miss_count += 1
        return "", 0
        
    def cache_prefix(self, prompt: str, kv_cache: Dict):
        """Cache a prompt prefix with its KV cache."""
        cache_key = self.get_cache_key(prompt)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._evict_oldest()
            
        self.cache[cache_key] = kv_cache
        self.access_times[cache_key] = time.time()
        
    def _evict_oldest(self):
        """Evict the least recently used cache entry."""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size
        }


class ModelCascader:
    """Implements model cascading for tiered inference."""
    
    def __init__(self):
        self.models = {
            'small': {'size': '7B', 'latency_ms': 50, 'cost': 1.0},
            'medium': {'size': '13B', 'latency_ms': 150, 'cost': 2.0},
            'large': {'size': '70B', 'latency_ms': 500, 'cost': 5.0}
        }
        self.routing_history = []
        
    def classify_request(self, prompt: str) -> str:
        """Classify request complexity to determine model tier."""
        # Simple heuristics for request classification
        words = prompt.split()
        
        # Short factual questions -> small model
        if len(words) < 20 and any(word in prompt.lower() for word in ['what', 'who', 'when', 'where']):
            return 'small'
            
        # Long creative requests -> large model
        if len(words) > 100 or any(word in prompt.lower() for word in ['explain', 'analyze', 'elaborate', 'creative']):
            return 'large'
            
        # Medium complexity -> medium model
        return 'medium'
        
    def route_request(self, prompt: str, user_tier: str = 'standard') -> str:
        """Route request to appropriate model tier."""
        complexity = self.classify_request(prompt)
        
        # Premium users get upgraded model
        if user_tier == 'premium' and complexity == 'small':
            complexity = 'medium'
        elif user_tier == 'premium' and complexity == 'medium':
            complexity = 'large'
            
        # Record routing decision
        self.routing_history.append({
            'timestamp': time.time(),
            'prompt_length': len(prompt.split()),
            'complexity': complexity,
            'user_tier': user_tier
        })
        
        return complexity
        
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {}
            
        recent_routes = self.routing_history[-100:]  # Last 100 routes
        
        complexity_counts = defaultdict(int)
        for route in recent_routes:
            complexity_counts[route['complexity']] += 1
            
        return {
            'total_routes': len(recent_routes),
            'complexity_distribution': dict(complexity_counts),
            'avg_prompt_length': np.mean([r['prompt_length'] for r in recent_routes])
        }


class StreamingResponse:
    """Implements streaming response for improved user experience."""
    
    def __init__(self, tokens_per_batch: int = 5, delay_ms: int = 50):
        self.tokens_per_batch = tokens_per_batch
        self.delay_ms = delay_ms
        self.response_queue = queue.Queue()
        
    def generate_streaming_response(self, prompt: str) -> List[str]:
        """Generate response with streaming simulation."""
        # Simulate token generation
        response_tokens = []
        current_batch = []
        
        # Simulate generating tokens
        for i in range(50):  # Generate 50 tokens
            token = f"token_{i}"
            current_batch.append(token)
            response_tokens.append(token)
            
            # Stream batch when full
            if len(current_batch) >= self.tokens_per_batch:
                self._stream_batch(current_batch)
                current_batch = []
                
            # Simulate generation delay
            time.sleep(self.delay_ms / 1000.0)
            
        # Stream remaining tokens
        if current_batch:
            self._stream_batch(current_batch)
            
        return response_tokens
        
    def _stream_batch(self, tokens: List[str]):
        """Stream a batch of tokens to client."""
        # Simulate streaming to client
        batch_text = ' '.join(tokens)
        print(f"Streaming: {batch_text}")
        
        # Add to response queue for monitoring
        self.response_queue.put({
            'timestamp': time.time(),
            'tokens': tokens,
            'batch_size': len(tokens)
        })


class InferenceOptimizer:
    """Main inference optimization system."""
    
    def __init__(self):
        self.monitor = MonitoringSystem()
        self.batcher = DynamicBatcher()
        self.quantizer = QuantizationManager()
        self.prefix_cache = PrefixCache()
        self.cascader = ModelCascader()
        self.streamer = StreamingResponse()
        
    def process_request(self, request_id: str, prompt: str, user_tier: str = 'standard') -> Dict:
        """Process a single inference request with optimizations."""
        start_time = time.time()
        
        # 1. Check prefix cache
        prefix, prefix_length = self.prefix_cache.find_longest_prefix(prompt)
        cache_hit = prefix_length > 0
        
        # 2. Route to appropriate model
        model_tier = self.cascader.route_request(prompt, user_tier)
        
        # 3. Add to dynamic batch
        self.batcher.add_request(request_id, prompt)
        
        # 4. Simulate inference processing
        processing_time = self._simulate_inference(prompt, model_tier, cache_hit)
        
        # 5. Generate streaming response
        response_tokens = self.streamer.generate_streaming_response(prompt)
        
        # 6. Record metrics
        total_time = time.time() - start_time
        self.monitor.record_metric('request_latency_ms', total_time * 1000)
        self.monitor.record_metric('cache_hit_rate', 1.0 if cache_hit else 0.0)
        self.monitor.record_metric('model_tier_usage', len(model_tier))
        
        return {
            'request_id': request_id,
            'model_tier': model_tier,
            'cache_hit': cache_hit,
            'prefix_length': prefix_length,
            'processing_time_ms': processing_time * 1000,
            'total_time_ms': total_time * 1000,
            'response_tokens': response_tokens
        }
        
    def _simulate_inference(self, prompt: str, model_tier: str, cache_hit: bool) -> float:
        """Simulate inference processing time."""
        base_time = self.cascader.models[model_tier]['latency_ms'] / 1000.0
        
        # Reduce time if cache hit
        if cache_hit:
            base_time *= 0.3  # 70% reduction for cache hits
            
        # Add some variance
        variance = np.random.normal(0, base_time * 0.1)
        processing_time = max(0.01, base_time + variance)
        
        time.sleep(processing_time)  # Simulate actual processing
        return processing_time
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'monitoring': self.monitor.generate_report(),
            'batching': self.batcher.get_batch_stats(),
            'caching': self.prefix_cache.get_cache_stats(),
            'routing': self.cascader.get_routing_stats()
        }
        return stats


def benchmark_inference_optimizations():
    """Benchmark the inference optimization system."""
    print("=== Inference Optimization Benchmark ===\n")
    
    optimizer = InferenceOptimizer()
    
    # Test different types of requests
    test_requests = [
        ("req_1", "What is the weather today?", "standard"),
        ("req_2", "Explain the theory of relativity in detail", "premium"),
        ("req_3", "Who is the president?", "standard"),
        ("req_4", "Write a creative story about a robot", "premium"),
        ("req_5", "What is 2+2?", "standard"),
        ("req_6", "Analyze the economic impact of AI on society", "premium"),
    ]
    
    print("1. Processing requests with optimizations...")
    results = []
    
    for request_id, prompt, user_tier in test_requests:
        result = optimizer.process_request(request_id, prompt, user_tier)
        results.append(result)
        print(f"Request {request_id}: {result['model_tier']} model, "
              f"cache_hit={result['cache_hit']}, "
              f"latency={result['total_time_ms']:.1f}ms")
    
    print("\n2. System Statistics:")
    stats = optimizer.get_system_stats()
    
    print(f"Cache hit rate: {stats['caching']['hit_rate']:.2%}")
    print(f"Average batch size: {stats['batching']['avg_batch_size']:.1f}")
    print(f"Model tier distribution: {stats['routing']['complexity_distribution']}")
    
    print("\n3. GPU and System Metrics:")
    gpu_metrics = stats['monitoring']['gpu_metrics']
    if gpu_metrics:
        print(f"GPU utilization: {gpu_metrics.get('gpu_utilization', 0):.1f}%")
        print(f"GPU memory usage: {gpu_metrics.get('gpu_memory_used', 0)}MB")
        print(f"GPU temperature: {gpu_metrics.get('gpu_temperature', 0)}°C")
    
    print("\n4. Alerts:")
    alerts = stats['monitoring']['alerts']
    if alerts:
        for alert in alerts:
            print(f"⚠️  {alert}")
    else:
        print("✅ No alerts - system running normally")
    
    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    benchmark_inference_optimizations()
