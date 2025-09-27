# Chapter 19 — `adaptive_parallelism_strategy.py`

Page ≈ 624 updates:

```diff
-class ParallelismStrategy(Enum):
-    TENSOR_PARALLEL = "tensor_parallel"
-    PIPELINE_PARALLEL = "pipeline_parallel"
-    HYBRID = "hybrid"
-
-class DynamicParallelismRouter:
-    def __init__(self):
-        self.available_gpus = 8
-        self.current_strategy = ParallelismStrategy.TENSOR_PARALLEL
+class ParallelismStrategy(Enum):
+    TENSOR_PARALLEL = "tensor_parallel"
+    PIPELINE_PARALLEL = "pipeline_parallel"
+    HYBRID = "hybrid"
+    DATA_PARALLEL = "data_parallel"
+
+@dataclass
+class WorkloadMetrics:
+    seq_len: int
+    batch_size: int
+    gpu_memory_util: float
+    concurrent_requests: int
+    avg_latency_ms: float
+    throughput_tokens_per_sec: float
+    memory_bandwidth_util: float
+    compute_utilization: float
+
+@dataclass
+class ParallelismConfig:
+    strategy: ParallelismStrategy
+    tensor_parallel_size: int
+    pipeline_parallel_size: int
+    data_parallel_size: int
+    estimated_latency_ms: float
+    estimated_throughput: float
+    memory_efficiency: float
+
+class DynamicParallelismRouter:
+    def __init__(self, available_gpus: int = 8):
+        self.available_gpus = available_gpus
+        self.current_strategy = ParallelismStrategy.TENSOR_PARALLEL
+        self.strategy_profiles = self._initialize_strategy_profiles()
+        self.metrics_history: List[WorkloadMetrics] = []
+        self.strategy_performance: Dict[ParallelismStrategy, List[float]] = {
+            strategy: [] for strategy in ParallelismStrategy
+        }
```

```diff
-    def _initialize_strategy_profiles(self):
-        return {}
+    def _initialize_strategy_profiles(self) -> Dict[ParallelismStrategy, ParallelismConfig]:
+        base_latency = 20.0
+        base_throughput = 100_000.0
+        return {
+            ParallelismStrategy.TENSOR_PARALLEL: ParallelismConfig(
+                strategy=ParallelismStrategy.TENSOR_PARALLEL,
+                tensor_parallel_size=min(8, self.available_gpus),
+                pipeline_parallel_size=1,
+                data_parallel_size=max(1, self.available_gpus // 8),
+                estimated_latency_ms=base_latency * 0.8,
+                estimated_throughput=base_throughput * 1.2,
+                memory_efficiency=0.7,
+            ),
+            ParallelismStrategy.PIPELINE_PARALLEL: ParallelismConfig(
+                strategy=ParallelismStrategy.PIPELINE_PARALLEL,
+                tensor_parallel_size=1,
+                pipeline_parallel_size=min(4, self.available_gpus),
+                data_parallel_size=max(1, self.available_gpus // 4),
+                estimated_latency_ms=base_latency * 1.1,
+                estimated_throughput=base_throughput * 0.9,
+                memory_efficiency=0.9,
+            ),
+            ParallelismStrategy.HYBRID: ParallelismConfig(
+                strategy=ParallelismStrategy.HYBRID,
+                tensor_parallel_size=min(4, self.available_gpus // 2),
+                pipeline_parallel_size=min(2, self.available_gpus // 2),
+                data_parallel_size=max(1, self.available_gpus // 4),
+                estimated_latency_ms=base_latency,
+                estimated_throughput=base_throughput * 1.1,
+                memory_efficiency=0.85,
+            ),
+            ParallelismStrategy.DATA_PARALLEL: ParallelismConfig(
+                strategy=ParallelismStrategy.DATA_PARALLEL,
+                tensor_parallel_size=1,
+                pipeline_parallel_size=1,
+                data_parallel_size=self.available_gpus,
+                estimated_latency_ms=base_latency * 1.2,
+                estimated_throughput=base_throughput,
+                memory_efficiency=0.95,
+            ),
+        }
```
