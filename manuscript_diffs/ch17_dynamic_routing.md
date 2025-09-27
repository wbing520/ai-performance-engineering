# Chapter 17 — `dynamic_routing.py`

Page ≈ 592 updates:

```diff
-class Request:
-    def __init__(self, id, prompt_tokens, priority, timestamp):
-        self.id = id
-        self.prompt_tokens = prompt_tokens
-        self.priority = priority
-        self.timestamp = timestamp
+@dataclass
+class Request:
+    id: str
+    prompt_tokens: List[int]
+    priority: Priority
+    timestamp: float
+    prefix_cached_length: int = 0
+    expected_output_length: int = 50
```

```diff
-class DisaggregatedRouter:
-    def __init__(self):
-        self.PREFILL_LENGTH_THRESHOLD = 128
-        self.PREFILL_QUEUE_MAX = 8
-        self.TTFT_SLO_MAX = 400
-        self.prefill_workers = {}
-        self.decode_workers = {}
+class DisaggregatedRouter:
+    def __init__(self, config_path: Optional[str] = None):
+        self.PREFILL_LENGTH_THRESHOLD = 100
+        self.PREFILL_QUEUE_MAX = 10
+        self.TTFT_SLO_MAX = 500
+        self.occupancy_weight = 0.7
+        self.active_req_weight = 0.3
+        self.prefill_workers: Dict[str, WorkerMetrics] = {}
+        self.decode_workers: Dict[str, WorkerMetrics] = {}
+        self.avg_prefill_time_per_req = 50.0
+        self.avg_decode_time_per_req = 10.0
+
+        if config_path:
+            self.load_config(config_path)
```

```diff
-    def load_config(self, path):
-        pass
+    def load_config(self, config_path: str):
+        with open(config_path, 'r') as f:
+            if config_path.endswith(('.yaml', '.yml')):
+                config = yaml.safe_load(f)
+            else:
+                config = json.load(f)
+
+        split_policy = config.get('split_policy', {})
+        self.PREFILL_LENGTH_THRESHOLD = split_policy.get('prompt_length_threshold', 100)
+        print(f"Loaded configuration from {config_path}")
+        print(f"Prefill threshold: {self.PREFILL_LENGTH_THRESHOLD} tokens")
```
