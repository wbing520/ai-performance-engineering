# Chapter 15 — `disaggregated_inference.py`

Page ≈ 528 updates:

```diff
-@dataclass
-class InferenceConfig:
-    model_size: int = 7_000_000_000
-    num_gpus: int = 8
-    batch_size: int = 32
-    sequence_length: int = 2048
-    num_experts: int = 8
-    top_k_experts: int = 2
-    capacity_factor: float = 1.2
-    use_speculative: bool = True
-    use_disaggregated: bool = True
+@dataclass
+class InferenceConfig:
+    model_size: int = 7_000_000_000
+    num_gpus: int = None
+    batch_size: int = 32
+    sequence_length: int = 2048
+    num_experts: int = 8
+    top_k_experts: int = 2
+    capacity_factor: float = 1.2
+    use_speculative: bool = True
+    use_disaggregated: bool = True
+
+    def __post_init__(self):
+        if self.num_gpus is None:
+            if torch.cuda.is_available():
+                self.num_gpus = torch.cuda.device_count()
+            else:
+                self.num_gpus = 1
+        if torch.cuda.is_available():
+            available_gpus = torch.cuda.device_count()
+            self.num_gpus = min(self.num_gpus, available_gpus)
```

```diff
-        for i in range(self.config.num_gpus // 2):
-            worker = PrefillWorker(worker_id=i, gpu_id=i, config=self.config)
+        num_prefill_workers = max(1, self.config.num_gpus // 2)
+        for i in range(num_prefill_workers):
+            gpu_id = i % self.config.num_gpus
+            worker = PrefillWorker(worker_id=i, gpu_id=gpu_id, config=self.config)
@@
-        for i in range(self.config.num_gpus // 2):
-            worker = DecodeWorker(worker_id=i, gpu_id=i + self.config.num_gpus // 2, config=self.config)
+        num_decode_workers = max(1, self.config.num_gpus // 2)
+        for i in range(num_decode_workers):
+            gpu_id = (i + self.config.num_gpus // 2) % self.config.num_gpus
+            worker = DecodeWorker(worker_id=i, gpu_id=gpu_id, config=self.config)
```
