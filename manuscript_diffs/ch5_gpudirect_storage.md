# Chapter 5 — `gpudirect_storage_example.py`

Page ≈ 196 updates:

```diff
-class StorageIOMonitor:
-    def __init__(self):
-        self.gpu_utilization = []
-        self.cpu_utilization = []
-        self.memory_usage = []
-
-    def record(self):
-        pass
+class StorageIOMonitor:
+    """Monitor GPU/CPU utilization while streaming batches."""
+
+    def __init__(self):
+        self.gpu_utilization = []
+        self.cpu_utilization = []
+        self.memory_usage = []
+        self.monitoring = False
+        self.monitor_thread = None
+
+    def start_monitoring(self):
+        self.monitoring = True
+        self.start_time = time.time()
+        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
+        self.monitor_thread.start()
+
+    def _monitor_loop(self):
+        while self.monitoring:
+            if torch.cuda.is_available():
+                gpu_util = torch.cuda.utilization()
+                self.gpu_utilization.append(gpu_util)
+            self.cpu_utilization.append(psutil.cpu_percent(interval=1))
+            self.memory_usage.append(psutil.virtual_memory().percent)
+            time.sleep(1)
+
+    def stop_monitoring(self):
+        self.monitoring = False
+        self.end_time = time.time()
+        if self.monitor_thread and self.monitor_thread.is_alive():
+            self.monitor_thread.join()
+
+    def get_metrics(self) -> Dict[str, float]:
+        if not self.gpu_utilization:
+            return {}
+        return {
+            'avg_gpu_utilization': np.mean(self.gpu_utilization),
+            'avg_cpu_utilization': np.mean(self.cpu_utilization),
+            'avg_memory_usage': np.mean(self.memory_usage),
+            'monitoring_duration': self.end_time - self.start_time if self.end_time else 0,
+            'max_gpu_utilization': max(self.gpu_utilization),
+            'min_gpu_utilization': min(self.gpu_utilization),
+        }
```

```diff
-        data = data.cuda()
-        target = target.cuda()
+        if torch.cuda.is_available():
+            data = data.cuda(non_blocking=True)
+            target = target.cuda(non_blocking=True)
```

```diff
-        metrics = {
-            'batches_per_second': batch_count / total_time,
-            'samples_per_second': throughput,
-        }
+        metrics = monitor.get_metrics()
+        metrics.update({
+            'batches_per_second': batch_count / total_time,
+            'samples_per_second': throughput,
+        })
+        return metrics
```
