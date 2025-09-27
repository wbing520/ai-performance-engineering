# Chapter 16 — `inference_profiling.py`

Page ≈ 556 updates:

```diff
-class MonitoringSystem:
-    def __init__(self):
-        self.metrics = {}
-
-    def record_metric(self, metric_name: str, value: float):
-        pass
+class MonitoringSystem:
+    """Comprehensive monitoring and metrics collection for inference systems."""
+
+    def __init__(self):
+        self.metrics = defaultdict(list)
+        self.alerts = []
+        self.start_time = time.time()
+
+    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
+        if timestamp is None:
+            timestamp = time.time()
+        self.metrics[metric_name].append((timestamp, value))
+
+    def get_gpu_metrics(self) -> Dict[str, float]:
+        try:
+            gpus = GPUtil.getGPUs()
+            if gpus:
+                gpu = gpus[0]
+                return {
+                    'gpu_utilization': gpu.load * 100,
+                    'gpu_memory_used': gpu.memoryUsed,
+                    'gpu_memory_total': gpu.memoryTotal,
+                    'gpu_temperature': gpu.temperature,
+                    'gpu_power_draw': gpu.power,
+                }
+        except Exception:
+            return {}
+
+    def get_system_metrics(self) -> Dict[str, float]:
+        cpu_percent = psutil.cpu_percent(interval=1)
+        memory = psutil.virtual_memory()
+        return {
+            'cpu_utilization': cpu_percent,
+            'memory_used_gb': memory.used / (1024**3),
+            'memory_total_gb': memory.total / (1024**3),
+            'memory_percent': memory.percent,
+        }
```

```diff
-    def check_alerts(self):
-        return []
+    def check_alerts(self) -> List[str]:
+        alerts = []
+        gpu_metrics = self.get_gpu_metrics() or {}
+        if gpu_metrics.get('gpu_utilization', 0) < 10:
+            alerts.append("GPU utilization below 10%")
+        elif gpu_metrics.get('gpu_utilization', 0) > 90:
+            alerts.append("GPU utilization above 90%")
+        if gpu_metrics.get('gpu_memory_total'):
+            usage = gpu_metrics.get('gpu_memory_used', 0) / gpu_metrics['gpu_memory_total']
+            if usage > 0.8:
+                alerts.append("GPU memory usage above 80%")
+        if gpu_metrics.get('gpu_temperature', 0) > 85:
+            alerts.append("GPU temperature above 85°C")
+        return alerts
```

```diff
-    def generate_report(self):
-        return {}
+    def generate_report(self) -> Dict[str, Any]:
+        report = {
+            'timestamp': time.time(),
+            'uptime_seconds': time.time() - self.start_time,
+            'gpu_metrics': self.get_gpu_metrics(),
+            'system_metrics': self.get_system_metrics(),
+            'alerts': self.check_alerts(),
+            'metric_summaries': {},
+        }
+        for metric_name, values in self.metrics.items():
+            if values:
+                recent_values = [v for _, v in values[-100:]]
+                report['metric_summaries'][metric_name] = {
+                    'current': recent_values[-1],
+                    'average': np.mean(recent_values),
+                    'min': np.min(recent_values),
+                    'max': np.max(recent_values),
+                }
+        return report
```
