# Chapter 1 — `performance_basics.py`

Page ≈ 42 updates:

```diff
-import torch
-import torch.nn as nn
-import torch.optim as optim
-from torch.profiler import profile, record_function
+import torch
+import torch.nn as nn
+import torch.optim as optim
+from torch.profiler import profile, record_function, ProfilerActivity, schedule
+import torch.cuda.nvtx as nvtx
+import psutil
+import GPUtil
+import numpy as np
+from typing import Dict, List, Tuple
+import os
+
+# Import architecture configuration
+import sys
+sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+from arch_config import arch_config, configure_optimizations
```

```diff
-def benchmark_model_performance(model: nn.Module, batch_size: int = 32,
-                               seq_length: int = 128, num_iterations: int = 20,
-                               use_compile: bool = True) -> Dict[str, float]:
+def benchmark_model_performance(
+    model: nn.Module,
+    batch_size: int = 32,
+    seq_length: int = 128,
+    num_iterations: int = 20,
+    use_compile: bool = True,
+) -> Dict[str, float]:
@@
-    dummy_input = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
-
-    if use_compile:
-        compiled_model = torch.compile(model)
-    else:
-        compiled_model = model
+    configure_architecture_optimizations()
+
+    dummy_input = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
+
+    if use_compile and device.type == 'cuda':
+        try:
+            compiled_model = torch.compile(
+                model,
+                mode="max-autotune",
+                fullgraph=False,
+                dynamic=False,
+            )
+        except Exception as e:
+            print(f"Warning: Compilation failed with error: {e}")
+            print("Falling back to uncompiled model")
+            compiled_model = model
+    else:
+        compiled_model = model
```

```diff
-    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
-        with torch.no_grad():
-            for _ in range(num_iterations):
-                with record_function("model_inference"):
-                    _ = compiled_model(dummy_input)
+    if use_compile:
+        with profile(
+            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
+            record_shapes=True,
+            with_stack=True,
+            with_flops=True,
+            with_modules=True,
+            profile_memory=True,
+            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
+        ) as prof:
+            with torch.no_grad():
+                for i in range(num_iterations):
+                    with record_function("model_inference"):
+                        with nvtx.range("forward_pass"):
+                            _ = compiled_model(dummy_input)
+                    if i % 10 == 0:
+                        overhead_start = time.time()
+                        time.sleep(0.001)
+                        monitor.record_overhead(time.time() - overhead_start)
+            try:
+                profiler_data = prof.key_averages().table(sort_by="cuda_time_total", row_limit=5)
+            except Exception as e:
+                print(f"Warning: Could not get profiler data: {e}")
+                profiler_data = None
+    else:
+        with torch.no_grad():
+            for i in range(num_iterations):
+                with record_function("model_inference"):
+                    with nvtx.range("forward_pass"):
+                        _ = compiled_model(dummy_input)
+                if i % 10 == 0:
+                    overhead_start = time.time()
+                    time.sleep(0.001)
+                    monitor.record_overhead(time.time() - overhead_start)
```

```diff
-def demonstrate_hardware_software_co_design():
-    print("=== Chapter 1: AI Systems Performance Engineering Demo ===
")
+def demonstrate_hardware_software_co_design():
+    print("=== Chapter 1: AI Systems Performance Engineering Demo (PyTorch 2.9) ===
")
@@
-    print("• PyTorch 2.8 torch.compile provides significant speedup")
+    print("• PyTorch 2.9 torch.compile provides significant speedup")
+    print("• CUDA 12.9 and Triton 3.4 support latest features")
```
