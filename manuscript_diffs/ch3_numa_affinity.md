# Chapter 3 — `bind_numa_affinity.py`

Page ≈ 116 updates:

```diff
-import torch
-import torch.distributed as dist
-import os
-import ctypes
-import psutil
-import subprocess
-import re
-from torch.utils.data import Dataset, DataLoader
+import torch.profiler as profiler
+from torch.profiler import profile, record_function, ProfilerActivity, schedule
+import torch.cuda.nvtx as nvtx
+import torch
+import torch.distributed as dist
+import os
+import ctypes
+import psutil
+import subprocess
+import re
+from torch.utils.data import Dataset, DataLoader
```

```diff
-def get_architecture_info():
-    return {"name": "Generic GPU"}
+def get_architecture_info():
+    """Get detailed architecture information."""
+    arch = get_architecture()
+    if arch == "blackwell":
+        return {
+            "name": "Blackwell B200/B300",
+            "compute_capability": "10.0",
+            "sm_version": "sm_100",
+            "memory_bandwidth": "8.0 TB/s",
+            "tensor_cores": "5th Gen",
+            "features": ["HBM3e", "TMA", "NVLink-C2C"],
+        }
+    return {
+        "name": "Other",
+        "compute_capability": "Unknown",
+        "sm_version": "Unknown",
+        "memory_bandwidth": "Unknown",
+        "tensor_cores": "Unknown",
+        "features": [],
+    }
```

```diff
-    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
+    dataloader = DataLoader(
+        dataset,
+        batch_size=32,
+        num_workers=0,  # Disable multiprocessing to avoid CUDA issues
+        pin_memory=True,
+        # worker_init_fn=worker_init_fn  # Disable worker init function
+    )
```
