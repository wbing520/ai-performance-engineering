# Chapter 2 — `hardware_info.py`

Page ≈ 78 updates:

```diff
-"""hardware_info.py
-Chapter 2: Hardware Topology Inspection
-
-Analyze CPU/GPU topology, memory bandwidth, and interconnect characteristics
-for NVIDIA Blackwell-based systems."""
-
-import torch
-import psutil
-import GPUtil
-import time
-import numpy as np
-from typing import Dict, Any, List
-import torch.cuda.nvtx as nvtx
+#!/usr/bin/env python3
+"""Chapter 2: Hardware Topology Inspection
+
+Analyze CPU/GPU topology, memory bandwidth, and interconnect characteristics
+for NVIDIA Blackwell-based systems.
+"""
+
+import time
+from typing import Any, Dict
+
+import GPUtil
+import psutil
+import torch
+import torch.cuda.nvtx as nvtx
```

```diff
-        "max_threads_per_block": device_props.max_threads_per_multi_processor,
+        "max_threads_per_block": device_props.max_threads_per_block,
@@
-        "tensor_cores": "4th Generation" if is_blackwell else "Unknown",
-        "max_unified_memory_gb": 30 if is_blackwell else None,
+        "tensor_cores": "5th Generation" if is_blackwell else "Unknown",
+        "max_unified_memory_tb": 30 if is_blackwell else None,
```

```diff
-        print("✓ 4th Generation Tensor Cores")
-        print("✓ Unified Memory Architecture")
-        print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_gb']} TB")
+        print("✓ 5th Generation Tensor Cores")
+        print("✓ Unified Memory Architecture")
+        print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_tb']} TB")
@@
-        print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")
+        bandwidth = gpu_info.get("memory_bandwidth_gbps")
+        bandwidth_str = f"{bandwidth:.1f} GB/s" if bandwidth else "Unknown"
+        print(f"Memory Bandwidth: {bandwidth_str}")
```

```diff
-        print("   • 4th Generation Tensor Cores")
-        print("   • FP8/FP4 precision support")
+        print("   • 5th Generation Tensor Cores")
+        print("   • FP8/FP4 precision support")
```

```diff
-print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")
+print(f"Memory Bandwidth: {_format_bandwidth(gpu_info.get('memory_bandwidth_gbps'))}")
+
+
def _format_bandwidth(bandwidth_gbps: Any) -> str:
+    if isinstance(bandwidth_gbps, (int, float)):
+        return f"{bandwidth_gbps:.1f} GB/s"
+    return "Unknown"
```

```diff
-        print("✓ 4th Generation Tensor Cores")
-        print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_gb']} TB")
+        print("✓ 5th Generation Tensor Cores")
+        print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_tb']} TB")
-        print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")
+        print(f"Memory Bandwidth: {_format_bandwidth(gpu_info.get('memory_bandwidth_gbps'))}")
```
