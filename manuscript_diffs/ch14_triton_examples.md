# Chapter 14 — `triton_examples.py`

Page ≈ 482 updates:

```diff
-# TODO: enable Triton optimizations once PyTorch exposes configuration hooks
+from arch_config import arch_config, configure_optimizations
+
+
+def setup_triton_optimizations():
+    """Setup Triton 3.5.0 optimizations for current architecture."""
+    configure_optimizations()
+
+    if torch.cuda.is_available():
+        device_props = torch.cuda.get_device_properties(0)
+        compute_capability = f"{device_props.major}.{device_props.minor}"
+
+        print(f"Triton 3.5.0 optimizations for {device_props.name}")
+        print(f"Compute Capability: {compute_capability}")
+
+        if compute_capability == "10.0":
+            if hasattr(triton.Config, 'use_blackwell_optimizations'):
+                triton.Config.use_blackwell_optimizations = True
+            if hasattr(triton.Config, 'hbm3e_optimizations'):
+                triton.Config.hbm3e_optimizations = True
+            if hasattr(triton.Config, 'tma_support'):
+                triton.Config.tma_support = True
+            if hasattr(triton.Config, 'stream_ordered_memory'):
+                triton.Config.stream_ordered_memory = True
```

```diff
-    BLOCK_SIZE = 2048
+    BLOCK_SIZE = 1024
```

```diff
-    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N))
+    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
```

```diff
-    matmul_kernel[grid](
-        a, b, c,
-        M, N, K,
-        stride_am, stride_ak,
-        stride_bk, stride_bn,
-        stride_cm, stride_cn,
-        BLOCK_SIZE_M=BLOCK_SIZE_M,
-        BLOCK_SIZE_N=BLOCK_SIZE_N,
-        BLOCK_SIZE_K=BLOCK_SIZE_K,
-    )
+    matmul_kernel[grid](
+        a, b, c,
+        M, N, K,
+        stride_am, stride_ak,
+        stride_bk, stride_bn,
+        stride_cm, stride_cn,
+        BLOCK_SIZE_M=BLOCK_SIZE_M,
+        BLOCK_SIZE_N=BLOCK_SIZE_N,
+        BLOCK_SIZE_K=BLOCK_SIZE_K,
+    )
```
