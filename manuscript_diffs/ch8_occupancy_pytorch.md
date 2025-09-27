# Chapter 8 — `occupancy_pytorch.py`

Page ≈ 296 updates:

```diff
-    # Small tensor operations
-    start = time.time()
-    for _ in range(100):
-        small_c = torch.mm(small_a, small_b)
+    start = time.time()
+    with torch.cuda.nvtx.range("small_matmul"):
+        for _ in range(100):
+            small_c = torch.mm(small_a, small_b)
@@
-    # Large tensor operations (fewer iterations to avoid long runtime)
-    start = time.time()
-    for _ in range(10):
-        large_c = torch.mm(large_a, large_b)
+    start = time.time()
+    with torch.cuda.nvtx.range("large_matmul"):
+        for _ in range(10):
+            large_c = torch.mm(large_a, large_b)
```

```diff
-    # Individual operations
-    start = time.time()
-    individual_results = []
-    for tensor in individual_tensors:
-        result = torch.relu(tensor)
-        result = torch.sigmoid(result)
-        individual_results.append(result)
+    start = time.time()
+    with torch.cuda.nvtx.range("individual_ops"):
+        individual_results = []
+        for tensor in individual_tensors:
+            result = torch.relu(tensor)
+            result = torch.sigmoid(result)
+            individual_results.append(result)
@@
-    start = time.time()
-    batched_result = torch.relu(batched_tensor)
-    batched_result = torch.sigmoid(batched_result)
+    start = time.time()
+    with torch.cuda.nvtx.range("batched_ops"):
+        batched_result = torch.relu(batched_tensor)
+        batched_result = torch.sigmoid(batched_result)
```
