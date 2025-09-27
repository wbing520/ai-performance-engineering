# Chapter 8 — `ilp_pytorch.py`

Page ≈ 288 updates:

```diff
-    start = time.time()
-    for _ in range(100):
-        result_dep = dependent_operations(a, b, c, d)
+    start = time.time()
+    with torch.cuda.nvtx.range("dependent_ops"):
+        for _ in range(100):
+            result_dep = dependent_operations(a, b, c, d)
@@
-    start = time.time()
-    for _ in range(100):
-        result_indep = independent_operations(a, b, c, d)
+    start = time.time()
+    with torch.cuda.nvtx.range("independent_ops"):
+        for _ in range(100):
+            result_indep = independent_operations(a, b, c, d)
```

```diff
-    for _ in range(50):
-        y1 = torch.sin(x)
-        y2 = torch.cos(x)
-        y3 = torch.exp(x * 0.1)
-        y4 = torch.log(torch.abs(x) + 1)
-        unfused = y1 + y2 + y3 + y4
+    with torch.cuda.nvtx.range("unfused_ilp"):
+        for _ in range(50):
+            y1 = torch.sin(x)
+            y2 = torch.cos(x)
+            y3 = torch.exp(x * 0.1)
+            y4 = torch.log(torch.abs(x) + 1)
+            unfused = y1 + y2 + y3 + y4
@@
-    for _ in range(50):
-        fused = fused_computation(x, 0.1)
+    with torch.cuda.nvtx.range("fused_ilp"):
+        for _ in range(50):
+            fused = fused_computation(x, 0.1)
```
