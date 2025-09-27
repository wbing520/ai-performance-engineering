# Chapter 14 — `torch_compiler_examples.py`

Page ≈ 468 updates:

```diff
-    modes = ['default', 'reduce-overhead']
+    modes = ['default', 'reduce-overhead', 'max-autotune']
```

```diff
-        compiled_model = torch.compile(model, mode=mode)
+        compiled_model = torch.compile(
+            model,
+            mode=mode,
+            fullgraph=True,
+            dynamic=True,
+        )
```

```diff
-        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
-            for _ in range(10):
-                output = compiled_model(x)
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
+                for _ in range(20):
+                    with nvtx.range(f"benchmark_{mode}"):
+                        output = compiled_model(x)
```

```diff
-    compiled_model = torch.compile(model)
+    compiled_model = torch.compile(
+        model,
+        mode="max-autotune",
+        fullgraph=False,
+        dynamic=False,
+    )
```
