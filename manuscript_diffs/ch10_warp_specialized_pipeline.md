# Chapter 10 — warp_specialized_pipeline.cu (update `__syncthreads()` to `cta.sync()`)

```diff
-    // Synchronize all warps
-    __syncthreads();
+    // Synchronize all warps
+    cta.sync();

-    // Synchronize all warps
-    __syncthreads();
+    // Synchronize all warps
+    cta.sync();

-    // Synchronize all warps before next iteration
-    __syncthreads();
+    // Synchronize all warps before next iteration
+    cta.sync();
```

> Apply this to the snippet beginning "// warp_specialized_pipeline.cu" on Chapter 10 page ~384 (around the loader/compute/storer loop).
