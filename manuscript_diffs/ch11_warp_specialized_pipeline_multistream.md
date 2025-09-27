# Chapter 11 — warp_specialized_pipeline_multistream.cu (multistream sync)

```diff
-        // Synchronize to ensure data is loaded
-        __syncthreads();
+        // Synchronize to ensure data is loaded
+        cta.sync();

-        // Synchronize to ensure computation is done
-        __syncthreads();
+        // Synchronize to ensure computation is done
+        cta.sync();

-        // Synchronize before next iteration
-        __syncthreads();
+        // Synchronize before next iteration
+        cta.sync();
```

> Applies to the Chapter 11 snippet “// warp_specialized_pipeline_multistream.cu” (Overlapping Kernel Execution with CUDA Streams, page ~427).
