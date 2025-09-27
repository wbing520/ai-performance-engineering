# Chapter 9 — `optimized_gemm_kernel`

Add cooperative-groups barriers to the shared-memory GEMM listing:

```diff
-#include <cuda_runtime.h>
-#include <cublas_v2.h>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+#include <cublas_v2.h>
+
+namespace cg = cooperative_groups;
 
 __global__ void optimized_gemm_kernel(...) {
     __shared__ float sA[32][32];
     __shared__ float sB[32][32];
+
+    cg::thread_block block = cg::this_thread_block();
 
     for (int k = 0; k < K; k += 32) {
         ...
-        __syncthreads();
+        block.sync();
         ...
-        __syncthreads();
+        block.sync();
     }
 }
```
