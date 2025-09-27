# Chapter 7 — `tiledMatMul` listing

Page ≈ 257 updates:

```diff
-#include <cuda_runtime.h>
-#include <iostream>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+#include <iostream>
+
+namespace cg = cooperative_groups;
 
 __global__ void tiledMatMul(const float* A, const float* B, float* C, int N) {
     __shared__ float sA[TILE_SIZE][TILE_SIZE];
     __shared__ float sB[TILE_SIZE][TILE_SIZE];
+
+    cg::thread_block block = cg::this_thread_block();
 
     ...
-        __syncthreads();
+        block.sync();
         for (int k = 0; k < TILE_SIZE; ++k) {
             sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
         }
-        __syncthreads();
+        block.sync();
     }
 }
```
