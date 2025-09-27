# Chapter 7 — `transposePadded` listing

Apply the same cooperative-groups barrier (page ≈ 256):

```diff
-#include <cuda_runtime.h>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+
+namespace cg = cooperative_groups;
 
 __global__ void transposePadded(const float *idata, float *odata, int width) {
     __shared__ float tile[TILE_DIM][TILE_DIM + PAD];
+
+    cg::thread_block block = cg::this_thread_block();
 
     ...
-    __syncthreads();
+    block.sync();
 
     odata[x * width + y] = tile[threadIdx.y][threadIdx.x];
 }
```
