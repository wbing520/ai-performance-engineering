# Chapter 7 — `transposeNaive` listing

Swap the `__syncthreads()` barrier for cooperative groups (page ≈ 254):

```diff
-#include <cuda_runtime.h>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+
+namespace cg = cooperative_groups;
 
 __global__ void transposeNaive(const float *idata, float *odata, int width) {
     __shared__ float tile[TILE_DIM][TILE_DIM];
+
+    cg::thread_block block = cg::this_thread_block();
 
     ...
-    __syncthreads();
+    block.sync();
 
     odata[x * width + y] = tile[threadIdx.y][threadIdx.x];
 }
```
