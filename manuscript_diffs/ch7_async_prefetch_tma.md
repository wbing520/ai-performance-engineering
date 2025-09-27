# Chapter 7 — `async_prefetch_tma.cu`

Replace the `__syncthreads()` barriers in the printed listing (page ≈ 279) with cooperative-groups synchronization:

```diff
-#include <cuda_runtime.h>
-#include <iostream>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+#include <iostream>
+
+namespace cg = cooperative_groups;
 
-__device__ void processTile(const float* tile) {
-    __syncthreads();
-    __shared__ float sum;
-    if (threadIdx.x == 0) sum = 0.0f;
-    __syncthreads();
+__device__ void processTile(const float* tile) {
+    cg::thread_block block = cg::this_thread_block();
+    block.sync();
+    __shared__ float sum;
+    if (threadIdx.x == 0) sum = 0.0f;
+    block.sync();
     for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
         atomicAdd(&sum, tile[i]);
     }
-    __syncthreads();
+    block.sync();
 }
 
 __global__ void kernelWithAsyncCopy(const float* __restrict__ global_ptr,
                                    int nTiles) {
     __shared__ float tile0[TILE_SIZE];
     __shared__ float tile1[TILE_SIZE];
     float* tiles[2] = { tile0, tile1 };
+
+    cg::thread_block block = cg::this_thread_block();
 
     for (int t = 0; t < nTiles; ++t) {
         ...
-        __syncthreads();
+        block.sync();
 
         processTile(tiles[t % 2]);
-        __syncthreads();
+        block.sync();
     }
 }
```

> Applies to the “Async Prefetch with TMA” code block in Chapter 7.
