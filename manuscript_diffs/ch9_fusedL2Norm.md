# Chapter 9 — `fusedL2Norm` family

For the L2-normalization listings, replace block barriers and include cooperative_groups:

```diff
-#include <cuda_runtime.h>
-#include <stdio.h>
-#include <math.h>
+#include <cuda_runtime.h>
+#include <cooperative_groups.h>
+#include <stdio.h>
+#include <math.h>
+
+namespace cg = cooperative_groups;
 
 __global__ void computeNorms(...) {
     extern __shared__ float sdata[];
+    cg::thread_block block = cg::this_thread_block();
     ...
-    __syncthreads();
+    block.sync();
     ...
-        __syncthreads();
+        block.sync();
 }
 
 __global__ void fusedL2Norm(...) {
     extern __shared__ float sdata[];
+    cg::thread_block block = cg::this_thread_block();
     ...
-    __syncthreads();
+    block.sync();
     ...
-        __syncthreads();
+        block.sync();
 }
 
 __global__ void fusedL2NormOptimized(...) {
     extern __shared__ float sdata[];
     const int tid = threadIdx.x;
     const int block_size = blockDim.x;
+    cg::thread_block block = cg::this_thread_block();
     ...
-    __syncthreads();
+    block.sync();
-    if (block_size >= 512) { ... __syncthreads(); }
+    if (block_size >= 512) { ... block.sync(); }
     // repeat for other stages
 }
```
