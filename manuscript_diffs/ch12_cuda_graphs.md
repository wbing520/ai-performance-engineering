# Chapter 12 — `setCondition` helper

Replace the reduction barrier with cooperative groups:

```diff
-__global__ void setCondition(...) {
-    __shared__ float sdata[32];
-    ...
-    __syncthreads();
+__global__ void setCondition(...) {
+    __shared__ float sdata[32];
+    cg::thread_block block = cg::this_thread_block();
+    ...
+    block.sync();
     for (int s = 16; s > 0; s >>= 1) {
         if (tid < s) {
             sdata[tid] += sdata[tid + s];
         }
-        __syncthreads();
+        block.sync();
     }
 }
```
