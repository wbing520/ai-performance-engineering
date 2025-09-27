# Chapter 12 — `computeKernelHierarchical`

Cooperative-groups sync for the hierarchical batching example:

```diff
 __global__ void computeKernelHierarchical(...) {
     __shared__ unsigned int blockBase;
     __shared__ int blockWorkCount;
+    cg::thread_block block = cg::this_thread_block();
     ...
-        __syncthreads();
+        block.sync();
         ...
-        __syncthreads();
+        block.sync();
     }
 }
```
