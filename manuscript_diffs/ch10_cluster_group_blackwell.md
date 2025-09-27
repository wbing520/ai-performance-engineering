# Chapter 10 — `cluster_group_blackwell.cu`

Update the reduction to use `cta.sync()`:

```diff
 sdata[threadIdx.x] = sum;
-__syncthreads();
+cta.sync();
 for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
     if (threadIdx.x < stride) {
         sdata[threadIdx.x] += sdata[threadIdx.x + stride];
     }
-    __syncthreads();
+    cta.sync();
 }
```
