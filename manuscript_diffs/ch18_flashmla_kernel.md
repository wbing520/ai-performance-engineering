# Chapter 18 — `flashmla_decode_kernel`

Swap loader/compute barriers to cooperative groups (page ≈ 780 if code excerpted):

```diff
 extern __shared__ half shared_mem[];
 half* shared_query = shared_mem;
 half* shared_scores = shared_query + head_dim;
 
 const int tid = threadIdx.x;
 const int lane_id = tid % WARP_SIZE;
+
+cg::thread_block block = cg::this_thread_block();
 ...
-__syncthreads();
+block.sync();
 ...
-__syncthreads();
+block.sync();
 ...
-__syncthreads();
+block.sync();
```

If the ThunderMLA mega-kernel listing is included, add the same `cg::thread_block block = cg::this_thread_block();` declaration near the top and replace its residual `__syncthreads()` calls with `block.sync()`.
