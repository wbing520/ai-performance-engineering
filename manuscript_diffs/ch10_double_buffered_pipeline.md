# Chapter 10 — double_buffered_pipeline.cu (replace block-wide barriers)

```diff
-        if (row < M && (t + threadIdx.x) < K) {
-            sA[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
-        } else {
-            sA[threadIdx.y][threadIdx.x] = 0.0f;
-        }
     
-        if ((t + threadIdx.y) < K && col < N) {
-            sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
-        } else {
-            sB[threadIdx.y][threadIdx.x] = 0.0f;
-        }
-        
-        __syncthreads();
+        cta.sync();
        
        // Compute using the tile loaded in shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        
-        __syncthreads();
+        cta.sync();
```

> Update the Chapter 10 double-buffered example block beginning “// double_buffered_pipeline.cu” on page ~385.
