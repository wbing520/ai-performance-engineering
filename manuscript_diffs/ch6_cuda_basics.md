# Chapter 6 — CUDA fundamentals

Page ≈ 244 updates:

```diff
-// my_first_kernel.cu
-#include <cuda_runtime.h>
-#include <stdio.h>
+// Architecture-specific optimizations for CUDA 12.9
+// Targets Blackwell B200/B300 (sm_100)
+// my_first_kernel.cu
+#include <cuda_runtime.h>
+#include <stdio.h>
@@
-    float *h_input = new float[N];
-    float *d_input = nullptr;
+    float *h_input = nullptr;
+    float *d_input = nullptr;
+
+    cudaMallocHost(&h_input, N * sizeof(float));
@@
-    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
-
-    delete[] h_input;
-    cudaFree(d_input);
+    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);
+
+    cudaFree(d_input);
+    cudaFreeHost(h_input);
```

```diff
-#include <cuda_runtime.h>
-#include <stdio.h>
-const int N = 1'000'000;
+#include <cuda_runtime.h>
+#include <stdio.h>
+#include <cuda/std/chrono>
+#include <nvtx3/nvToolsExt.h>
+
+const int N = 1'000'000;
@@
-    float *h_A = (float*)malloc(N * sizeof(float));
-    float *h_B = (float*)malloc(N * sizeof(float));
-    float *h_C = (float*)malloc(N * sizeof(float));
+    float* h_A = nullptr;
+    float* h_B = nullptr;
+    float* h_C = nullptr;
+    cudaMallocHost(&h_A, N * sizeof(float));
+    cudaMallocHost(&h_B, N * sizeof(float));
+    cudaMallocHost(&h_C, N * sizeof(float));
@@
-    addParallel<<<blocks, threads>>>(d_A, d_B, d_C, N);
+    nvtxRangePushA("add_parallel_kernel");
+    addParallel<<<blocks, threads>>>(d_A, d_B, d_C, N);
+    nvtxRangePop();
@@
-    free(h_A);
-    free(h_B);
-    free(h_C);
+    cudaFreeHost(h_A);
+    cudaFreeHost(h_B);
+    cudaFreeHost(h_C);
```
