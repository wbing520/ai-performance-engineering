# Chapter 12 — `persistentScheduler`

Introduce a cooperative-groups barrier for the simulated work loop:

```diff
-__device__ cudaGraphExec_t g_graphExec;
-__device__ int g_workIndex = 0;
-
-__global__ void persistentScheduler(float* workData, int numTasks, int maxIterations) {
+__device__ cudaGraphExec_t g_graphExec;
+__device__ int g_workIndex = 0;
+
+__global__ void persistentScheduler(float* workData, int numTasks, int maxIterations) {
+    cg::thread_block block = cg::this_thread_block();
     while (true) {
         ...
-        for (int i = 0; i < 1000; ++i) {
-            __syncthreads();
-        }
+        for (int i = 0; i < 1000; ++i) {
+            block.sync();
+        }
     }
 }
```
