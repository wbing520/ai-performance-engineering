# Chapter 20 — `ai_kernel_generator.py`

Page ≈ 672 updates:

```diff
-class OptimizationTarget(Enum):
-    LATENCY = "latency"
-    THROUGHPUT = "throughput"
-    MEMORY = "memory"
+class OptimizationTarget(Enum):
+    LATENCY = "latency"
+    THROUGHPUT = "throughput"
+    MEMORY = "memory"
+    POWER = "power"
```

```diff
-@dataclass
-class KernelCandidate:
-    code: str
-    compile_success: bool
-    runtime_ms: float
-    memory_mb: float
-    correctness_score: float
-    iteration: int
-
-    @property
-    def is_valid(self) -> bool:
-        return self.compile_success and self.correctness_score > 0.95
-
-    @property
-    def performance_score(self) -> float:
-        if not self.compile_success:
-            return 0.0
-        runtime_score = max(0, 1.0 - self.runtime_ms / 50.0)
-        memory_score = max(0, 1.0 - self.memory_mb / 512.0)
-        return self.correctness_score * (0.6 * runtime_score + 0.4 * memory_score)
+@dataclass
+class KernelCandidate:
+    code: str
+    compile_success: bool
+    runtime_ms: float
+    memory_mb: float
+    correctness_score: float
+    iteration: int
+    feedback: str = ""
+
+    @property
+    def is_valid(self) -> bool:
+        return self.compile_success and self.correctness_score > 0.85
+
+    @property
+    def performance_score(self) -> float:
+        if not self.compile_success:
+            return 0.0
+        runtime_score = max(0, 1.0 - self.runtime_ms / 100.0)
+        memory_score = max(0, 1.0 - self.memory_mb / 1000.0)
+        return self.correctness_score * (0.7 * runtime_score + 0.3 * memory_score)
```

```diff
-        self.kernel_templates = {
-            "attention": "...",
-        }
+        self.kernel_templates = self._load_kernel_templates()
+
+    def _load_kernel_templates(self) -> Dict[str, str]:
+        return {
+            "attention": """
+__global__ void attention_kernel(
+    const float* __restrict__ query,
+    const float* __restrict__ key,
+    const float* __restrict__ value,
+    float* __restrict__ output,
+    const int seq_len,
+    const int head_dim,
+    const float scale
+) {
+    int tid = blockIdx.x * blockDim.x + threadIdx.x;
+    int head_idx = blockIdx.y;
+
+    if (tid >= seq_len || head_idx >= gridDim.y) return;
+
+    extern __shared__ float shared_mem[];
+    float* shared_query = shared_mem;
+    float* shared_scores = shared_query + head_dim;
+
+    if (threadIdx.x < head_dim) {
+        shared_query[threadIdx.x] = query[head_idx * head_dim + threadIdx.x];
+    }
+    __syncthreads();
+
+    float max_score = -1e9f;
+    for (int pos = 0; pos < seq_len; pos++) {
+        float score = 0.0f;
+        for (int d = 0; d < head_dim; d++) {
+            score += shared_query[d] * key[pos * head_dim + d];
+        }
+        score *= scale;
+        shared_scores[pos] = score;
+        max_score = fmaxf(max_score, score);
+    }
+
+    float sum_exp = 0.0f;
+    for (int pos = 0; pos < seq_len; pos++) {
+        shared_scores[pos] = expf(shared_scores[pos] - max_score);
+        sum_exp += shared_scores[pos];
+    }
+
+    if (tid < head_dim) {
+        float result = 0.0f;
+        for (int pos = 0; pos < seq_len; pos++) {
+            float weight = shared_scores[pos] / sum_exp;
+            result += weight * value[pos * head_dim + tid];
+        }
+        output[head_idx * head_dim + tid] = result;
+    }
+}
+""",
+        }
```
