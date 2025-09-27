# Chapter 19 — `token_precision_switching.py`

Page ≈ 642 updates:

```diff
-class PrecisionLevel(Enum):
-    FP32 = "fp32"
-    FP16 = "fp16"
-    INT8 = "int8"
+class PrecisionLevel(Enum):
+    FP32 = "fp32"
+    FP16 = "fp16"
+    BF16 = "bf16"
+    FP8 = "fp8"
+    INT8 = "int8"
+    INT4 = "int4"
```

```diff
-class ConfidenceMetrics:
-    pass
+@dataclass
+class ConfidenceMetrics:
+    max_probability: float
+    entropy: float
+    logit_variance: float
+    logit_max_diff: float
+    temperature_adjusted: float
+
+    @property
+    def confidence_score(self) -> float:
+        prob_score = self.max_probability
+        entropy_score = max(0, 1.0 - self.entropy / 4.0)
+        logit_score = min(1.0, self.logit_max_diff / 10.0)
+        return 0.5 * prob_score + 0.3 * entropy_score + 0.2 * logit_score
```

```diff
-    def __init__(self, model, initial_precision=PrecisionLevel.FP16):
-        self.model = model
-        self.current_precision = initial_precision
-        self.confidence_threshold_high = 0.8
-        self.confidence_threshold_low = 0.5
-        self.precision_history = []
+    def __init__(self, model: torch.nn.Module, initial_precision: PrecisionLevel = PrecisionLevel.FP16):
+        self.model = model
+        self.current_precision = initial_precision
+        self.confidence_threshold_high = 0.9
+        self.confidence_threshold_low = 0.6
+        self.entropy_threshold_low = 0.5
+        self.entropy_threshold_high = 2.0
+        self.logit_diff_threshold = 2.0
+        self.precision_history = []
+        self.switch_count = 0
+        self.quality_metrics = []
+        self.executor = ThreadPoolExecutor(max_workers=2)
+        self.switching_enabled = True
+        self.conservative_mode = False
```

```diff
-    def decide_precision(self, confidence, current_precision):
-        if confidence > self.confidence_threshold_high:
-            return PrecisionLevel.INT8
-        if confidence < self.confidence_threshold_low:
-            return PrecisionLevel.FP16
-        return current_precision
+    def decide_precision(self, confidence: ConfidenceMetrics, current_precision: PrecisionLevel) -> PrecisionLevel:
+        if not self.switching_enabled:
+            return current_precision
+
+        if confidence.confidence_score >= self.confidence_threshold_high and confidence.entropy <= self.entropy_threshold_low:
+            if current_precision == PrecisionLevel.FP32:
+                return PrecisionLevel.BF16
+            if current_precision == PrecisionLevel.BF16:
+                return PrecisionLevel.FP8
+            if current_precision == PrecisionLevel.FP8:
+                return PrecisionLevel.INT8
+            return PrecisionLevel.INT8
+
+        if confidence.confidence_score <= self.confidence_threshold_low or confidence.entropy >= self.entropy_threshold_high:
+            if current_precision in (PrecisionLevel.INT4, PrecisionLevel.INT8):
+                return PrecisionLevel.FP8
+            if current_precision == PrecisionLevel.FP8:
+                return PrecisionLevel.BF16
+            return PrecisionLevel.FP16
+
+        return current_precision
```
