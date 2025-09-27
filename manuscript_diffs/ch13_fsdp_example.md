# Chapter 13 — `fsdp_example.py`

Page ≈ 384 updates:

```diff
-from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
-from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
+from torch.distributed.fsdp import (
+    FullyShardedDataParallel as FSDP,
+    MixedPrecision,
+    BackwardPrefetch,
+    ShardingStrategy,
+)
+from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
```

```diff
-    fsdp_model = FSDP(model)
+    mixed_precision_policy = MixedPrecision(
+        param_dtype=torch.float16,
+        reduce_dtype=torch.float16,
+        buffer_dtype=torch.float16,
+    )
+
+    auto_wrap_policy = functools.partial(
+        transformer_auto_wrap_policy,
+        transformer_layer_cls={TransformerBlock},
+    )
+
+    fsdp_model = FSDP(
+        model,
+        auto_wrap_policy=auto_wrap_policy,
+        mixed_precision=mixed_precision_policy,
+        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
+        sharding_strategy=ShardingStrategy.FULL_SHARD,
+        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
+        sync_module_states=True,
+    )
```

```diff
-    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
+    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```
