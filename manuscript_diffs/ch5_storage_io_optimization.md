# Chapter 5 — `storage_io_optimization.py`

Page ≈ 208 updates:

```diff
-def create_optimized_dataloader(dataset, batch_size=32, num_workers=4):
-    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
+def create_optimized_dataloader(dataset, batch_size=32, num_workers=8):
+    return DataLoader(
+        dataset,
+        batch_size=batch_size,
+        num_workers=num_workers,
+        pin_memory=True,
+        prefetch_factor=4,
+        persistent_workers=True,
+        shuffle=True,
+        drop_last=True,
+    )
```

```diff
-        data = data.to(device)
-        target = target.to(device)
+        data = data.to(device, non_blocking=True)
+        target = target.to(device, non_blocking=True)
@@
-        if batch_idx % 100 == 0:
-            print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
+        if batch_idx % 100 == 0:
+            print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
```

```diff
-    dataset = TensorDataset(X, y)
-    dataloader = create_optimized_dataloader(dataset)
+    dataset = TensorDataset(X, y)
+    dataloader = create_optimized_dataloader(dataset, batch_size=256, num_workers=8)
```
