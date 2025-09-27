# Chapter 4 — `after_overlap_ddp.py`

Page ≈ 162 updates:

```diff
-def train_ddp(rank, world_size, data, target):
-    dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
-
-    torch.cuda.set_device(rank)
-    model = MultiLayerNet(data.size(1)).cuda(rank)
-    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
-    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
-
-    data = data.cuda(rank)
-    target = target.cuda(rank)
-
-    output = ddp_model(data)
-    loss = nn.functional.mse_loss(output, target)
-    loss.backward()
-    optimizer.step()
-
-    dist.destroy_process_group()
+def train_ddp(rank, world_size, data, target):
+    if torch.cuda.device_count() < world_size:
+        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but {world_size} requested.", flush=True)
+        print("Falling back to single-GPU training.", flush=True)
+        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
+        model = MultiLayerNet(data.size(1)).to(device)
+        optimizer = optim.SGD(model.parameters(), lr=0.01)
+        data = data.to(device)
+        target = target.to(device)
+        output = model(data)
+        loss = nn.functional.mse_loss(output, target)
+        loss.backward()
+        optimizer.step()
+        print(f"Single-GPU training completed. Loss: {loss.item():.4f}", flush=True)
+        return
+
+    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:34568", world_size=world_size, rank=rank)
+
+    torch.cuda.set_device(rank)
+    model = MultiLayerNet(data.size(1)).cuda(rank)
+    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
+    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
+
+    data = data.cuda(rank)
+    target = target.cuda(rank)
+
+    output = ddp_model(data)
+    loss = nn.functional.mse_loss(output, target)
+    loss.backward()
+    optimizer.step()
+
+    dist.destroy_process_group()
```

```diff
-if __name__ == "__main__":
-    world_size = 2
-    inp = torch.randn(128, 1024)
-    tgt = torch.randn(128, 1)
-    mp.spawn(train_ddp, args=(world_size, inp, tgt), nprocs=world_size)
+if __name__ == "__main__":
+    print(f"Starting DDP training with {torch.cuda.device_count()} GPU(s)", flush=True)
+    world_size = min(2, torch.cuda.device_count())
+    inp = torch.randn(128, 1024)
+    tgt = torch.randn(128, 1)
+
+    if world_size == 1:
+        print("Running single-GPU training", flush=True)
+        train_ddp(0, 1, inp, tgt)
+    else:
+        print("Running multi-GPU training", flush=True)
+        mp.spawn(train_ddp, args=(world_size, inp, tgt), nprocs=world_size)
```
