# Chapter 13 — `train_deepseek_v3.py`

Page ≈ 402 updates:

```diff
-    model_name = "deepseek-ai/DeepSeek-V3"
-    tokenizer = AutoTokenizer.from_pretrained(model_name)
-    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
+    model_name = "deepseek-ai/DeepSeek-V3"
+    # Use a tiny model so the example fits on developer hardware
+    model_name = "sshleifer/tiny-gpt2"
+
+    tokenizer = AutoTokenizer.from_pretrained(model_name)
+    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
```

```diff
-    with profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
-        optimizer.zero_grad()
-        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
-        loss = outputs.loss
-        loss.backward()
-        optimizer.step()
+    with profiler.profile(
+        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
+        record_shapes=True,
+        profile_memory=True,
+        with_stack=True,
+    ) as prof:
+        with profiler.record_function("training_step"):
+            optimizer.zero_grad()
+
+            with profiler.record_function("forward"):
+                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
+                loss = outputs.loss
+
+            with profiler.record_function("backward"):
+                loss.backward()
+
+            with profiler.record_function("optimizer_step"):
+                optimizer.step()
```

```diff
-    print(prof.key_averages().table(sort_by="cuda_time_total"))
+    prof.export_chrome_trace("deepseek_v3_trace.json")
+    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
+
+    trace_dir = "./hta_traces"
+    os.makedirs(trace_dir, exist_ok=True)
+    prof.export_chrome_trace(f"{trace_dir}/rank_0.json")
```
