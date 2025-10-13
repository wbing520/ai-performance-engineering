"""Profiling helper for tiny causal LM training (Chapter 13).

Implements guidance:
- Tiny HF model for quick runtimes.
- Warmup loop outside the profiler.
- Optional AMP/fused optimizer usage.
- CUDA Graph capture compatible data.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
BATCH = 2
WARMUP = 2
PROFILE_STEPS = 3


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=device.type == "cuda")

    texts = ["DeepSeek is great." for _ in range(BATCH)]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].clone()

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()

    model.train()
    for _ in range(WARMUP):
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(**batch, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(PROFILE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(**batch, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    prof.export_chrome_trace("deepseek_v3_trace.json")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    hta_dir = "hta_traces"
    os.makedirs(hta_dir, exist_ok=True)
    with open(os.path.join(hta_dir, "rank_0.json"), "w") as f:
        json.dump(json.loads(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)), f)


if __name__ == "__main__":
    main()
