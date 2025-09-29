"""FlexDecoding showcase aligned with PyTorch 2.9 (CUDA 12.9)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import flex_attention
    HAS_FLEX = True
except (ImportError, AttributeError):
    HAS_FLEX = False


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    elapsed = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<28}: {elapsed:7.3f} ms")
    return elapsed


def _score_mod_causal(offset: torch.Tensor):
    def score_mod(score, _b, _h, q_idx, kv_idx):
        q = q_idx + offset
        return torch.where(q >= kv_idx, score, torch.neg(torch.inf_like(score)))

    return score_mod


@dataclass
class FlexDecodingConfig:
    dim: int = 256
    heads: int = 4
    max_seq_len: int = 1024
    window: int = 128


class FlexDecodingModule(torch.nn.Module):
    def __init__(self, cfg: FlexDecodingConfig):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.cfg = cfg
        self.head_dim = cfg.dim // cfg.heads

        self.q_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.register_buffer("k_cache", torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim))
        self.register_buffer("v_cache", torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim))
        self.register_buffer("offset", torch.zeros(1, dtype=torch.long))

        self.prefill_impl = None
        self.decode_impl = None

    def _compile(self, pattern: str = "causal") -> None:
        device = next(self.parameters()).device
        head_dim = self.head_dim
        heads = self.cfg.heads

        q_prefill = torch.randn(1, heads, 256, head_dim, device=device)
        kv_prefill = torch.randn_like(q_prefill)
        q_decode = torch.randn(1, heads, 1, head_dim, device=device)

        if HAS_FLEX:
            score_mod = _score_mod_causal(self.offset)

            def prefill(q, k, v):
                return flex_attention.flex_attention(q, k, v, score_mod=score_mod)

            def decode(q, k, v):
                return flex_attention.flex_attention(q, k, v, score_mod=score_mod)

            self.prefill_impl = torch.compile(prefill, mode="max-autotune", fullgraph=True)
            self.decode_impl = torch.compile(decode, mode="max-autotune", fullgraph=True)

            self.prefill_impl(q_prefill, kv_prefill, kv_prefill)
            self.decode_impl(q_decode, kv_prefill, kv_prefill)
        else:
            def prefill(q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            self.prefill_impl = torch.compile(prefill, mode="max-autotune", fullgraph=True)
            self.decode_impl = torch.compile(prefill, mode="max-autotune", fullgraph=True)

            q_prefill_eager = q_prefill.transpose(1, 2)
            kv_prefill_eager = kv_prefill.transpose(1, 2)
            self.prefill_impl(q_prefill_eager, kv_prefill_eager, kv_prefill_eager)
            self.decode_impl(q_prefill_eager[:, :1], kv_prefill_eager, kv_prefill_eager)

    def ensure_compiled(self) -> None:
        if self.prefill_impl is None or self.decode_impl is None:
            self._compile()

    def clear_cache(self, batch: int) -> None:
        if self.k_cache.shape[0] != batch:
            device = self.k_cache.device
            self.k_cache = torch.zeros(batch, self.cfg.max_seq_len, self.cfg.heads, self.head_dim, device=device)
            self.v_cache = self.k_cache.clone()
        else:
            self.k_cache.zero_()
            self.v_cache.zero_()

    def prefill(self, tokens: torch.Tensor, past: int = 0) -> torch.Tensor:
        batch, seqlen, _ = tokens.shape
        device = tokens.device
        self.ensure_compiled()

        q = self.q_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        k = self.k_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        v = self.v_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)

        self.k_cache[:, past:past + seqlen] = k
        self.v_cache[:, past:past + seqlen] = v

        if HAS_FLEX:
            out = self.prefill_impl(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.prefill_impl(q, k, v)
        return self.o_proj(out.reshape(batch, seqlen, self.cfg.dim))

    def decode(self, token: torch.Tensor, position: int) -> torch.Tensor:
        batch, _, _ = token.shape
        device = token.device
        self.ensure_compiled()

        q = self.q_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        k = self.k_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        v = self.v_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)

        self.k_cache[:, position:position + 1] = k
        self.v_cache[:, position:position + 1] = v
        past_k = self.k_cache[:, :position + 1]
        past_v = self.v_cache[:, :position + 1]

        self.offset.fill_(position)

        if HAS_FLEX:
            out = self.decode_impl(q.transpose(1, 2), past_k.transpose(1, 2), past_v.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.decode_impl(q, past_k, past_v)
        return self.o_proj(out.reshape(batch, 1, self.cfg.dim))


def jagged_batch(model: FlexDecodingModule, sequences: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    outputs: List[torch.Tensor] = []
    for seq in sequences:
        model.clear_cache(seq.shape[0])
        pref = model.prefill(seq)
        cur = pref[:, -1:, :]
        tokens = [cur]
        for step in range(3):
            nxt = model.decode(cur, seq.shape[1] + step)
            tokens.append(nxt)
            cur = nxt
        outputs.append(torch.cat(tokens, dim=1))
    return outputs


def benchmark(model: FlexDecodingModule) -> None:
    device = _device()
    torch.manual_seed(0)

    batch = 4
    seq_len = 256
    tokens = torch.randn(batch, seq_len, model.cfg.dim, device=device)
    model.ensure_compiled()

    print("\nPrefill vs decode timings")
    _benchmark("Prefill", lambda: model.prefill(tokens), iters=3)
    single = torch.randn(batch, 1, model.cfg.dim, device=device)
    _benchmark("Decode", lambda: model.decode(single, seq_len), iters=20)


def paged_attention_demo() -> None:
    print("\nPagedAttention-style block mapping")
    batch, heads, head_dim = 2, 4, 32
    block, blocks = 8, 16
    logical = torch.randn(batch, blocks * block, heads, head_dim)
    physical = torch.zeros(blocks, block, heads, head_dim)
    table = torch.randint(0, blocks, (batch, blocks))
    for b in range(batch):
        for blk in range(blocks):
            src = logical[b, blk * block : (blk + 1) * block]
            dst = table[b, blk]
            physical[dst].copy_(src)
    print(f"Logical {logical.shape} -> Physical {physical.shape}")


def main() -> None:
    device = _device()
    print("FlexDecoding example (PyTorch 2.9 / CUDA 12.9)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    cfg = FlexDecodingConfig()
    model = FlexDecodingModule(cfg).to(device)
    model.ensure_compiled()

    prompt = torch.randn(1, 32, cfg.dim, device=device)
    print("\nPrefill output shape", model.prefill(prompt).shape)
    token = torch.randn(1, 1, cfg.dim, device=device)
    print("Decode output shape", model.decode(token, 32).shape)

    sequences = [
        torch.randn(1, 16, cfg.dim, device=device),
        torch.randn(1, 32, cfg.dim, device=device),
        torch.randn(1, 64, cfg.dim, device=device),
    ]
    outs = jagged_batch(model, sequences)
    for idx, out in enumerate(outs):
        print(f"Sequence {idx}: {out.shape[1]} tokens emitted")

    if torch.cuda.is_available():
        benchmark(model)

    paged_attention_demo()

    print("\nKey takeaways:")
    print("- torch.compile can specialize prefill vs decode paths.")
    print("- FlexAttention (when available) supplies custom score mods.")
    print("- Jagged batches illustrate variable sequence handling.")
    print("- Block remapping mirrors PagedAttention KV packing.")


if __name__ == "__main__":
    main()
