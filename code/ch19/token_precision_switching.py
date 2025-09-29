"""Chapter 19: Token precision switching and cache quantization helpers."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from hqq.core.quantize import Quantizer as HQQQuantizer
    HQQ_AVAILABLE = True
except Exception:
    HQQ_AVAILABLE = False


class PrecisionLevel(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ConfidenceMetrics:
    max_probability: float
    entropy: float
    logit_max_diff: float

    @property
    def confidence_score(self) -> float:
        entropy_score = max(0.0, 1.0 - self.entropy / 4.0)
        diff_score = min(1.0, self.logit_max_diff / 10.0)
        return 0.6 * self.max_probability + 0.4 * entropy_score * diff_score


class TokenPrecisionController:
    def __init__(self, model: torch.nn.Module, precision: PrecisionLevel = PrecisionLevel.FP16) -> None:
        self.model = model
        self.current_precision = precision
        self.threshold_high = 0.9
        self.threshold_low = 0.6
        self.switch_count = 0
        self.history: List[PrecisionLevel] = []

    @staticmethod
    def _confidence(logits: torch.Tensor, temperature: float = 1.0) -> ConfidenceMetrics:
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        max_prob = float(probs.max())
        log_probs = F.log_softmax(scaled, dim=-1)
        entropy = float(-(probs * log_probs).sum())
        top2 = torch.topk(scaled, k=2).values
        diff = float(top2[0] - top2[1]) if top2.numel() == 2 else 0.0
        return ConfidenceMetrics(max_prob, entropy, diff)

    def _choose_precision(self, metrics: ConfidenceMetrics) -> PrecisionLevel:
        score = metrics.confidence_score
        if score > self.threshold_high and self.current_precision == PrecisionLevel.FP16:
            return PrecisionLevel.INT8
        if score < self.threshold_low and self.current_precision == PrecisionLevel.INT8:
            return PrecisionLevel.FP16
        return self.current_precision

    def _cast_logits(self, logits: torch.Tensor, precision: PrecisionLevel) -> torch.Tensor:
        if precision == PrecisionLevel.FP32:
            return logits.float()
        if precision == PrecisionLevel.FP16:
            return logits.half()
        if precision == PrecisionLevel.BF16:
            return logits.bfloat16()
        if precision == PrecisionLevel.INT8:
            scale = logits.abs().max() / 127.0
            quant = torch.clamp((logits / scale).round(), -128, 127)
            return quant * scale
        if precision == PrecisionLevel.INT4:
            scale = logits.abs().max() / 7.0
            quant = torch.clamp((logits / scale).round(), -8, 7)
            return quant * scale
        return logits

    def generate(self, input_ids: torch.Tensor, max_length: int = 20, temperature: float = 1.0) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        tokens = input_ids.clone()
        stats: List[Dict[str, float]] = []
        for step in range(max_length):
            with torch.no_grad():
                logits = self.model(tokens).logits[0, -1, :]
                logits = self._cast_logits(logits, self.current_precision)
                metrics = self._confidence(logits, temperature)
                next_precision = self._choose_precision(metrics)
                if next_precision != self.current_precision:
                    self.switch_count += 1
                self.current_precision = next_precision
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                stats.append({"confidence": metrics.confidence_score, "precision": self.current_precision.value})
                if next_token.item() == 0:
                    break
        return tokens, stats


class DynamicQuantizedCache:
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _memory_ratio(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        device = torch.cuda.current_device()
        used = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return used / total

    def maybe_quantize(self, layers: List[object], policy: str = "conservative") -> None:
        if self._memory_ratio() < self.threshold:
            return
        nbits = 8 if policy == "conservative" else 4
        for layer in layers:
            for attr in ("key_cache", "value_cache"):
                tensor = getattr(layer, attr, None)
                if tensor is None:
                    continue
                if not HQQ_AVAILABLE:
                    scale = tensor.abs().max() / (127 if nbits == 8 else 7)
                    quant = torch.clamp((tensor / scale).round(), -128, 127) * scale
                    setattr(layer, attr, quant)
                else:
                    future = self.executor.submit(HQQQuantizer.quantize, tensor, nbits=nbits, optimize=False)
                    try:
                        quant, meta = future.result(timeout=2.0)
                        setattr(layer, attr, quant)
                        setattr(layer, f"{attr}_meta", meta)
                    except Exception as exc:
                        logger.warning("Quantization failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    class ToyModel(torch.nn.Module):
        def __init__(self, vocab=1000, dim=512) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.linear = torch.nn.Linear(dim, vocab)
        def forward(self, input_ids: torch.Tensor):
            x = self.embed(input_ids)
            logits = self.linear(x)
            return type("Output", (), {"logits": logits})()

    model = ToyModel()
    controller = TokenPrecisionController(model)
    generated, stats = controller.generate(torch.randint(1, 100, (1, 8)))
    print("Generated:", generated.tolist())
    print("Switches:", controller.switch_count)
    print("Stats:", stats[:5])

    class MockLayer:
        def __init__(self):
            self.key_cache = torch.randn(4, 128)
            self.value_cache = torch.randn(4, 128)

    cache = DynamicQuantizedCache(threshold=0.0)
    cache.maybe_quantize([MockLayer(), MockLayer()], policy="conservative")


if __name__ == "__main__":
    main()
