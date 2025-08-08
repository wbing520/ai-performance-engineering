import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "9.0":
        return "hopper"  # H100/H200
    elif compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return {
            "name": "Hopper H100/H200",
            "compute_capability": "9.0",
            "sm_version": "sm_90",
            "memory_bandwidth": "3.35 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3", "Transformer Engine", "Dynamic Programming"]
        }
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "3.2 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    else:
        return {
            "name": "Other",
            "compute_capability": "Unknown",
            "sm_version": "Unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": []
        }
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import repeat
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pytest
import torch


@pytest.mark.parametrize(
    ("generated", "stop_tokens", "expected"),
    [
        (repeat(1), (), [1] * 8),
        ([1, 2, 3, 0], ([0],), [1, 2, 3]),
        ([1, 2, 3, 0], ([9], [2, 4], [1, 2, 3, 0]), []),
        ([1, 2, 3, 0, 0], ([0, 0, 0], [0, 0]), [1, 2, 3]),
        ([3, 1, 2], ([1, 2], [3]), []),
        ([1, 2, 3, 0, 3, 2, 1, 0], ([4, 3, 2, 1], [2, 4]), [1, 2, 3, 0, 3, 2, 1, 0]),
    ],
)
def test_generate(monkeypatch, generated, stop_tokens, expected):
    import chat.base as chat
    import generate.base as generate

    input_idx = torch.tensor([5, 3])
    max_returned_tokens = len(input_idx) + 8
    model = MagicMock()
    model.config.block_size = 100
    model.max_seq_length = 100
    it = iter(generated)

    def multinomial(*_, **__):
        out = next(it)
        return torch.tensor([out])

    monkeypatch.setattr(generate, "multinomial_num_samples_1", multinomial)
    actual = chat.generate(model, input_idx, max_returned_tokens, stop_tokens=stop_tokens)
    actual = list(actual)

    assert len(actual) == len(expected)
    if not actual:
        assert actual == expected
    else:
        for t in actual:
            assert t.dtype == torch.long
        assert torch.cat(actual).tolist() == expected


@pytest.mark.parametrize("tokenizer_backend", ["huggingface", "sentencepiece"])
def test_decode(tokenizer_backend):
    from lightning.fabric import Fabric

    import chat.base as chat

    class Tokenizer:
        backend = tokenizer_backend
        id2token = {1: "foo ", 2: "bar ", 3: "baz "}

        def decode(self, tensor: torch.Tensor) -> str:
            tensor = [tensor] if tensor.ndim == 0 else tensor
            return "".join(self.id2token[int(value)] for value in tensor)

    tokenizer_mock = Tokenizer()

    fabric = Fabric(devices=1, accelerator="cpu")

    token_stream = torch.tensor([3, 2, 1])
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.decode(fabric, tokenizer_mock, token_stream)

    assert out.getvalue() == "baz bar foo "


@patch("chat.base.input")
@pytest.mark.parametrize("stop_iteration", [KeyboardInterrupt, ""])
def test_main(mocked_input, stop_iteration, fake_checkpoint_dir, monkeypatch, tensor_like):
    import chat.base as chat

    # these values will be iteratively provided for each `input()` call
    mocked_input.side_effect = ["Hello", stop_iteration]

    config_path = fake_checkpoint_dir / "lit_config.json"
    config = {"block_size": 128, "vocab_size": 50, "n_layer": 2, "n_head": 4, "n_embd": 8, "rotary_percentage": 1}
    config_path.write_text(json.dumps(config))

    load_mock = Mock()
    load_mock.return_value = load_mock
    monkeypatch.setattr(chat, "load_checkpoint", load_mock)
    tokenizer_mock = Mock()
    tokenizer_mock.return_value.backend = "sentencepiece"
    tokenizer_mock.return_value.encode.return_value = torch.tensor([1, 2, 3])
    tokenizer_mock.return_value.decode.return_value = "foo bar baz"
    monkeypatch.setattr(chat, "Tokenizer", tokenizer_mock)
    generate_mock = Mock()
    generate_mock.return_value = torch.tensor([3, 2, 1])
    monkeypatch.setattr(chat, "generate", generate_mock)

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        chat.main(temperature=2.0, top_k=2, checkpoint_dir=fake_checkpoint_dir)

    # decoding is done per each generated item
    assert len(tokenizer_mock.return_value.decode.mock_calls) == generate_mock.return_value.numel()
    assert torch.allclose(tokenizer_mock.return_value.decode.call_args[0][0], generate_mock.return_value)
    assert generate_mock.mock_calls == [
        call(ANY, tensor_like, 128, temperature=2.0, top_k=2, stop_tokens=([tokenizer_mock.return_value.eos_id],))
    ]
    # # only the generated result is printed to stdout
    assert out.getvalue() == ">> Reply: foo bar baz\n"

    assert "'padded_vocab_size': 512, 'n_layer': 2, 'n_head': 4" in err.getvalue()


def test_cli():
    cli_path = Path(__file__).parent.parent / "chat" / "base.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Starts a conversation" in output

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == "10.0":  # Blackwell B200/B300
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.8 features
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.autotune_mode = "max-autotune"
    torch._dynamo.config.automatic_dynamic_shapes = True
