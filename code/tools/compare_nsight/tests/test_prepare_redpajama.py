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
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from scripts.prepare_alpaca import download_if_missing


def test_prepare_sample(tmp_path):
    vocabulary_path = tmp_path / "tokenizer.json"
    download_if_missing(
        vocabulary_path, "https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer.json"
    )
    tokenizer_path = tmp_path / "tokenizer_config.json"
    download_if_missing(
        tokenizer_path, "https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer_config.json"
    )
    with open(tmp_path / "lit_config.json", "w") as f:
        json.dump({"block_size": 2048}, f)

    sample_path = tmp_path / "sample"
    source_path = sample_path / "source"
    dest_path = sample_path / "dest"

    source_path.mkdir(parents=True)

    sample = {"meta": {"some": "info"}, "text": "some text"}

    jsonl_sample = "\n".join([json.dumps(el) for el in [sample] * 2])

    import scripts.prepare_redpajama as prepare_redpajama

    for filename in prepare_redpajama.filenames_sample:
        with open(source_path / filename, "w") as f:
            f.write(jsonl_sample)

    prepare_redpajama.prepare(source_path=source_path, checkpoint_dir=tmp_path, destination_path=dest_path)

    bin_files = [el.replace(".jsonl", "_0000000000.bin") for el in prepare_redpajama.filenames_sample]

    assert set(os.listdir(dest_path)) == set(bin_files)

    from lit_gpt import Tokenizer
    from lit_gpt.packed_dataset import PackedDataset

    tokenizer = Tokenizer(tmp_path)

    # artificially set block_size to fit the text
    block_size = len(tokenizer.encode("some text"))

    for filename in bin_files:
        filenames = [os.path.join(dest_path, filename)]
        dataset = PackedDataset(filenames=filenames, n_chunks=1, block_size=block_size, shuffle=False)
        dataset_iter = iter(dataset)
        assert tokenizer.decode(next(dataset_iter)) == "some text"
        assert tokenizer.decode(next(dataset_iter)) == "some text"


def test_prepare_full(tmp_path):
    vocabulary_path = tmp_path / "tokenizer.json"
    download_if_missing(
        vocabulary_path, "https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer.json"
    )
    tokenizer_path = tmp_path / "tokenizer_config.json"
    download_if_missing(
        tokenizer_path, "https://huggingface.co/stabilityai/stablelm-base-alpha-3b/raw/main/tokenizer_config.json"
    )
    with open(tmp_path / "lit_config.json", "w") as f:
        json.dump({"block_size": 2048}, f)

    full_path = tmp_path / "full"
    source_path = full_path / "source"
    dest_path = full_path / "dest"

    source_path.mkdir(parents=True)

    sample = {"meta": {"some": "info"}, "text": "some text"}

    jsonl_sample = "\n".join([json.dumps(el) for el in [sample] * 2])

    import scripts.prepare_redpajama as prepare_redpajama

    arxiv_file = source_path / "arxiv" / "arxiv_0.jsonl"
    arxiv_file.parent.mkdir(parents=True)
    with open(arxiv_file, "w") as f:
        f.write(jsonl_sample)

    import zstandard as zstd

    cc_file = source_path / "common_crawl" / "cc_0.jsonl"
    cc_file.parent.mkdir(parents=True)
    with zstd.open(cc_file, "wt", encoding="utf-8") as f:
        f.write(jsonl_sample)

    filename_sets = {"arxiv": "arxiv/arxiv*", "common_crawl": "common_crawl/*"}

    with mock.patch.object(prepare_redpajama, "filename_sets", filename_sets):
        prepare_redpajama.prepare(
            source_path=source_path, checkpoint_dir=tmp_path, destination_path=dest_path, sample=False
        )

        all_names = prepare_redpajama.filename_sets.keys()
        bin_files = [el + "_0000000000.bin" for el in all_names]

    assert set(os.listdir(dest_path)) == set(bin_files)

    from lit_gpt import Tokenizer
    from lit_gpt.packed_dataset import PackedDataset

    tokenizer = Tokenizer(tmp_path)

    # artificially set block_size to fit the text
    block_size = len(tokenizer.encode("some text"))

    filenames = [os.path.join(dest_path, el) for el in bin_files]

    for filename in filenames:
        dataset = PackedDataset(filenames=[filename], n_chunks=1, block_size=block_size, shuffle=False)
        dataset_iter = iter(dataset)
        assert tokenizer.decode(next(dataset_iter)) == "some text"
        assert tokenizer.decode(next(dataset_iter)) == "some text"


def test_cli():
    cli_path = Path(__file__).parent.parent / "scripts" / "prepare_redpajama.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert 'Prepare the "Red Pajama"' in output

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
