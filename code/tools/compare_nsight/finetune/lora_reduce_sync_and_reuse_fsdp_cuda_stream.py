# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.loggers import CSVLogger
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor

# UNCOMMENT THIS IF YOU WANT TO USE torch.compile
# import os
# os.environ["TORCH_LOGS"] = "recompiles"
# torch.compiler.allow_in_graph(sys.audit)
# torch.set_float32_matmul_precision("high")
# # Enable detailed logging for TorchDynamo.
# import torch._dynamo.config as dynamo_cfg
# dynamo_cfg.verbose = True
# dynamo_cfg.suppress_errors = False
# # Enable debugging for TorchInductor.
# import torch._inductor.config as inductor_cfg
# inductor_cfg.debug = True

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.args import EvalArgs, IOArgs, TrainArgs
from lit_gpt.lora_reduce_sync import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils_reduce_sync import (
    CLI,
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
)
from scripts.prepare_alpaca import generate_prompt


def setup(
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = False,
    lora_value: bool = True,
    lora_projection: bool = False,
    lora_mlp: bool = False,
    lora_head: bool = False,
    io: IOArgs = IOArgs(
        train_data_dir=Path("data/alpaca"),
        val_data_dir=Path("data/alpaca"),
        checkpoint_dir=Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
        out_dir=Path("out/lora/alpaca"),
    ),
    train: TrainArgs = TrainArgs(
        save_interval=100, # was 1000
        log_interval=1,
        global_batch_size=32,
        micro_batch_size=1,
        lr_warmup_steps=10,  # was 100
        epochs=1, # was 5
        epoch_size=100,  # was 1000
        learning_rate=3e-4,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
    profile_only: Optional[str] = None
) -> None:
    print(locals())
    precision = precision or get_default_supported_precision(training=True)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    logger = CSVLogger(io.out_dir.parent, io.out_dir.name, flush_logs_every_n_steps=train.log_interval)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)

    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    fabric.launch(
        main,
        devices,
        Config.from_name(
            name=io.checkpoint_dir.name,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        ),
        io,
        train,
        eval,
        profile_only
    )


def main(fabric: L.Fabric, devices: int, config: Config, io: IOArgs, train: TrainArgs, eval: EvalArgs, profile_only) -> None:
    validate_args(io, train, eval)

    steps_per_epoch = train.epoch_size // devices // train.batch_size(devices)
    lr_max_steps = train.epochs * steps_per_epoch

    check_valid_checkpoint_dir(io.checkpoint_dir)

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(io.out_dir, exist_ok=True)

    train_data = torch.load(io.train_data_dir / "train.pt")
    val_data = torch.load(io.val_data_dir / "test.pt")

    checkpoint_path = io.checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    # [DOES NOT CURRENTLY WORK] UNCOMMENT THIS TO Enable maximum autotuning for TorchInductor if desired
    #   NOTE: THIS IS CURRENTLY FAILING DUE TO sys.audit() not supported by Torch 2.6 graph breaks
    #         https://github.com/pytorch/pytorch/issues/133185
    # import torch._inductor.config as inductor_cfg
    # inductor_cfg.max_autotune = True  # Enables max autotuning, which may increase compile time but can give better optimized kernels
    # # Wrap the model with torch.compile.
    # model = torch.compile(model, mode="max-autotune")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.PagedAdamW
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params, lr=train.learning_rate, weight_decay=train.weight_decay, betas=(train.beta1, train.beta2)
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)

    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    results_dir = Path("./profile_results") #Path.home() / "profile_results"
    results_dir.mkdir(exist_ok=True)

    train_time = time.perf_counter()
    fit_fn = lambda: fit(fabric, model, optimizer, scheduler, train_data, val_data, devices, io, train, eval)
    
    if profile_only == "torch":
        profiler_activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True, activities=profiler_activities) as p:
            fit_fn()

        prefix = f"{int(time.time())}"
        p.export_chrome_trace(str(results_dir / f"{prefix}_trace.json.gz"))
        p.export_memory_timeline(str(results_dir / f"{prefix}_memory.html"))
        return
    elif profile_only == "nsys":
        torch.cuda.cudart().cudaProfilerStart()
        fit_fn()
        torch.cuda.cudart().cudaProfilerStop()
        return
    elif profile_only == "memory":
        torch.cuda.memory._record_memory_history()
        fit_fn()
        torch.cuda.memory._dump_snapshot(str(results_dir / f"{int(time.time())}_memory.pickle"))
        return

    assert profile_only is None, profile_only
    
    fit_fn()
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = io.out_dir / "lit_model_lora_finetuned.pth"
    save_lora_checkpoint(fabric, model, save_path)


def fit(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: List[Dict],
    val_data: List[Dict],
    devices: int,
    io: IOArgs,
    train: TrainArgs,
    eval: EvalArgs,
) -> None:
    tokenizer = Tokenizer(io.checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(train_data)
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    # Sanity check using validation
    validate(fabric, model, val_data, tokenizer, dataclasses.replace(eval, max_iters=2), train)

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    for iter_num in range(1, train.max_iters(devices) + 1):
        # Mark beginning of new CUDA Graph step
        torch.compiler.cudagraph_mark_step_begin()

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, train.micro_batch_size, train.max_seq_length,
            longest_seq_ix if iter_num == 1 else None
        )

        is_accumulating = (iter_num % train.gradient_accumulation_iters(devices)) != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # Shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            # Optionally free up logits to avoid spikes
            del logits
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        # Before performing the optimizer step,
        # run any FSDP-specific post-backward processing on parameters.
        if not is_accumulating:
            # Attempt to retrieve the FSDP stream.
            try:
                import torch.distributed.fsdp as fsdp
                if isinstance(model, fsdp.FullyShardedDataParallel) and hasattr(model, "_fsdp_state"):
                    fsdp_state = model._fsdp_state
                    # Attempt to retrieve the forward prefetch stream.
                    stream = getattr(fsdp_state, "forward_prefetch_stream", None)
                    if stream is None:
                        raise AttributeError("The _fsdp_state.forward_prefetch_stream attribute is missing or None.")
                    print("FOUND THE FSDP STREAM:", stream)
                else:
                    raise AttributeError("Model is not wrapped in FSDP or missing _fsdp_state.")
            except Exception as e:
                # If FSDP is not set up or the attribute is missing, fall back.
                stream = torch.cuda.current_stream(device=fabric.device)
                print("FALLING BACK TO CURRENT STREAM:", stream, "; Reason:", e)

            # Iterate over model parameters (or handles) to clear and prepare gradients.
            for param in model.parameters():
                # Clear pending post-backward hook state if present.
                if hasattr(param, "_post_backward_hook_state"):
                    param._post_backward_hook_state = None
                # Mark that post-backward processing has been performed.
                param._post_backward_called = True

                # Execute preparation on a dedicated CUDA stream.
                with torch.cuda.stream(stream):
                    if hasattr(param, "_cls_prepare_gradient_for_optim"):
                        # Call internal method to prepare the gradient.
                        param._cls_prepare_gradient_for_optim(param)
                    # Optionally, add any additional logic (e.g. check if grad is None).

            # Now perform optimizer and scheduler steps.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1

        total_lengths += input_ids.numel()
        if iter_num % train.log_interval == 0:
            loss_item = loss.item()  # This call synchronizes device-to-host.
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num,
                samples=iter_num * train.micro_batch_size,
                lengths=total_lengths
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} | step {step_count}: loss {loss_item:.4f}, iter time: {(t1 - iter_t0) * 1000:.2f} ms"
                f"{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, eval, train)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            fabric.barrier()
        if not is_accumulating and step_count % train.save_interval == 0:
            checkpoint_path = io.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)

# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, eval: EvalArgs, train: TrainArgs
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval.max_iters)
    for k in range(eval.max_iters):
        input_ids, targets = get_batch(fabric, val_data, train.micro_batch_size, train.max_seq_length)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model, encoded, max_returned_tokens=len(encoded) + eval.max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric,
    data: List[Dict],
    micro_batch_size: int,
    max_seq_length: Optional[int],
    longest_seq_ix: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    # Truncate if needed
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(io: IOArgs, train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [
        (io, ["checkpoint_dir", "train_data_dir", "val_data_dir"]),
        (train, ["epoch_size", "epochs"]),
        (eval, ["max_new_tokens"]),
    ]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(setup)
