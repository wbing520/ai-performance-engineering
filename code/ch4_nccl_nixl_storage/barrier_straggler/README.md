# Straggler Detection Example

This example demonstrates how to use PyTorch's `torch.distributed.monitored_barrier()` to detect and log straggler processes in a distributed setup.

## Files

- `barrier_straggler_example.py`: The main script demonstrating monitored_barrier.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.11
- PyTorch nightly 2.8.0+cu13

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

On a single node with 2 GPUs:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
torchrun --nnodes=1 --nproc_per_node=2 barrier_straggler.py
```

The script simulates one rank taking longer and shows how `monitored_barrier` times out or succeeds.
