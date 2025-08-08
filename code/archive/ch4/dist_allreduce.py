#!/usr/bin/env python
import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Multi-node all-reduce benchmark")
    parser.add_argument("--data-size", type=int, default=1024*1024*100,  # 100M floats ≈ 400 MB
                        help="Number of float elements in the tensor")
    args = parser.parse_args()

    # Initialize default ProcessGroup (env:// uses env vars like MASTER_ADDR/PORT, etc.)
    dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Allocate a large tensor on GPU for all-reduce
    tensor = torch.ones(args.data_size, dtype=torch.float32, device="cuda")

    # Synchronize and measure all-reduce
    dist.barrier()
    if rank == 0:
        start = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        elapsed = time.time() - start
        mb = args.data_size * 4 / 1e6
        print(f"Rank0: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms  "
              f"({mb/elapsed/1e3:.1f} GB/s)")
        # Example expected output with Gloo: ~200 ms (2 GB/s) for 400 MB
        # With NCCL and RDMA: ~4 ms (100 GB/s)

    # Verify correctness (each element should equal world_size after SUM)
    if rank == 0:
        assert torch.allclose(tensor[0], torch.tensor(float(world_size))), "Result incorrect"

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
