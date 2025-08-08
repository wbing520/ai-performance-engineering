import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    torch.cuda.set_device(rank)
    for i in range(5):  # simulate 5 iterations
        # BAD: re-initialize NCCL communicator each iteration (very slow!)
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:45678",
                                 world_size=world_size, rank=rank)
        # small all-reduce to simulate work
        tensor = torch.ones(1).cuda(rank)
        dist.all_reduce(tensor)
        if rank == 0:
            print(f"Iter {i} done")
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
