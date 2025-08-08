# Atomic Work Queue Example

This example demonstrates dynamic work distribution using atomic counters and batching techniques.

## Features

- Shows before/after batching approaches for atomic work queues
- Demonstrates warp-level batching to reduce atomic contention
- Uses L2-cache atomics for fast work distribution
- Implements dynamic work termination

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- **Before Batching**: Each thread performs one atomic operation, leading to high contention
- **After Batching**: Each warp claims a batch of 32 work items, reducing atomic contention
- **L2-Cache Atomics**: Modern GPUs provide fast atomic operations in L2 cache
- **Dynamic Termination**: Warps exit when no more work is available
- **Warp-Level Coordination**: Uses `__shfl_sync` to broadcast work indices within warps

## Performance Benefits

- Can achieve 2x speedup in extreme imbalance cases
- Typically provides 10-30% improvement for moderate imbalance
- Eliminates SM idle time by ensuring all warps stay busy
- Reduces atomic contention through batching
