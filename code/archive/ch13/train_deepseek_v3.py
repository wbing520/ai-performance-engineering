#!/usr/bin/env python3
"""
Chapter 13: Profiling, Tuning, and Scaling PyTorch
DeepSeek-V3 Training Example with Comprehensive Profiling

This example demonstrates:
- PyTorch Profiler usage with NVTX markers
- System profiling with Nsight Systems
- Kernel analysis with Nsight Compute
- CPU profiling with Linux perf
- PyTorch Compiler (torch.compile)
- CUDA Streams for overlapping computation and communication
- CUDA Graphs for reducing kernel launch overhead
- Memory optimization techniques
- FSDP for distributed training
- Performance benchmarking and CI integration
"""

import time
import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
import functools
import os

# Set up model and data
model_name = "deepseek-ai/DeepSeek-V3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_size = 4
input_texts = ["DeepSeek is great."] * batch_size
enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()  # For LM training, labels are the inputs (next-token prediction)

def warm_up_model():
    """Warm up the model to avoid capturing one-time setup costs"""
    print("Warming up model...")
    for _ in range(5):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    print("Warm-up complete")

def profile_with_pytorch_profiler():
    """Profile using PyTorch Profiler with NVTX markers"""
    print("\n=== PyTorch Profiler Analysis ===")
    
    with profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,  # record tensor shapes
        profile_memory=True,  # track GPU memory usage per op
        with_stack=True,  # enable stack tracing
        with_flops=True  # capture FLOPs counters
    ) as prof:
        with profiler.record_function("train_step"):
            # Forward pass
            torch.cuda.nvtx.range_push("forward")
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            torch.cuda.nvtx.range_pop()  # end of forward
            
            # Backward pass and optimization
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_push("optimizer_step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()  # end of optimizer_step
            optimizer.zero_grad()
            torch.cuda.nvtx.range_pop()  # end of backward
    
    # Print top operations by CUDA time
    print("\nTop 10 operations by CUDA execution time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10,
        fields=["self_cuda_time_total", "calls"]
    ))

def compare_eager_vs_compiled():
    """Compare eager mode vs compiled execution"""
    print("\n=== Eager vs Compiled Mode Comparison ===")
    
    # Make runs deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Eager mode timing
    optimizer.zero_grad()
    torch.cuda.synchronize()
    start = time.time()
    
    # Forward + backward + optimize (Eager)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    eager_time = end - start
    print(f"Eager-mode step time: {eager_time:.4f} s")
    
    # Warm up for compilation
    print("Warming up for compilation...")
    compiled_model = torch.compile(model, mode="max-autotune")
    for _ in range(3):  # Warm up compiled model
        outputs = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Compiled mode timing
    optimizer.zero_grad()
    torch.cuda.synchronize()
    start = time.time()
    
    # Forward + backward + optimize (Compiled)
    outputs = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    compiled_time = end - start
    print(f"Compiled-mode step time: {compiled_time:.4f} s")
    
    speedup = eager_time / compiled_time
    print(f"Speedup: {speedup:.2f}x")

def demonstrate_cuda_streams():
    """Demonstrate CUDA streams for overlapping computation and communication"""
    print("\n=== CUDA Streams Example ===")
    
    # Set up streams
    device = 'cuda'
    transfer_stream = torch.cuda.Stream(device)  # for H2D data transfers
    compute_stream = torch.cuda.default_stream(device)  # for compute
    
    # Create dummy data loader
    class DummyDataLoader:
        def __init__(self, num_batches=5):
            self.num_batches = num_batches
            self.current = 0
            
        def __iter__(self):
            return self
            
        def __next__(self):
            if self.current >= self.num_batches:
                raise StopIteration
            self.current += 1
            return (torch.randn(4, 512, device='cpu'), 
                   torch.randint(0, 1000, (4, 512), device='cpu'))
    
    dataloader = DummyDataLoader()
    dataloader_iter = iter(dataloader)
    
    # Preload the very first batch onto GPU
    first_batch = next(dataloader_iter, None)
    if first_batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs, next_labels = (
                first_batch[0].to(device, non_blocking=True),
                first_batch[1].to(device, non_blocking=True),
            )
    
    print("Running training with overlapping streams...")
    for i in range(3):  # Run 3 iterations
        # 1) Wait for transfer of `next` batch to finish, then swap it into compute variables
        compute_stream.wait_stream(transfer_stream)
        inputs, labels = next_inputs, next_labels
        
        # 2) Kick off transfer of the *following* batch on the transfer_stream
        batch = next(dataloader_iter, None)
        if batch:
            with torch.cuda.stream(transfer_stream):
                next_inputs, next_labels = (
                    batch[0].to(device, non_blocking=True),
                    batch[1].to(device, non_blocking=True),
                )
        
        # 3) Run forward/backward on compute_stream
        with torch.cuda.stream(compute_stream):
            outputs = model(inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Iteration {i+1} completed")
    
    print("CUDA streams example completed")

def demonstrate_cuda_graphs():
    """Demonstrate CUDA Graph capture and replay"""
    print("\n=== CUDA Graphs Example ===")
    
    # Create a simple model for demonstration
    simple_model = torch.nn.Linear(1024, 1024).cuda()
    simple_optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)
    
    # Prepare static inputs and outputs
    batch_shape = (4, 1024)
    output_shape = (4, 1024)
    static_input = torch.randn(batch_shape, device='cuda')
    static_output = torch.empty(output_shape, device='cuda')
    
    # Create CUDA graph
    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    
    # Warm-up step on capture_stream to allocate buffers without recording
    with torch.cuda.stream(capture_stream):
        tmp = simple_model(static_input)
        static_output.copy_(tmp)
    capture_stream.synchronize()
    
    # Begin graph capture
    with torch.cuda.graph(g, stream=capture_stream):
        tmp = simple_model(static_input)
        static_output.copy_(tmp)
    capture_stream.synchronize()
    
    print("CUDA graph captured successfully")
    
    # Replay the graph multiple times
    for i in range(3):
        # Load new data into pre-allocated input tensor
        new_batch = torch.randn(batch_shape, device='cuda')
        static_input.copy_(new_batch)
        
        # Execute the captured graph
        g.replay()
        
        # Retrieve the output (clone if you plan to modify it)
        result = static_output.clone()
        print(f"Graph replay {i+1} completed, output shape: {result.shape}")

def demonstrate_fsdp():
    """Demonstrate FSDP with automatic checkpointing and offloading"""
    print("\n=== FSDP Example ===")
    
    # Initialize distributed (single process for demo)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    
    # Build a simple model
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Linear(4096, 4096),
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = MyModel().cuda()
    
    # Auto-wrap transformer blocks if needed
    auto_wrap_policy = transformer_auto_wrap_policy(
        model,
        min_num_params=1e8,
    )
    
    # Wrap with FSDP + checkpointing + CPU offload
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(
            offload_params=True,
            offload_gradients=True),
        use_orig_params=True,
        mixed_precision=False,
        backward_prefetch=True,
        activation_checkpointing_policy={
            torch.nn.TransformerEncoderLayer,
            torch.nn.TransformerDecoderLayer,
            torch.nn.MultiheadAttention,
        }
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-4)
    
    # Training example
    x = torch.randn(8, 4096, device="cuda")
    for epoch in range(2):
        optimizer.zero_grad()
        out = fsdp_model(x)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        print(f"FSDP epoch {epoch+1} completed")

def demonstrate_memory_allocator():
    """Demonstrate custom memory allocator configuration"""
    print("\n=== Memory Allocator Configuration ===")
    
    # Set memory allocator configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
        'max_split_size_mb:128,'
        'roundup_power2_divisions:[256:1,512:2,1024:4,>:8],'
        'garbage_collection_threshold:0.8,'
        'backend:cudaMallocAsync'
    )
    
    print("Memory allocator configured with:")
    print("- max_split_size_mb:128 (keep large blocks intact)")
    print("- roundup_power2_divisions (reduce fragmentation)")
    print("- garbage_collection_threshold:0.8 (aggressive cleanup)")
    print("- backend:cudaMallocAsync (asynchronous allocation)")
    
    # Demonstrate memory stats
    print(f"\nCurrent memory stats:")
    stats = torch.cuda.memory_stats()
    print(f"Allocated: {stats['allocated_bytes.all.current'] / 1024**3:.2f} GB")
    print(f"Reserved: {stats['reserved_bytes.all.current'] / 1024**3:.2f} GB")
    print(f"Peak allocated: {stats['allocated_bytes.all.peak'] / 1024**3:.2f} GB")

def demonstrate_mlperf_logging():
    """Demonstrate MLPerf-style logging"""
    print("\n=== MLPerf Logging Example ===")
    
    # Simulate training iteration timing
    step_time_ms = 24.0
    forward_ms = 10.5
    backward_ms = 9.0
    allreduce_ms = 4.0
    other_ms = 0.5
    
    # Create MLPerf-style log entry
    log_entry = {
        "step_time_ms": step_time_ms,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "allreduce_ms": allreduce_ms,
        "other_ms": other_ms
    }
    
    print("MLPerf Log Entry:")
    print(f":::MLL {log_entry}")
    
    # Calculate percentages
    total = step_time_ms
    print(f"\nBreakdown:")
    print(f"Forward pass: {forward_ms} ms ({forward_ms/total*100:.1f}%)")
    print(f"Backward pass: {backward_ms} ms ({backward_ms/total*100:.1f}%)")
    print(f"All-Reduce: {allreduce_ms} ms ({allreduce_ms/total*100:.1f}%)")
    print(f"Other overhead: {other_ms} ms ({other_ms/total*100:.1f}%)")
    print(f"Total: {total} ms (100%)")

def main():
    """Main function demonstrating all profiling and optimization techniques"""
    print("Chapter 13: Profiling, Tuning, and Scaling PyTorch")
    print("=" * 60)
    
    # Warm up the model
    warm_up_model()
    
    # Run profiling examples
    profile_with_pytorch_profiler()
    compare_eager_vs_compiled()
    demonstrate_cuda_streams()
    demonstrate_cuda_graphs()
    demonstrate_fsdp()
    demonstrate_memory_allocator()
    demonstrate_mlperf_logging()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nKey takeaways:")
    print("- Use PyTorch Profiler to identify bottlenecks")
    print("- Enable torch.compile for automatic optimizations")
    print("- Use CUDA streams for overlapping computation and communication")
    print("- Use CUDA graphs to reduce kernel launch overhead")
    print("- Use FSDP for large model training")
    print("- Configure memory allocator for better performance")
    print("- Set up continuous performance monitoring")

if __name__ == "__main__":
    main()
