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
# memory_profiling.py
import torch
import torch.nn as nn

def demonstrate_memory_profiling():
    """Demonstrate PyTorch memory profiling capabilities."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Clear any existing allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("=== Memory Profiling Demo ===")
    
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1024, 2048)
            self.linear2 = nn.Linear(2048, 4096)
            self.linear3 = nn.Linear(4096, 1000)
            
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            return self.linear3(x)
    
    model = SimpleModel().to(device)
    
    # Create input data
    batch_size = 32
    input_data = torch.randn(batch_size, 1024, device=device)
    target = torch.randint(0, 1000, (batch_size,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    if torch.cuda.is_available():
        # Memory snapshot before training
        print("\nInitial memory state:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        
        # Take memory snapshot
        torch.cuda.memory._record_memory_history(True)
    
    # Training loop with memory tracking
    for epoch in range(3):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            print(f"\nEpoch {epoch + 1}:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
            print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    
    if torch.cuda.is_available():
        # Memory snapshot
        snapshot = torch.cuda.memory._snapshot()
        print(f"\nMemory snapshot contains {len(snapshot)} entries")
        
        # Note: You can save this snapshot and load it into PyTorch memory visualizer
        # torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
        # Stop recording
        torch.cuda.memory._record_memory_history(False)
        
        # Detailed memory stats
        memory_stats = torch.cuda.memory_stats()
        print(f"\nDetailed Memory Statistics:")
        print(f"Peak allocated bytes: {memory_stats.get('allocated_bytes.all.peak', 0) / 1e6:.1f} MB")
        print(f"Peak reserved bytes: {memory_stats.get('reserved_bytes.all.peak', 0) / 1e6:.1f} MB")
        print(f"Number of allocations: {memory_stats.get('num_alloc_retries', 0)}")
        print(f"Number of OOM: {memory_stats.get('num_ooms', 0)}")

def demonstrate_memory_optimization():
    """Show memory optimization techniques."""
    print("\n=== Memory Optimization Techniques ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Gradient Checkpointing
    class CheckpointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(1024, 1024) for _ in range(10)
            ])
            
        def forward(self, x):
            for layer in self.layers:
                # Use checkpoint to trade compute for memory
                x = torch.utils.checkpoint.checkpoint(torch.relu, layer(x))
            return x
    
    print("1. Gradient Checkpointing:")
    model = CheckpointModel().to(device)
    x = torch.randn(32, 1024, device=device, requires_grad=True)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    if torch.cuda.is_available():
        print(f"Peak memory with checkpointing: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    
    # 2. Memory-efficient attention (scaled_dot_product_attention)
    print("\n2. Memory-efficient attention:")
    
    def efficient_attention_demo():
        batch_size, seq_len, embed_dim = 8, 512, 512
        
        query = torch.randn(batch_size, seq_len, embed_dim, device=device)
        key = torch.randn(batch_size, seq_len, embed_dim, device=device)
        value = torch.randn(batch_size, seq_len, embed_dim, device=device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Use PyTorch's memory-efficient attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            result = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
            if torch.cuda.is_available():
                print(f"Memory-efficient attention peak: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        else:
            print("scaled_dot_product_attention not available in this PyTorch version")
    
    efficient_attention_demo()
    
    # 3. Mixed precision
    print("\n3. Mixed precision training:")
    
    def mixed_precision_demo():
        model = nn.Linear(1024, 1024).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        x = torch.randn(32, 1024, device=device)
        target = torch.randn(32, 1024, device=device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if torch.cuda.is_available():
            print(f"Mixed precision peak memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
    
    mixed_precision_demo()

if __name__ == "__main__":
    demonstrate_memory_profiling()
    demonstrate_memory_optimization()

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
