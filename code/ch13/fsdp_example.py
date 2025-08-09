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
            "memory_bandwidth": "8.0 TB/s",
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
# fsdp_example.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
import os

class TransformerBlock(nn.Module):
    """Simple transformer block for FSDP demonstration."""
    def __init__(self, dim=512, num_heads=8, ff_dim=2048):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class MyModel(nn.Module):
    def __init__(self, num_layers=12, dim=512):
        super().__init__()
        self.embedding = nn.Embedding(10000, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 10000)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

def setup_distributed():
    """Initialize distributed training."""
    if not dist.is_initialized():
        # Set default values for single-node training
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        
        # Initialize process group
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    return rank, world_size

def create_fsdp_model():
    """Create a model with FSDP wrapping."""
    rank, world_size = setup_distributed()
    
    # Create the model
    model = MyModel(num_layers=12, dim=512)
    
    # Mixed precision policy for FP16 training
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # Auto-wrap policy for transformer layers
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    # Wrap model with FSDP
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,  # Ensure all ranks start with same parameters
    )
    
    return fsdp_model, rank, world_size

def train_step(model, batch, optimizer, criterion):
    """Single training step with FSDP."""
    optimizer.zero_grad()
    
    # Forward pass
    input_ids, labels = batch
    outputs = model(input_ids)
    
    # Compute loss (shift for language modeling)
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()

def main():
    """Main training function."""
    try:
        fsdp_model, rank, world_size = create_fsdp_model()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Create dummy data
        batch_size = 4
        seq_length = 128
        vocab_size = 10000
        
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        
        # Generate random data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
        labels = input_ids.clone()
        
        if rank == 0:
            print(f"Training with FSDP on {world_size} rank(s)")
            print(f"Model parameters: {sum(p.numel() for p in fsdp_model.parameters()):,}")
            
            if torch.cuda.is_available():
                print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Training loop
        for step in range(5):
            loss = train_step(fsdp_model, (input_ids, labels), optimizer, criterion)
            
            if rank == 0:
                print(f"Step {step}: Loss = {loss:.4f}")
                
                if torch.cuda.is_available():
                    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        if rank == 0:
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Error in training: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def demonstrate_memory_efficiency():
    """Compare memory usage with and without FSDP."""
    print("\n=== FSDP Memory Efficiency Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory efficiency demo")
        return
    
    # Test without FSDP
    print("Testing without FSDP:")
    torch.cuda.reset_peak_memory_stats()
    
    regular_model = MyModel(num_layers=6, dim=512).cuda()
    optimizer = torch.optim.AdamW(regular_model.parameters())
    
    input_ids = torch.randint(0, 10000, (4, 128), device="cuda")
    labels = input_ids.clone()
    
    # Forward and backward pass
    outputs = regular_model(input_ids)
    loss = nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), 
        labels.view(-1)
    )
    loss.backward()
    
    regular_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak memory without FSDP: {regular_memory:.2f} GB")
    
    # Clean up
    del regular_model, optimizer, outputs, loss
    torch.cuda.empty_cache()
    
    # Test with FSDP (if distributed is available)
    try:
        if not dist.is_initialized():
            # For single GPU demo, we can still show the setup
            print("FSDP would reduce memory usage through parameter sharding")
            print("In multi-GPU setup, memory would be distributed across devices")
    except:
        print("Distributed training not available for FSDP demo")

if __name__ == "__main__":
    main()
    demonstrate_memory_efficiency()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        # Enable Hopper-specific optimizations if available
        try:
            if hasattr(torch._inductor.config.triton, 'use_hopper_optimizations'):
                torch._inductor.config.triton.use_hopper_optimizations = True
            if hasattr(torch._inductor.config.triton, 'hbm3_optimizations'):
                torch._inductor.config.triton.hbm3_optimizations = True
        except AttributeError:
            print("Hopper optimizations not available in this PyTorch version")
    elif compute_capability == "10.0":  # Blackwell B200/B300
        # Enable Blackwell-specific optimizations if available
        try:
            if hasattr(torch._inductor.config.triton, 'use_blackwell_optimizations'):
                torch._inductor.config.triton.use_blackwell_optimizations = True
            if hasattr(torch._inductor.config.triton, 'hbm3e_optimizations'):
                torch._inductor.config.triton.hbm3e_optimizations = True
            if hasattr(torch._inductor.config.triton, 'tma_support'):
                torch._inductor.config.triton.tma_support = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch version")
    
    # Enable latest PyTorch 2.8 features if available
    try:
        if hasattr(torch._inductor.config.triton, 'unique_kernel_names'):
            torch._inductor.config.triton.unique_kernel_names = True
        if hasattr(torch._inductor.config.triton, 'autotune_mode'):
            torch._inductor.config.triton.autotune_mode = "max-autotune"
        if hasattr(torch._dynamo.config, 'automatic_dynamic_shapes'):
            torch._dynamo.config.automatic_dynamic_shapes = True
    except AttributeError:
        print("Some PyTorch 2.8 features not available in this version")
