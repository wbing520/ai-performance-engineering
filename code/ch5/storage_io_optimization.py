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
    if compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
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
# storage_io_optimization.py
import torch
from torch.utils.data import DataLoader

# Create a DataLoader that prefetches 4 batches per worker into pinned CPU memory.

def create_optimized_dataloader(dataset, batch_size=32, num_workers=8):
    """
    Create an optimized DataLoader for high-performance storage I/O.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
    
    Returns:
        Optimized DataLoader with pinned memory and prefetching
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,        # Use pinned (page-locked) memory for faster GPU transfers
        prefetch_factor=4,      # Prefetch 4 batches per worker
        persistent_workers=True, # Keep workers alive between epochs
        shuffle=True,
        drop_last=True          # Drop incomplete batches for consistent performance
    )
    return loader

def optimized_training_loop(model, dataloader, device, optimizer, criterion):
    """
    Training loop with optimized data transfer patterns.
    
    Demonstrates:
    - Asynchronous GPU transfers with non_blocking=True
    - Overlapping data transfer with computation
    """
    model.train()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Asynchronously copy the batch to the GPU
        # This allows the transfer to happen in background while GPU processes previous batch
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # While the copy is still happening in the background, 
        # the GPU may begin processing the current batch
        # This call will block only if the copy isn't yet complete
        outputs = model(data)
        
        loss = criterion(outputs, target)
        
        # Standard training steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Optional: Log progress without blocking
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')

# Example usage with different dataset scenarios
def example_usage():
    """
    Example showing different optimization strategies for various scenarios.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    
    # Create dummy dataset for demonstration
    X = torch.randn(10000, 224*224*3)  # Simulated image data
    y = torch.randint(0, 10, (10000,))  # Simulated labels
    dataset = TensorDataset(X, y)
    
    # Scenario 1: Standard configuration
    standard_loader = DataLoader(
        dataset, 
        batch_size=32, 
        num_workers=4,
        pin_memory=False  # Default, slower
    )
    
    # Scenario 2: Optimized for throughput
    optimized_loader = create_optimized_dataloader(
        dataset, 
        batch_size=64,     # Larger batches for better GPU utilization
        num_workers=8      # More workers for faster data loading
    )
    
    # Scenario 3: Memory-constrained environment
    memory_efficient_loader = DataLoader(
        dataset,
        batch_size=16,     # Smaller batches to fit in memory
        num_workers=2,     # Fewer workers to reduce memory overhead
        pin_memory=True,
        prefetch_factor=2, # Less prefetching to save memory
        persistent_workers=False  # Don't keep workers alive
    )
    
    # Create simple model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Linear(224*224*3, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Running optimized training loop...")
    # Run a few iterations with the optimized loader
    for i, (data, target) in enumerate(optimized_loader):
        if i >= 5:  # Just demonstrate first few batches
            break
        optimized_training_loop(model, [(data, target)], device, optimizer, criterion)

if __name__ == "__main__":
    example_usage()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Note: These configuration options may not be available in all PyTorch versions
    # Commenting out for compatibility
    #     torch._inductor.config.triton.use_hopper_optimizations = True
    #     torch._inductor.config.triton.hbm3_optimizations = True
    # if compute_capability == "10.0":  # Blackwell B200/B300
    #     torch._inductor.config.triton.use_blackwell_optimizations = True
    #     torch._inductor.config.triton.hbm3e_optimizations = True
    #     torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.9 features (if available)
    try:
        torch._inductor.config.triton.unique_kernel_names = True
    except AttributeError:
        pass
    
    try:
        torch._inductor.config.triton.autotune_mode = "max-autotune"
    except AttributeError:
        pass
    
    try:
        torch._dynamo.config.automatic_dynamic_shapes = True
    except AttributeError:
        pass
