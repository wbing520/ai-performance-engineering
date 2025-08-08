# custom_allocator.py
import torch
import torch.distributed as dist
from torch.cuda.memory import CUDAPluggableAllocator
import os

def demonstrate_custom_allocator():
    """Demonstrate custom CUDA memory allocator setup."""
    print("=== Custom CUDA Allocator Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping allocator demo")
        return
    
    # Show current allocator info
    print(f"Default allocator backend: {torch.cuda.get_allocator_backend()}")
    
    # Memory allocation patterns
    def test_allocation_pattern(name, allocator_config=None):
        print(f"\nTesting {name}:")
        
        if allocator_config:
            # Set custom allocator configuration
            os.environ.update(allocator_config)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate various tensor sizes
        tensors = []
        sizes = [1024, 2048, 4096, 8192, 16384]
        
        for size in sizes:
            tensor = torch.randn(size, size, device='cuda')
            tensors.append(tensor)
        
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        
        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved: {reserved:.1f} MB")
        print(f"  Efficiency: {allocated/reserved*100:.1f}%")
        
        # Free tensors
        del tensors
        torch.cuda.empty_cache()
        
        return allocated, reserved
    
    # Test default allocator
    default_alloc, default_reserved = test_allocation_pattern("Default Allocator")
    
    # Test with custom configuration
    custom_config = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.6"
    }
    custom_alloc, custom_reserved = test_allocation_pattern(
        "Custom Configuration", custom_config
    )
    
    # Show improvement
    print(f"\nMemory efficiency improvement: {(custom_alloc/custom_reserved) / (default_alloc/default_reserved) - 1:.1%}")

def demonstrate_memory_pool():
    """Demonstrate memory pool management."""
    print("\n=== Memory Pool Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory pool demo")
        return
    
    # Get current memory pool
    device = torch.cuda.current_device()
    
    print(f"Current device: {device}")
    
    # Memory statistics before allocation
    print("\nMemory stats before allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Allocate some tensors
    tensors = []
    for i in range(5):
        tensor = torch.randn(1024, 1024, device=device)
        tensors.append(tensor)
    
    print("\nMemory stats after allocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Free some tensors but keep references
    del tensors[::2]  # Delete every other tensor
    
    print("\nMemory stats after partial deallocation:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Empty cache to return memory to OS
    torch.cuda.empty_cache()
    
    print("\nMemory stats after empty_cache():")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved(device) / 1e6:.1f} MB")
    
    # Clean up remaining tensors
    del tensors
    torch.cuda.empty_cache()

def demonstrate_memory_snapshot():
    """Demonstrate memory snapshot for debugging."""
    print("\n=== Memory Snapshot Demo ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping snapshot demo")
        return
    
    # Enable memory history tracking
    torch.cuda.memory._record_memory_history(True)
    
    # Simulate some allocations
    tensors = []
    for i in range(3):
        # Create tensors of different sizes
        size = 1024 * (i + 1)
        tensor = torch.randn(size, size, device='cuda')
        tensors.append(tensor)
        print(f"Allocated tensor {i+1}: {size}x{size}")
    
    # Take snapshot
    snapshot = torch.cuda.memory._snapshot()
    print(f"\nSnapshot contains {len(snapshot)} memory events")
    
    # Analyze snapshot (simplified)
    total_allocated = 0
    total_freed = 0
    
    for event in snapshot[:10]:  # Show first 10 events
        action = event.get('action', 'unknown')
        size = event.get('size', 0)
        
        if action == 'alloc':
            total_allocated += size
        elif action == 'free':
            total_freed += size
    
    print(f"Sample analysis - Allocated: {total_allocated / 1e6:.1f} MB, Freed: {total_freed / 1e6:.1f} MB")
    
    # Note: You can save this data for visualization
    # torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # print("Memory snapshot saved to memory_snapshot.pickle")
    # print("Load this file in PyTorch memory visualizer: https://pytorch.org/memory_viz")
    
    # Stop recording
    torch.cuda.memory._record_memory_history(False)
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()

def demonstrate_distributed_memory():
    """Demonstrate memory management in distributed settings."""
    print("\n=== Distributed Memory Demo ===")
    
    try:
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"Running in distributed mode: rank {rank}/{world_size}")
        else:
            print("Not in distributed environment, simulating single rank")
            rank = 0
            world_size = 1
        
        if torch.cuda.is_available():
            # Set device based on rank
            device = rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
            
            print(f"Using GPU {device}")
            
            # Memory allocation per rank
            tensor_size = 1024 // world_size  # Distribute memory load
            tensor = torch.randn(tensor_size, tensor_size, device=f'cuda:{device}')
            
            print(f"Allocated {tensor_size}x{tensor_size} tensor on GPU {device}")
            print(f"Memory used: {torch.cuda.memory_allocated(device) / 1e6:.1f} MB")
            
            del tensor
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Distributed memory demo error: {e}")

if __name__ == "__main__":
    demonstrate_custom_allocator()
    demonstrate_memory_pool()
    demonstrate_memory_snapshot()
    demonstrate_distributed_memory()
