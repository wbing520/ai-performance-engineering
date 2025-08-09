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
import torch

def vectorized_lookup(table, N):
    """
    Efficient lookup operation using PyTorch's optimized indexing.
    """
    flat = table.view(-1)
    T = flat.size(0)
    
    # build indices [0,1,2,…,N-1] % T all on GPU
    idx = torch.arange(N, device=flat.device) % T
    
    # one gather kernel does all N loads in parallel
    return flat.index_select(0, idx)

def embedding_lookup_example(vocab_size, embedding_dim, sequence_length):
    """
    Example of using PyTorch's optimized embedding lookup.
    This demonstrates read-only cache usage patterns.
    """
    # Create embedding table (read-only during inference)
    embedding = torch.nn.Embedding(vocab_size, embedding_dim)
    embedding = embedding.cuda()
    
    # Create random input sequence
    input_ids = torch.randint(0, vocab_size, (sequence_length,), device='cuda')
    
    # Efficient lookup using PyTorch's optimized embedding
    with torch.cuda.nvtx.range("embedding_lookup"):
        embeddings = embedding(input_ids)
    
    return embeddings

def main():
    """
    Demonstrate read-only cache optimization patterns.
    """
    # Simple lookup table example
    T = 1024
    N = 1 << 20
    
    print(f"Lookup table size: {T}")
    print(f"Number of lookups: {N}")
    
    table = torch.arange(T, dtype=torch.float32, device='cuda')
    
    print("\n=== Vectorized Lookup ===")
    with torch.cuda.nvtx.range("vectorized_lookup"):
        out = vectorized_lookup(table, N)
    print(f"Output size: {out.shape}")
    print(f"Sample values: {out[:5]}")
    
    # Embedding lookup example (common in NLP)
    print("\n=== Embedding Lookup Example ===")
    vocab_size = 50000
    embedding_dim = 768
    sequence_length = 512
    
    embeddings = embedding_lookup_example(vocab_size, embedding_dim, sequence_length)
    print(f"Embedding output shape: {embeddings.shape}")
    print(f"Sample embedding norm: {embeddings[0].norm().item():.4f}")
    
    print("\nFor profiling, use:")
    print("nsys profile --trace=cuda,nvtx python lookup_pytorch.py")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        exit(1)
    
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Enable latest PyTorch 2.8 features (only use existing config options)
    try:
        torch._inductor.config.triton.unique_kernel_names = True
    except AttributeError:
        pass
    
    print(f"Running on {device_props.name} with compute capability {compute_capability}")
