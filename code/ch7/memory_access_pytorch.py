import torch

def uncoalesced_copy(input_tensor, stride):
    """
    Demonstrates uncoalesced memory access pattern using strided indexing.
    This creates a gather operation that causes uncoalesced loads.
    """
    # Flatten to 1D so we know exactly which dimension we're indexing
    flat_tensor = input_tensor.contiguous().view(-1)
    
    # Generate indices with a fixed stride to gather
    idx = torch.arange(0, flat_tensor.numel(), stride,
                      device=flat_tensor.device, dtype=torch.long)
    
    # index_select uses a gather kernel that issues uncoalesced loads
    return torch.index_select(flat_tensor, 0, idx)

def coalesced_copy(input_tensor):
    """
    Demonstrates coalesced memory access - PyTorch handles this efficiently.
    """
    # PyTorch's clone() operation is already optimized for coalesced access
    return input_tensor.clone()

def main():
    """
    Compare uncoalesced vs coalesced memory access patterns.
    """
    n, stride = 1 << 20, 2
    
    # Create input tensor
    inp = torch.arange(n * stride, device='cuda', dtype=torch.float32)
    
    print(f"Input tensor size: {inp.shape}")
    print(f"Stride: {stride}")
    
    # Uncoalesced access (inefficient)
    print("\n=== Uncoalesced Copy ===")
    with torch.cuda.nvtx.range("uncoalesced_copy"):
        out_uncoalesced = uncoalesced_copy(inp, stride)
    print(f"Output size: {out_uncoalesced.shape}")
    
    # Coalesced access (efficient)
    print("\n=== Coalesced Copy ===")
    with torch.cuda.nvtx.range("coalesced_copy"):
        out_coalesced = coalesced_copy(inp)
    print(f"Output size: {out_coalesced.shape}")
    
    print("\nFor profiling, use:")
    print("nsys profile --trace=cuda,nvtx python memory_access_pytorch.py")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        exit(1)
    
    main()
