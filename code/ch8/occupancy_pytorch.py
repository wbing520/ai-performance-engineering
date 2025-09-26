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
        return
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
import torch
import torch.utils.cpp_extension

def analyze_tensor_operations():
    """
    Analyze how PyTorch tensor operations relate to GPU occupancy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small tensors - may not fully utilize GPU
    small_a = torch.randn(100, 100, device=device)
    small_b = torch.randn(100, 100, device=device)
    
    # Large tensors - better GPU utilization
    large_a = torch.randn(4096, 4096, device=device)
    large_b = torch.randn(4096, 4096, device=device)
    
    print("=== GPU Occupancy Considerations in PyTorch ===")
    print(f"Small tensor size: {small_a.shape}")
    print(f"Large tensor size: {large_a.shape}")
    
    # Time operations to see the difference
    import time
    
    # Small tensor operations
    start = time.time()
    with torch.cuda.nvtx.range("small_matmul"):
        for _ in range(100):
            small_c = torch.mm(small_a, small_b)
    torch.cuda.synchronize()
    small_time = time.time() - start
    
    # Large tensor operations (fewer iterations to avoid long runtime)
    start = time.time()
    with torch.cuda.nvtx.range("large_matmul"):
        for _ in range(10):
            large_c = torch.mm(large_a, large_b)
    torch.cuda.synchronize()
    large_time = time.time() - start
    
    print(f"\nSmall tensor ops (100x): {small_time:.4f}s")
    print(f"Large tensor ops (10x): {large_time:.4f}s")
    print(f"Avg small op time: {small_time/100*1000:.3f}ms")
    print(f"Avg large op time: {large_time/10*1000:.3f}ms")
    
    return small_c, large_c

def batching_for_occupancy():
    """
    Demonstrate how batching improves GPU utilization.
    """
    device = torch.device('cuda')
    
    # Individual small operations vs batched operations
    individual_tensors = [torch.randn(256, 256, device=device) for _ in range(16)]
    batched_tensor = torch.stack(individual_tensors, dim=0)  # Shape: [16, 256, 256]
    
    print("\n=== Batching for Better Occupancy ===")
    print(f"Individual tensor shape: {individual_tensors[0].shape}")
    print(f"Batched tensor shape: {batched_tensor.shape}")
    
    import time
    
    # Individual operations
    start = time.time()
    with torch.cuda.nvtx.range("individual_ops"):
        individual_results = []
        for tensor in individual_tensors:
            result = torch.relu(tensor)
            result = torch.sigmoid(result)
            individual_results.append(result)
    torch.cuda.synchronize()
    individual_time = time.time() - start
    
    # Batched operations
    start = time.time()
    with torch.cuda.nvtx.range("batched_ops"):
        batched_result = torch.relu(batched_tensor)
        batched_result = torch.sigmoid(batched_result)
    torch.cuda.synchronize()
    batched_time = time.time() - start
    
    print(f"Individual operations time: {individual_time*1000:.3f}ms")
    print(f"Batched operations time: {batched_time*1000:.3f}ms")
    print(f"Batching speedup: {individual_time/batched_time:.2f}x")
    
    return individual_results, batched_result

def torch_compile_occupancy():
    """
    Demonstrate how torch.compile can improve occupancy through fusion.
    """
    device = torch.device('cuda')
    
    @torch.compile(fullgraph=True, mode="max-autotune")
    def fused_operations(x):
        """Compile and fuse multiple operations."""
        x = torch.relu(x)
        x = x * 2.0
        x = torch.sigmoid(x)
        x = x + 1.0
        return x
    
    def unfused_operations(x):
        """Individual operations without fusion."""
        x = torch.relu(x)
        x = x * 2.0
        x = torch.sigmoid(x)
        x = x + 1.0
        return x
    
    print("\n=== Torch Compile for Better Occupancy ===")
    
    # Create test data
    x = torch.randn(1024, 1024, device=device)
    
    import time
    
    # Warm up
    _ = fused_operations(x)
    _ = unfused_operations(x)
    
    # Time unfused operations
    start = time.time()
    with torch.cuda.nvtx.range("unfused_ops"):
        for _ in range(50):
            result_unfused = unfused_operations(x)
    torch.cuda.synchronize()
    unfused_time = time.time() - start
    
    # Time fused operations
    start = time.time()
    with torch.cuda.nvtx.range("fused_ops"):
        for _ in range(50):
            result_fused = fused_operations(x)
    torch.cuda.synchronize()
    fused_time = time.time() - start
    
    print(f"Unfused operations time: {unfused_time*1000:.3f}ms")
    print(f"Fused operations time: {fused_time*1000:.3f}ms")
    print(f"Fusion speedup: {unfused_time/fused_time:.2f}x")
    
    # Verify results are the same
    diff = torch.abs(result_unfused - result_fused).max().item()
    print(f"Max difference between fused/unfused: {diff:.2e}")
    
    return result_unfused, result_fused

def memory_layout_occupancy():
    """
    Show how memory layout affects occupancy.
    """
    device = torch.device('cuda')
    
    # Create tensors with different memory layouts
    channels_last = torch.randn(32, 128, 64, 64, device=device).to(memory_format=torch.channels_last)
    channels_first = torch.randn(32, 128, 64, 64, device=device)
    
    print("\n=== Memory Layout and Occupancy ===")
    print(f"Channels last contiguous: {channels_last.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Channels first contiguous: {channels_first.is_contiguous()}")
    
    # Simple convolution operation
    conv = torch.nn.Conv2d(128, 256, 3, padding=1).cuda()
    
    import time
    
    # Channels last
    start = time.time()
    with torch.cuda.nvtx.range("channels_last_conv"):
        for _ in range(20):
            result_cl = conv(channels_last)
    torch.cuda.synchronize()
    cl_time = time.time() - start
    
    # Channels first
    start = time.time()
    with torch.cuda.nvtx.range("channels_first_conv"):
        for _ in range(20):
            result_cf = conv(channels_first)
    torch.cuda.synchronize()
    cf_time = time.time() - start
    
    print(f"Channels last conv time: {cl_time*1000:.3f}ms")
    print(f"Channels first conv time: {cf_time*1000:.3f}ms")
    print(f"Channels last speedup: {cf_time/cl_time:.2f}x")
    
    return result_cl, result_cf

def main():
    """
    Main function demonstrating PyTorch occupancy considerations.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU occupancy examples")
        return
    
    print("=== PyTorch GPU Occupancy Optimization Examples ===")
    print(f"Using device: {torch.cuda.get_device_name()}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run examples
    analyze_tensor_operations()
    batching_for_occupancy()
    torch_compile_occupancy()
    memory_layout_occupancy()
    
    print("\n=== Profiling Commands ===")
    print("To profile these examples with Nsight Systems:")
    print("nsys profile --trace=cuda,nvtx python occupancy_pytorch.py")
    
    print("\nTo see occupancy in PyTorch Profiler:")
    print("Use torch.profiler.profile() with activities=[torch.profiler.ProfilerActivity.CUDA]")

if __name__ == "__main__":
    main()
