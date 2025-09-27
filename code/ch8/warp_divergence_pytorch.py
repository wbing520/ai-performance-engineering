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
import time

def threshold_operations():
    """
    Demonstrate warp divergence avoidance using PyTorch operations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data with mix of positive and negative values
    N = 1 << 18  # 262K elements
    X = torch.randn(N, device=device)
    threshold = 0.0
    
    print("=== Threshold Operations - Avoiding Warp Divergence ===")
    print(f"Input size: {X.shape}")
    print(f"Threshold: {threshold}")
    
    # Method 1: Using torch.maximum (efficient, no divergence)
    start = time.time()
    with torch.cuda.nvtx.range("threshold_maximum"):
        for _ in range(20):
            Y1 = torch.maximum(X, torch.zeros_like(X))
    torch.cuda.synchronize()
    maximum_time = time.time() - start
    
    # Method 2: Using torch.where (also efficient)
    start = time.time()
    with torch.cuda.nvtx.range("threshold_where"):
        for _ in range(20):
            Y2 = torch.where(X > threshold, X, torch.zeros_like(X))
    torch.cuda.synchronize()
    where_time = time.time() - start
    
    # Method 3: Using torch.clamp (another option)
    start = time.time()
    with torch.cuda.nvtx.range("threshold_clamp"):
        for _ in range(20):
            Y3 = torch.clamp(X, min=threshold)
    torch.cuda.synchronize()
    clamp_time = time.time() - start
    
    # Method 4: ReLU (most optimized for this specific case)
    start = time.time()
    with torch.cuda.nvtx.range("threshold_relu"):
        for _ in range(20):
            Y4 = torch.relu(X)
    torch.cuda.synchronize()
    relu_time = time.time() - start
    
    print(f"torch.maximum time: {maximum_time*1000:.3f}ms")
    print(f"torch.where time: {where_time*1000:.3f}ms")
    print(f"torch.clamp time: {clamp_time*1000:.3f}ms")
    print(f"torch.relu time: {relu_time*1000:.3f}ms")
    
    # Verify all methods give same result
    print(f"All results equal: {torch.allclose(Y1, Y2) and torch.allclose(Y2, Y3) and torch.allclose(Y3, Y4)}")
    
    return Y1, Y2, Y3, Y4

def conditional_operations():
    """
    Show different ways to handle conditional operations efficiently.
    """
    device = torch.device('cuda')
    
    # Create test data
    x = torch.randn(200000, device=device)
    y = torch.randn(200000, device=device)
    
    print("\n=== Conditional Operations ===")
    
    # Inefficient: Python loop (would cause massive divergence if done per-element)
    # We'll just demonstrate the concept with a small subset
    def python_loop_approach(x_small, y_small):
        result = torch.zeros_like(x_small)
        for i in range(len(x_small)):
            if x_small[i] > 0:
                result[i] = x_small[i] * 2
            else:
                result[i] = y_small[i] * 3
        return result
    
    # Efficient: Vectorized approach
    def vectorized_approach(x, y):
        mask = x > 0
        result = torch.where(mask, x * 2, y * 3)
        return result
    
    # Test with small subset for Python loop
    x_small = x[:1000]
    y_small = y[:1000]
    
    start = time.time()
    result_loop = python_loop_approach(x_small, y_small)
    loop_time = time.time() - start
    
    # Test vectorized approach on full data
    start = time.time()
    with torch.cuda.nvtx.range("vectorized_conditional"):
        result_vectorized = vectorized_approach(x, y)
    torch.cuda.synchronize()
    vectorized_time = time.time() - start
    
    # Compare first 1000 elements
    result_vectorized_small = vectorized_approach(x_small, y_small)
    
    print(f"Python loop time (1K elements): {loop_time*1000:.3f}ms")
    print(f"Vectorized time (200K elements): {vectorized_time*1000:.3f}ms")
    print(f"Results equal (first 1K): {torch.allclose(result_loop, result_vectorized_small)}")
    
    return result_loop, result_vectorized

def compiled_conditional_operations():
    """
    Use torch.compile to optimize conditional operations.
    """
    device = torch.device('cuda')
    
    @torch.compile(fullgraph=True)
    def compiled_conditional(x, y, threshold):
        """Compiled version of conditional operation."""
        mask1 = x > threshold
        mask2 = x < -threshold
        
        result = torch.where(mask1, x * 2.0, x)
        result = torch.where(mask2, y * 0.5, result)
        result = torch.where(~mask1 & ~mask2, x + y, result)
        
        return result
    
    def uncompiled_conditional(x, y, threshold):
        """Uncompiled version of the same operation."""
        mask1 = x > threshold
        mask2 = x < -threshold
        
        result = torch.where(mask1, x * 2.0, x)
        result = torch.where(mask2, y * 0.5, result)
        result = torch.where(~mask1 & ~mask2, x + y, result)
        
        return result
    
    print("\n=== Compiled Conditional Operations ===")
    
    # Create test data
    x = torch.randn(200000, device=device)
    y = torch.randn(200000, device=device)
    threshold = 0.5
    
    # Warm up compiled version
    _ = compiled_conditional(x, y, threshold)
    
    # Time uncompiled version
    start = time.time()
    with torch.cuda.nvtx.range("uncompiled_conditional"):
        for _ in range(20):
            result_uncompiled = uncompiled_conditional(x, y, threshold)
    torch.cuda.synchronize()
    uncompiled_time = time.time() - start
    
    # Time compiled version
    start = time.time()
    with torch.cuda.nvtx.range("compiled_conditional"):
        for _ in range(20):
            result_compiled = compiled_conditional(x, y, threshold)
    torch.cuda.synchronize()
    compiled_time = time.time() - start
    
    print(f"Uncompiled time: {uncompiled_time*1000:.3f}ms")
    print(f"Compiled time: {compiled_time*1000:.3f}ms")
    print(f"Compilation speedup: {uncompiled_time/compiled_time:.2f}x")
    
    # Verify results
    diff = torch.abs(result_uncompiled - result_compiled).max().item()
    print(f"Max difference: {diff:.2e}")
    
    return result_uncompiled, result_compiled

def mask_operations():
    """
    Demonstrate efficient mask-based operations.
    """
    device = torch.device('cuda')
    
    # Create sparse-like data where only some elements need processing
    data = torch.randn(200000, device=device)
    
    # Create a sparse mask (only 10% of elements are "active")
    torch.manual_seed(42)
    mask = torch.rand(200000, device=device) < 0.1
    
    print("\n=== Mask-Based Operations ===")
    print(f"Data size: {data.shape}")
    print(f"Active elements: {mask.sum().item()}/{len(mask)} ({mask.float().mean()*100:.1f}%)")
    
    # Method 1: Process all elements, use mask to zero out unused
    start = time.time()
    with torch.cuda.nvtx.range("mask_all_elements"):
        for _ in range(20):
            processed = torch.sin(data) * torch.cos(data)  # Some computation
            result1 = torch.where(mask, processed, torch.zeros_like(processed))
    torch.cuda.synchronize()
    mask_time = time.time() - start
    
    # Method 2: Use advanced indexing to process only active elements
    start = time.time()
    with torch.cuda.nvtx.range("mask_active_only"):
        for _ in range(20):
            active_indices = torch.nonzero(mask, as_tuple=False).squeeze()
            active_data = data[active_indices]
            processed_active = torch.sin(active_data) * torch.cos(active_data)
            
            result2 = torch.zeros_like(data)
            result2[active_indices] = processed_active
    torch.cuda.synchronize()
    active_time = time.time() - start
    
    print(f"Process all + mask time: {mask_time*1000:.3f}ms")
    print(f"Process active only time: {active_time*1000:.3f}ms")
    print(f"Active-only speedup: {mask_time/active_time:.2f}x")
    
    # Verify results are similar (may have small floating point differences)
    difference = torch.abs(result1 - result2).max().item()
    print(f"Max difference: {difference:.2e}")
    
    return result1, result2

def main():
    """
    Main function demonstrating warp divergence avoidance in PyTorch.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU examples")
        return
    
    print("=== PyTorch Warp Divergence Avoidance Examples ===")
    print(f"Using device: {torch.cuda.get_device_name()}")
    
    # Run examples
    threshold_operations()
    conditional_operations()
    compiled_conditional_operations()
    mask_operations()
    
    print("\n=== Key Takeaways ===")
    print("1. Use vectorized operations like torch.where, torch.maximum instead of Python loops")
    print("2. torch.compile can fuse multiple conditional operations for better efficiency")
    print("3. For sparse computations, consider processing only active elements")
    print("4. Built-in functions like torch.relu are highly optimized")
    
    print("\n=== Profiling Commands ===")
    print("To profile warp divergence with Nsight Compute:")
    print("ncu --metrics smsp__warp_execution_efficiency.avg.pct_of_peak_sustained python warp_divergence_pytorch.py")

if __name__ == "__main__":
    main()
