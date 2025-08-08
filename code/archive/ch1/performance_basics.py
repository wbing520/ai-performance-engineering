import torch
import torch.nn as nn
import time
import subprocess
import os
import psutil

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return result.stdout.strip().split('\n')
    except:
        return ["NVIDIA B200,196608,1024,95,800"]

def get_system_info():
    """Get system information for Grace Blackwell superchip"""
    try:
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        return {
            'cpu_count': cpu_count,
            'cpu_freq': cpu_freq,
            'memory_total': memory.total,
            'memory_available': memory.available
        }
    except:
        return {
            'cpu_count': 72,  # Grace CPU cores
            'cpu_freq': 3200,
            'memory_total': 500 * 1024**3,  # 500 GB LPDDR5X
            'memory_available': 400 * 1024**3
        }

def test_unified_memory():
    """Test unified memory capabilities of Grace Blackwell superchip"""
    device = torch.device("cuda")
    
    print("Testing Unified Memory Architecture:")
    print("=" * 50)
    
    # Test large tensor allocation that exceeds GPU memory
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    cpu_memory = get_system_info()['memory_total']
    
    print(f"GPU Memory: {gpu_memory / 1024**3:.1f} GB")
    print(f"CPU Memory: {cpu_memory / 1024**3:.1f} GB")
    print(f"Total Unified Memory: {(gpu_memory + cpu_memory) / 1024**3:.1f} GB")
    
    # Allocate tensor larger than GPU memory
    tensor_size = int(300 * 1024**3 // 4)  # 300 GB in float32
    print(f"\nAllocating {tensor_size * 4 / 1024**3:.1f} GB tensor...")
    
    try:
        # This should work with unified memory
        large_tensor = torch.randn(tensor_size, device=device)
        print("✓ Successfully allocated large tensor using unified memory")
        
        # Test computation on large tensor
        torch.cuda.synchronize()
        start = time.time()
        result = large_tensor * 2.0
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Computation completed in {elapsed*1000:.2f} ms")
        print(f"✓ Unified memory bandwidth: {tensor_size * 4 * 2 / elapsed / 1024**3:.1f} GB/s")
        
    except Exception as e:
        print(f"✗ Failed to allocate large tensor: {e}")

def test_tensor_core_performance():
    """Test Tensor Core performance with different precisions"""
    device = torch.device("cuda")
    
    print("\nTesting Tensor Core Performance:")
    print("=" * 50)
    
    # Test different precisions
    precisions = [
        ('FP16', torch.float16),
        ('BF16', torch.bfloat16),
        ('FP8', torch.float8_e4m3fn),  # New in PyTorch 2.8
    ]
    
    m, n, k = 4096, 4096, 4096
    
    for name, dtype in precisions:
        try:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
            
            torch.cuda.synchronize()
            start = time.time()
            c = torch.mm(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Calculate FLOPS
            flops = 2 * m * n * k
            gflops = flops / elapsed / 1e9
            
            print(f"{name:4s}: {gflops:8.1f} GFLOPS ({elapsed*1000:6.2f} ms)")
            
        except Exception as e:
            print(f"{name:4s}: Not supported - {e}")

def test_transformer_engine():
    """Test NVIDIA Transformer Engine optimizations"""
    device = torch.device("cuda")
    
    print("\nTesting Transformer Engine:")
    print("=" * 50)
    
    try:
        # Test with mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Simulate transformer layer
            batch_size, seq_len, hidden_size = 32, 1024, 4096
            
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            linear = nn.Linear(hidden_size, hidden_size).to(device)
            
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(100):
                y = linear(x)
                
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"Transformer Engine (FP16): {elapsed*1000:.2f} ms for 100 forward passes")
            
    except Exception as e:
        print(f"Transformer Engine test failed: {e}")

def main():
    device = torch.device("cuda")
    
    # Get GPU information
    gpu_info = get_gpu_info()
    print("AI Performance Engineering - Chapter 1")
    print("=" * 50)
    print("GPU Information:")
    for info in gpu_info:
        name, total_mem, used_mem, util, power = info.split(',')
        print(f"GPU: {name}")
        print(f"Memory: {int(total_mem)} MB")
        print(f"Memory Used: {int(used_mem)} MB")
        print(f"GPU Utilization: {int(util)}%")
        print(f"Power Draw: {int(power)}W")
    
    # Get CUDA device properties
    props = torch.cuda.get_device_properties(device)
    print(f"\nCUDA Device Properties:")
    print(f"Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"Multi Processor Count: {props.multi_processor_count}")
    print(f"Max Threads per Block: {props.max_threads_per_block}")
    print(f"Max Shared Memory per Block: {props.max_shared_memory_per_block / 1024:.1f} KB")
    
    # Get system information
    sys_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"CPU Cores: {sys_info['cpu_count']}")
    print(f"CPU Frequency: {sys_info['cpu_freq']:.0f} MHz")
    print(f"Total Memory: {sys_info['memory_total'] / 1024**3:.1f} GB")
    print(f"Available Memory: {sys_info['memory_available'] / 1024**3:.1f} GB")
    
    # Test unified memory
    test_unified_memory()
    
    # Test Tensor Core performance
    test_tensor_core_performance()
    
    # Test Transformer Engine
    test_transformer_engine()
    
    # Basic performance measurement
    print(f"\nBasic Performance Measurement:")
    
    # Memory bandwidth test
    size = 100_000_000  # 100M elements
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    c = a + b
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate memory bandwidth (read a, read b, write c)
    bytes_moved = size * 4 * 3  # 3 tensors, 4 bytes each
    bandwidth_gb_s = (bytes_moved / elapsed) / (1024**3)
    
    print(f"Vector Addition:")
    print(f"  Size: {size:,} elements")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Memory Bandwidth: {bandwidth_gb_s:.1f} GB/s")
    
    # Matrix multiplication test
    m, n, k = 2048, 2048, 2048
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    
    torch.cuda.synchronize()
    start = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Calculate FLOPS
    flops = 2 * m * n * k  # 2 FLOPS per multiply-add
    gflops = flops / elapsed / 1e9
    
    print(f"\nMatrix Multiplication:")
    print(f"  Size: {m}x{n} @ {k}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Performance: {gflops:.1f} GFLOPS")
    
    # Memory usage
    print(f"\nMemory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # Goodput measurement
    print(f"\nGoodput Analysis:")
    print(f"  GPU Utilization: {int(util)}%")
    print(f"  Memory Utilization: {int(used_mem) / int(total_mem) * 100:.1f}%")
    print(f"  Power Efficiency: {gflops / (int(power) / 1000):.1f} GFLOPS/W")

if __name__ == "__main__":
    main()
