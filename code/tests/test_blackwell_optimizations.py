"""
Comprehensive Test Suite for Blackwell Optimizations
===================================================

This module provides unit tests, integration tests, and performance
regression tests for all Blackwell optimizations.

Test Categories:
1. Correctness tests (numerical accuracy)
2. Performance tests (speedup validation)
3. Memory tests (usage validation)
4. Integration tests (end-to-end)

Requirements:
- pytest
- PyTorch 2.9+
- Blackwell B200/B300 (for full tests)

Usage:
    pytest test_blackwell_optimizations.py -v
    pytest test_blackwell_optimizations.py -k test_fp8  # Run FP8 tests only

Author: Blackwell Optimization Project
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules to test
try:
    from ch19.native_fp8_training import FP8Linear, FP8ScalingManager
    from ch16.inference_optimizations_blackwell import (
        DynamicQuantizedKVCache,
        OptimizedDecoderLayer,
    )
    FP8_AVAILABLE = True
except (ImportError, AttributeError):
    FP8_AVAILABLE = False

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE_NAME = torch.cuda.get_device_name(0)
    IS_BLACKWELL = "B200" in DEVICE_NAME or "B300" in DEVICE_NAME
else:
    IS_BLACKWELL = False


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get device for testing"""
    return "cuda" if CUDA_AVAILABLE else "cpu"


@pytest.fixture
def small_tensor(device):
    """Small tensor for quick tests"""
    return torch.randn(4, 32, 64, device=device, dtype=torch.float32)


@pytest.fixture
def medium_tensor(device):
    """Medium tensor for standard tests"""
    return torch.randn(8, 128, 512, device=device, dtype=torch.float32)


# ============================================================================
# 1. Correctness Tests
# ============================================================================

class TestNumericalCorrectness:
    """Test numerical accuracy of optimizations"""
    
    @pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 not available")
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_fp8_linear_accuracy(self):
        """Test FP8 linear layer maintains reasonable accuracy"""
        device = "cuda"
        in_features, out_features = 256, 128
        batch_size = 16
        
        # Create FP16 baseline
        fp16_linear = nn.Linear(in_features, out_features).to(device).half()
        
        # Create FP8 version
        fp8_linear = FP8Linear(in_features, out_features).to(device)
        fp8_linear.weight.data = fp16_linear.weight.data.clone()
        fp8_linear.bias.data = fp16_linear.bias.data.clone()
        
        # Test input
        x = torch.randn(batch_size, in_features, device=device, dtype=torch.float16)
        
        # Forward pass
        with torch.no_grad():
            y_fp16 = fp16_linear(x)
            y_fp8 = fp8_linear(x)
        
        # Check relative error
        rel_error = (y_fp16 - y_fp8).abs() / (y_fp16.abs() + 1e-5)
        mean_rel_error = rel_error.mean().item()
        
        print(f"\nFP8 vs FP16 relative error: {mean_rel_error:.6f}")
        
        # Assert accuracy within tolerance
        # FP8 E4M3 has only 3 mantissa bits vs FP16's 10 bits
        # 10-20% error is expected even with hardware support
        assert mean_rel_error < 0.20, f"FP8 error too high: {mean_rel_error}"
    
    @pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 not available")
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_kv_cache_correctness(self):
        """Test KV cache returns correct values"""
        device = "cuda"
        batch_size, num_heads, seq_len, head_dim = 1, 8, 32, 64
        
        # Create cache
        cache = DynamicQuantizedKVCache(
            num_layers=1,
            max_batch_size=batch_size,
            max_seq_len=seq_len * 2,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )
        
        # Create test tensors
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Store in cache
        cached_key, cached_value = cache.update(0, key, value, 0)
        
        # Check shapes
        assert cached_key.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cached_value.shape == (batch_size, num_heads, seq_len, head_dim)
        
        # Check numerical accuracy (should be close after FP8 quantization)
        key_error = (key - cached_key[0]).abs().mean().item()
        value_error = (value - cached_value[0]).abs().mean().item()
        
        print(f"\nKV cache errors - Key: {key_error:.6f}, Value: {value_error:.6f}")
        
        # Tolerance is higher for FP8
        assert key_error < 0.1, f"Key error too high: {key_error}"
        assert value_error < 0.1, f"Value error too high: {value_error}"


# ============================================================================
# 2. Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance improvements"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.slow
    def test_torch_compile_speedup(self):
        """Test torch.compile provides speedup"""
        device = "cuda"
        batch_size, seq_len, d_model = 4, 128, 512
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        ).to(device)
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark baseline
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        baseline_time = start.elapsed_time(end) / 100
        
        # Compile model
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        # Warmup compiled
        for _ in range(10):
            _ = compiled_model(x)
        torch.cuda.synchronize()
        
        # Benchmark compiled
        start.record()
        for _ in range(100):
            _ = compiled_model(x)
        end.record()
        torch.cuda.synchronize()
        compiled_time = start.elapsed_time(end) / 100
        
        speedup = baseline_time / compiled_time
        print(f"\ntorch.compile speedup: {speedup:.2f}x")
        print(f"Baseline: {baseline_time:.2f}ms, Compiled: {compiled_time:.2f}ms")
        
        # Assert at least 10% improvement
        assert speedup > 1.1, f"torch.compile speedup too low: {speedup:.2f}x"
    
    @pytest.mark.skipif(not FP8_AVAILABLE, reason="FP8 not available")
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.slow
    def test_fp8_memory_savings(self):
        """Test FP8 reduces memory usage"""
        device = "cuda"
        batch_size, num_heads, seq_len, head_dim = 4, 32, 4096, 128
        
        # Allocate FP16 cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cache_fp16 = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device,
            dtype=torch.float16
        )
        fp16_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Allocate FP8 cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            cache_fp8 = torch.randn(
                batch_size, num_heads, seq_len, head_dim,
                device=device,
                dtype=torch.float8_e4m3fn
            )
            fp8_memory = torch.cuda.max_memory_allocated() / 1e9
            
            memory_saved = fp16_memory - fp8_memory
            percent_saved = (memory_saved / fp16_memory) * 100
            
            print(f"\nMemory usage:")
            print(f"  FP16: {fp16_memory:.3f} GB")
            print(f"  FP8:  {fp8_memory:.3f} GB")
            print(f"  Saved: {memory_saved:.3f} GB ({percent_saved:.1f}%)")
            
            # Assert approximately 50% savings
            assert percent_saved > 40, f"Memory savings too low: {percent_saved:.1f}%"
        except RuntimeError:
            pytest.skip("FP8 tensor creation not supported")


# ============================================================================
# 3. Memory Tests
# ============================================================================

class TestMemoryManagement:
    """Test memory management and limits"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_no_memory_leak(self):
        """Test for memory leaks in repeated operations"""
        device = "cuda"
        
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Perform operations
        for _ in range(100):
            x = torch.randn(1024, 1024, device=device)
            y = x @ x.T
            del x, y
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase after 100 iterations: {memory_increase / 1e6:.2f} MB")
        
        # Assert no significant memory increase
        assert memory_increase < 10 * 1e6, f"Possible memory leak: {memory_increase / 1e6:.2f} MB increase"


# ============================================================================
# 4. Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_optimized_decoder_layer(self):
        """Test complete optimized decoder layer"""
        device = "cuda"
        batch_size, seq_len, d_model = 2, 64, 512
        num_heads = 8
        
        # Create layer
        layer = OptimizedDecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
        )
        
        # Test input
        hidden_states = torch.randn(
            batch_size, seq_len, d_model,
            device=device,
            dtype=torch.float32
        )
        
        # Forward pass
        output = layer(hidden_states)
        
        # Check output shape
        assert output.shape == hidden_states.shape
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        print(f"\n Decoder layer test passed")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")


# ============================================================================
# 5. Blackwell-Specific Tests
# ============================================================================

class TestBlackwellFeatures:
    """Tests specific to Blackwell B200/B300"""
    
    @pytest.mark.skipif(not IS_BLACKWELL, reason="Requires Blackwell GPU")
    def test_blackwell_detection(self):
        """Test Blackwell GPU is detected correctly"""
        assert CUDA_AVAILABLE
        assert "B200" in DEVICE_NAME or "B300" in DEVICE_NAME
        
        prop = torch.cuda.get_device_properties(0)
        assert prop.major == 10  # Blackwell compute capability
        assert prop.minor == 0
        
        print(f"\n Blackwell GPU detected: {DEVICE_NAME}")
        print(f"  Compute Capability: {prop.major}.{prop.minor}")
        print(f"  Memory: {prop.total_memory / 1e9:.1f} GB")
    
    @pytest.mark.skipif(not IS_BLACKWELL, reason="Requires Blackwell GPU")
    def test_hbm3e_bandwidth(self):
        """Test HBM3e bandwidth is achievable"""
        device = "cuda"
        size = 1024 * 1024 * 1024  # 1GB
        
        # Allocate tensors
        x = torch.randn(size // 4, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        # Warmup
        for _ in range(10):
            y.copy_(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 100
        start.record()
        for _ in range(iterations):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        bandwidth_gbs = (size * iterations / elapsed_ms) / 1e6
        
        print(f"\nHBM3e bandwidth test:")
        print(f"  Achieved: {bandwidth_gbs:.1f} GB/s")
        print(f"  Target: ~8000 GB/s (8 TB/s)")
        print(f"  Utilization: {bandwidth_gbs / 8000 * 100:.1f}%")
        
        # Assert reasonable bandwidth (at least 50% of peak)
        # B200 HBM3e theoretical: 8 TB/s, expect > 4 TB/s achievable
        assert bandwidth_gbs > 4000, f"Bandwidth too low: {bandwidth_gbs:.1f} GB/s"


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "blackwell: marks tests that require Blackwell GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    skip_blackwell = pytest.mark.skip(reason="Blackwell GPU not available")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    
    for item in items:
        if "blackwell" in item.keywords and not IS_BLACKWELL:
            item.add_marker(skip_blackwell)
        if "cuda" in str(item.fspath) and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=== Blackwell Optimization Test Suite ===\n")
    
    print("Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {DEVICE_NAME}")
        print(f"  Blackwell: {IS_BLACKWELL}")
    print(f"  FP8 available: {FP8_AVAILABLE}")
    
    print("\nTo run tests:")
    print("  pytest test_blackwell_optimizations.py -v")
    print("  pytest test_blackwell_optimizations.py -m 'not slow'  # Skip slow tests")
    print("  pytest test_blackwell_optimizations.py -k test_fp8    # Run FP8 tests only")
    
    print("\nTest Categories:")
    print("  1. Correctness - Numerical accuracy validation")
    print("  2. Performance - Speedup validation")
    print("  3. Memory - Memory usage validation")
    print("  4. Integration - End-to-end tests")
    print("  5. Blackwell - Hardware-specific tests")
    print("  6. Regression - Performance regression detection")
    print("  7. Stress - Large-scale stress tests")


# ============================================================================
# 6. Performance Regression Tests
# ============================================================================

class TestPerformanceRegression:
    """Test for performance regressions"""
    
    # Expected performance baselines (measured on Blackwell B200)
    BASELINES = {
        'fp8_linear_speedup': 1.4,  # FP8 should be 1.4x+ faster than FP16
        'compiled_speedup': 1.25,   # torch.compile should be 1.25x+ faster
        'kv_cache_memory_ratio': 0.55,  # KV cache should use <55% memory
        'inference_latency_ms': 50,  # Inference should be <50ms for batch=8
    }
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or not FP8_AVAILABLE, 
                        reason="Requires CUDA and FP8")
    @pytest.mark.slow
    def test_fp8_linear_regression(self):
        """Test FP8 linear doesn't regress in performance"""
        device = "cuda"
        in_features, out_features = 2048, 2048
        batch_size = 64
        iterations = 100
        
        # Create layers
        fp16_linear = nn.Linear(in_features, out_features).to(device).half()
        fp8_linear = FP8Linear(in_features, out_features).to(device)
        
        # Test input
        x_fp16 = torch.randn(batch_size, in_features, device=device, dtype=torch.float16)
        x_fp8 = x_fp16.clone()
        
        # Warmup
        for _ in range(10):
            _ = fp16_linear(x_fp16)
            _ = fp8_linear(x_fp8)
        torch.cuda.synchronize()
        
        # Benchmark FP16
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            _ = fp16_linear(x_fp16)
        end.record()
        torch.cuda.synchronize()
        fp16_time = start.elapsed_time(end) / iterations
        
        # Benchmark FP8
        start.record()
        for _ in range(iterations):
            _ = fp8_linear(x_fp8)
        end.record()
        torch.cuda.synchronize()
        fp8_time = start.elapsed_time(end) / iterations
        
        speedup = fp16_time / fp8_time
        
        print(f"\nFP8 Linear Performance:")
        print(f"  FP16 time: {fp16_time:.3f} ms")
        print(f"  FP8 time: {fp8_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Baseline: {self.BASELINES['fp8_linear_speedup']:.2f}x")
        
        # Assert no regression
        assert speedup >= self.BASELINES['fp8_linear_speedup'] * 0.9, \
            f"Performance regression: {speedup:.2f}x < {self.BASELINES['fp8_linear_speedup']:.2f}x"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA")
    @pytest.mark.slow
    def test_torch_compile_regression(self):
        """Test torch.compile doesn't regress"""
        device = "cuda"
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).to(device)
        
        # Compiled version
        compiled_model = torch.compile(model, mode="max-autotune")
        
        # Test input
        x = torch.randn(32, 512, device=device)
        
        # Warmup
        for _ in range(10):
            _ = model(x)
            _ = compiled_model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = 100
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Eager mode
        start.record()
        for _ in range(iterations):
            _ = model(x)
        end.record()
        torch.cuda.synchronize()
        eager_time = start.elapsed_time(end) / iterations
        
        # Compiled mode
        start.record()
        for _ in range(iterations):
            _ = compiled_model(x)
        end.record()
        torch.cuda.synchronize()
        compiled_time = start.elapsed_time(end) / iterations
        
        speedup = eager_time / compiled_time
        
        print(f"\ntorch.compile Performance:")
        print(f"  Eager time: {eager_time:.3f} ms")
        print(f"  Compiled time: {compiled_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Baseline: {self.BASELINES['compiled_speedup']:.2f}x")
        
        assert speedup >= self.BASELINES['compiled_speedup'] * 0.9, \
            f"Performance regression: {speedup:.2f}x < {self.BASELINES['compiled_speedup']:.2f}x"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or not FP8_AVAILABLE,
                        reason="Requires CUDA and FP8")
    def test_kv_cache_memory_regression(self):
        """Test KV cache memory usage doesn't regress"""
        device = "cuda"
        batch_size, seq_len, num_heads, head_dim = 8, 2048, 32, 128
        
        # Standard cache
        k_standard = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v_standard = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        standard_bytes = k_standard.element_size() * k_standard.numel() * 2
        
        # Quantized cache
        cache = DynamicQuantizedKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            device=device,
            dtype=torch.float16  # Will use FP8 if available
        )
        
        # Update cache (layer 0, batch 0)
        for b in range(min(batch_size, 1)):  # Test with first batch only
            cache.update(0, k_standard[b:b+1], v_standard[b:b+1], batch_idx=b)
        
        # Calculate memory usage
        quantized_bytes = cache.cache.numel() * cache.cache.element_size()
        ratio = quantized_bytes / standard_bytes
        
        print(f"\nKV Cache Memory:")
        print(f"  Standard: {standard_bytes / 1e6:.2f} MB")
        print(f"  Quantized: {quantized_bytes / 1e6:.2f} MB")
        print(f"  Ratio: {ratio:.2%}")
        print(f"  Baseline: <{self.BASELINES['kv_cache_memory_ratio']:.0%}")
        
        assert ratio <= self.BASELINES['kv_cache_memory_ratio'], \
            f"Memory regression: {ratio:.2%} > {self.BASELINES['kv_cache_memory_ratio']:.0%}"


# ============================================================================
# 7. Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for large-scale scenarios"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA")
    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test large batch processing doesn't OOM"""
        device = "cuda"
        
        # Get available memory
        props = torch.cuda.get_device_properties(0)
        available_gb = props.total_memory / 1e9
        
        print(f"\nLarge Batch Stress Test:")
        print(f"  Available memory: {available_gb:.1f} GB")
        
        # Test with 80% memory usage target
        batch_size = 128
        seq_len = 2048
        hidden_dim = 4096
        
        model = nn.Linear(hidden_dim, hidden_dim).to(device)
        
        try:
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            y = model(x)
            
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  Batch size: {batch_size}")
            print(f"  Sequence length: {seq_len}")
            print(f"  Memory used: {allocated_gb:.2f} GB")
            print(f"  Utilization: {allocated_gb / available_gb * 100:.1f}%")
            print(f"   No OOM!")
            
            assert allocated_gb < available_gb * 0.95, "Memory usage too high"
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.fail(f"OOM with batch_size={batch_size}, seq_len={seq_len}")
            raise
        finally:
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or not FP8_AVAILABLE,
                        reason="Requires CUDA and FP8")
    @pytest.mark.slow
    def test_long_sequence_inference(self):
        """Test inference on very long sequences"""
        device = "cuda"
        
        # Simulate long-context inference (64K tokens)
        batch_size = 1
        seq_len = 65536
        num_heads = 32
        head_dim = 128
        
        print(f"\nLong Sequence Inference Stress Test:")
        print(f"  Sequence length: {seq_len:,}")
        print(f"  Heads: {num_heads}")
        print(f"  Head dim: {head_dim}")
        
        # Create KV cache
        cache = DynamicQuantizedKVCache(
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            device=device,
        )
        
        # Simulate token-by-token generation
        chunk_size = 2048
        num_chunks = seq_len // chunk_size
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        for i in range(num_chunks):
            k = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device)
            v = torch.randn(batch_size, num_heads, chunk_size, head_dim, device=device)
            
            cache.update(k, v, batch_indices=torch.zeros(batch_size, dtype=torch.long))
        
        end_event.record()
        torch.cuda.synchronize()
        
        total_time = start_event.elapsed_time(end_event)
        tokens_per_sec = seq_len / (total_time / 1000)
        memory_gb = cache.get_memory_usage() / 1e9
        
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"  Memory: {memory_gb:.2f} GB")
        print(f"   Completed successfully!")
        
        assert tokens_per_sec > 1000, f"Throughput too low: {tokens_per_sec:.0f} tokens/sec"
        assert memory_gb < 10, f"Memory usage too high: {memory_gb:.2f} GB"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA")
    @pytest.mark.slow
    def test_continuous_training(self):
        """Test continuous training without memory leaks"""
        device = "cuda"
        
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        
        print(f"\nContinuous Training Stress Test:")
        
        initial_memory = torch.cuda.memory_allocated() / 1e6
        
        # Run 1000 training steps
        for step in range(1000):
            x = torch.randn(32, 512, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 200 == 0:
                current_memory = torch.cuda.memory_allocated() / 1e6
                print(f"  Step {step:4d}: {current_memory:.1f} MB")
        
        final_memory = torch.cuda.memory_allocated() / 1e6
        memory_growth = final_memory - initial_memory
        
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        
        # Assert memory growth is reasonable (<50 MB absolute increase)
        # PyTorch caching allocator keeps some memory, so allow small growth
        assert memory_growth < 50, \
            f"Possible memory leak: {memory_growth:.1f} MB growth"
        
        print(f"   No memory leak detected!")


# ============================================================================
# 8. End-to-End Integration Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or not FP8_AVAILABLE,
                        reason="Requires CUDA and FP8")
    @pytest.mark.slow
    def test_complete_training_pipeline(self):
        """Test complete training pipeline with all optimizations"""
        device = "cuda"
        
        print(f"\nComplete Training Pipeline Test:")
        
        # Create model with FP8
        model = nn.Sequential(
            FP8Linear(512, 1024),
            nn.ReLU(),
            FP8Linear(1024, 512),
        ).to(device)
        
        # Compile
        compiled_model = torch.compile(model, mode="max-autotune")
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        num_steps = 50
        batch_size = 32
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        for step in range(num_steps):
            x = torch.randn(batch_size, 512, device=device)
            y = compiled_model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        end.record()
        torch.cuda.synchronize()
        
        total_time = start.elapsed_time(end)
        time_per_step = total_time / num_steps
        
        print(f"  Steps: {num_steps}")
        print(f"  Total time: {total_time:.2f} ms")
        print(f"  Time per step: {time_per_step:.2f} ms")
        print(f"   Pipeline completed successfully!")
        
        assert time_per_step < 100, f"Training too slow: {time_per_step:.2f} ms/step"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE or not FP8_AVAILABLE,
                        reason="Requires CUDA and FP8")
    def test_complete_inference_pipeline(self):
        """Test complete inference pipeline with all optimizations"""
        device = "cuda"
        
        print(f"\nComplete Inference Pipeline Test:")
        
        batch_size = 8
        seq_len = 512
        num_heads = 16
        head_dim = 64
        
        # Create decoder layer with all optimizations
        layer = OptimizedDecoderLayer(
            d_model=num_heads * head_dim,
            num_heads=num_heads,
            use_flex_attention=True,
        ).to(device)
        
        # Compile
        compiled_layer = torch.compile(layer, mode="max-autotune")
        
        # Test input
        x = torch.randn(batch_size, seq_len, num_heads * head_dim, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = compiled_layer(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 100
        start.record()
        with torch.no_grad():
            for _ in range(iterations):
                output = compiled_layer(x)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / iterations
        tokens_per_sec = (batch_size * seq_len) / (time_ms / 1000)
        
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Latency: {time_ms:.2f} ms")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
        print(f"   Inference pipeline working!")
        
        assert time_ms < 100, f"Inference too slow: {time_ms:.2f} ms"
        assert output.shape == x.shape


# Update main to include new test categories
if __name__ == "__main__":
    print("=== Blackwell Optimization Test Suite ===\n")
    
    print("Environment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"  GPU: {DEVICE_NAME}")
        print(f"  Blackwell: {IS_BLACKWELL}")
    print(f"  FP8 available: {FP8_AVAILABLE}")
    
    print("\nTo run tests:")
    print("  pytest test_blackwell_optimizations.py -v")
    print("  pytest test_blackwell_optimizations.py -m 'not slow'  # Skip slow tests")
    print("  pytest test_blackwell_optimizations.py -k test_fp8    # Run FP8 tests only")
    
    print("\nTest Categories:")
    print("  1. Correctness - Numerical accuracy validation")
    print("  2. Performance - Speedup validation")
    print("  3. Memory - Memory usage validation")
    print("  4. Integration - End-to-end tests")
    print("  5. Blackwell - Hardware-specific tests")
    print("  6. Regression - Performance regression detection")
    print("  7. Stress - Large-scale stress tests")
    print("  8. End-to-End - Complete pipeline validation")

