#!/usr/bin/env python3
"""
Chapter 13: DeepSeek-V3 Training with Profiling
Focused example demonstrating profiling techniques for MoE models
"""

import time
import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_model_and_data():
    """Set up DeepSeek-V3 model and training data"""
    print("Setting up DeepSeek-V3 model...")
    
    model_name = "deepseek-ai/DeepSeek-V3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    batch_size = 4
    input_texts = ["DeepSeek is great."] * batch_size
    enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    labels = input_ids.clone()  # For LM training, labels are the inputs
    
    return model, optimizer, input_ids, attention_mask, labels

def warm_up_model(model, optimizer, input_ids, attention_mask, labels):
    """Warm up the model to avoid capturing one-time setup costs"""
    print("Warming up model...")
    for _ in range(5):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    print("Warm-up complete")

def profile_with_pytorch_profiler(model, optimizer, input_ids, attention_mask, labels):
    """Profile using PyTorch Profiler with NVTX markers"""
    print("\n=== PyTorch Profiler Analysis ===")
    
    with profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,  # record tensor shapes
        profile_memory=True,  # track GPU memory usage per op
        with_stack=True,  # enable stack tracing
        with_flops=True  # capture FLOPs counters
    ) as prof:
        with profiler.record_function("train_step"):
            # Forward pass
            torch.cuda.nvtx.range_push("forward")
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            torch.cuda.nvtx.range_pop()  # end of forward
            
            # Backward pass and optimization
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_push("optimizer_step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()  # end of optimizer_step
            optimizer.zero_grad()
            torch.cuda.nvtx.range_pop()  # end of backward
    
    # Print top operations by CUDA time
    print("\nTop 10 operations by CUDA execution time:")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=10,
        fields=["self_cuda_time_total", "calls"]
    ))
    
    return prof

def analyze_memory_usage(model, optimizer, input_ids, attention_mask, labels):
    """Analyze memory usage during training"""
    print("\n=== Memory Usage Analysis ===")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Run a few iterations and track memory
    for i in range(3):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print memory stats after each iteration
        stats = torch.cuda.memory_stats()
        allocated_gb = stats['allocated_bytes.all.current'] / 1024**3
        reserved_gb = stats['reserved_bytes.all.current'] / 1024**3
        peak_gb = stats['allocated_bytes.all.peak'] / 1024**3
        
        print(f"Iteration {i+1}:")
        print(f"  Allocated: {allocated_gb:.2f} GB")
        print(f"  Reserved: {reserved_gb:.2f} GB")
        print(f"  Peak: {peak_gb:.2f} GB")

def compare_eager_vs_compiled(model, optimizer, input_ids, attention_mask, labels):
    """Compare eager mode vs compiled execution"""
    print("\n=== Eager vs Compiled Mode Comparison ===")
    
    # Make runs deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Eager mode timing
    optimizer.zero_grad()
    torch.cuda.synchronize()
    start = time.time()
    
    # Forward + backward + optimize (Eager)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    eager_time = end - start
    print(f"Eager-mode step time: {eager_time:.4f} s")
    
    # Warm up for compilation
    print("Warming up for compilation...")
    compiled_model = torch.compile(model, mode="max-autotune")
    for _ in range(3):  # Warm up compiled model
        outputs = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Compiled mode timing
    optimizer.zero_grad()
    torch.cuda.synchronize()
    start = time.time()
    
    # Forward + backward + optimize (Compiled)
    outputs = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    compiled_time = end - start
    print(f"Compiled-mode step time: {compiled_time:.4f} s")
    
    speedup = eager_time / compiled_time
    print(f"Speedup: {speedup:.2f}x")
    
    return eager_time, compiled_time, speedup

def demonstrate_attention_optimizations():
    """Demonstrate PyTorch Attention optimization techniques"""
    print("\n=== Attention Optimization Techniques ===")
    
    # Create sample data for attention
    batch_size, seq_len, num_heads, head_dim = 2, 512, 16, 64
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    
    print("Attention optimization techniques:")
    print("1. Scaled Dot Product Attention (SPDA)")
    print("   - Use torch.nn.functional.scaled_dot_product_attention")
    print("   - Automatically uses fastest available kernel (e.g. FlashAttention)")
    
    print("\n2. FlexAttention")
    print("   - Compiler-based approach for custom sparsity patterns")
    print("   - Can be 2x faster for specific sparse attention patterns")
    
    print("\n3. FlexDecoding")
    print("   - Optimizes decoder side of sequence generation")
    print("   - Uses KV caching efficiently across timesteps")
    
    print("\n4. Context Parallel")
    print("   - Parallelizes attention across multiple implementations")
    print("   - Uses torch.context_parallel() context manager")
    
    # Demonstrate SPDA
    print("\nDemonstrating SPDA...")
    torch.cuda.synchronize()
    start = time.time()
    
    # Standard attention implementation
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    standard_output = torch.matmul(attn_weights, v)
    
    torch.cuda.synchronize()
    standard_time = time.time() - start
    
    # SPDA implementation
    torch.cuda.synchronize()
    start = time.time()
    
    spda_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    torch.cuda.synchronize()
    spda_time = time.time() - start
    
    print(f"Standard attention time: {standard_time:.4f} s")
    print(f"SPDA time: {spda_time:.4f} s")
    print(f"SPDA speedup: {standard_time / spda_time:.2f}x")

def demonstrate_quantization():
    """Demonstrate torch.ao quantization techniques"""
    print("\n=== Quantization with torch.ao ===")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024)
    ).cuda()
    
    print("torch.ao quantization features:")
    print("1. Post-training quantization (PTQ)")
    print("2. Quantization-aware training (QAT)")
    print("3. QConfigMapping APIs")
    print("4. Support for INT8, FP8, and emerging formats")
    
    print("\nQuantization benefits:")
    print("- Reduced memory usage")
    print("- Faster inference on supported hardware")
    print("- Maintained accuracy with proper calibration")
    
    # Show model size before quantization
    total_params = sum(p.numel() for p in model.parameters())
    fp32_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per FP32 parameter
    int8_size_mb = total_params * 1 / 1024 / 1024   # 1 byte per INT8 parameter
    
    print(f"\nModel size comparison:")
    print(f"FP32: {fp32_size_mb:.2f} MB")
    print(f"INT8: {int8_size_mb:.2f} MB")
    print(f"Memory reduction: {fp32_size_mb / int8_size_mb:.1f}x")

def main():
    """Main function demonstrating DeepSeek profiling"""
    print("Chapter 13: DeepSeek-V3 Training with Profiling")
    print("=" * 60)
    
    # Set up model and data
    model, optimizer, input_ids, attention_mask, labels = setup_model_and_data()
    
    # Warm up the model
    warm_up_model(model, optimizer, input_ids, attention_mask, labels)
    
    # Run profiling examples
    prof = profile_with_pytorch_profiler(model, optimizer, input_ids, attention_mask, labels)
    analyze_memory_usage(model, optimizer, input_ids, attention_mask, labels)
    eager_time, compiled_time, speedup = compare_eager_vs_compiled(
        model, optimizer, input_ids, attention_mask, labels
    )
    demonstrate_attention_optimizations()
    demonstrate_quantization()
    
    print("\n" + "=" * 60)
    print("DeepSeek profiling completed!")
    print(f"Eager time: {eager_time:.4f}s")
    print(f"Compiled time: {compiled_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print("\nKey takeaways:")
    print("- Use PyTorch Profiler to identify bottlenecks in MoE models")
    print("- Enable torch.compile for automatic optimizations")
    print("- Monitor memory usage for large models")
    print("- Use attention optimizations for transformer models")
    print("- Consider quantization for deployment")

if __name__ == "__main__":
    main()
