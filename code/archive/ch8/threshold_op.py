# threshold_op.py
# Chapter 8: PyTorch version of threshold operation

import torch
import time

def main():
    # Use updated PyTorch 2.8 features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    N = 1024 * 1024
    
    # Create input tensor (half positive, half negative for maximum divergence test)
    X = torch.empty(N, device=device)
    for i in range(N):
        X[i] = 1.0 if i % 2 == 0 else -1.0
    
    # Warm up
    Y = torch.maximum(X, torch.zeros_like(X))  # equivalent to Y = X > 0 ? X : 0
    torch.cuda.synchronize()
    
    # Time the operation
    start_time = time.time()
    
    # Use CUDA events for more accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    Y = torch.maximum(X, torch.zeros_like(X))
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"PyTorch threshold operation time: {elapsed_time:.4f} ms")
    
    # Verify results
    expected = torch.where(X > 0, X, torch.zeros_like(X))
    correct = torch.allclose(Y, expected)
    print(f"Results: {'PASS' if correct else 'FAIL'}")

if __name__ == "__main__":
    main()
