# jit_threshold_op.py
# Chapter 8: PyTorch compiled version using torch.compile

import torch
import time

# PyTorch 2.8 compiled function
@torch.compile(fullgraph=True)
def threshold_op(X):
    return torch.maximum(X, torch.zeros_like(X))

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
    
    # Warm up compilation
    Y = threshold_op(X)
    torch.cuda.synchronize()
    
    # Time the compiled operation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    Y = threshold_op(X)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"PyTorch compiled threshold operation time: {elapsed_time:.4f} ms")
    
    # Verify results
    expected = torch.where(X > 0, X, torch.zeros_like(X))
    correct = torch.allclose(Y, expected)
    print(f"Results: {'PASS' if correct else 'FAIL'}")

if __name__ == "__main__":
    main()
