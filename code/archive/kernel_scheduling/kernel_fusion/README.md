# Kernel Fusion Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3, Python 3.11, PyTorch nightly

## Build (C++)
```bash
make
```

## Run (C++)
```bash
./add_mul_naive
./fused_add_mul
```

## Run (Python)
```bash
python add_mul_naive.py
python fused_add_mul_ext.py
```

## Profile
```bash
nsys profile --trace=cuda --output fusion_report ./fused_add_mul
ncu --set full --output fusion_ncu_report ./fused_add_mul
```

### Sample Metrics

| Metric                  | Naive | Fused  |
|-------------------------|-------|--------|
| Kernel Launches         | 2     | 1      |
| Total Time              | 20 ms | 12 ms  |
