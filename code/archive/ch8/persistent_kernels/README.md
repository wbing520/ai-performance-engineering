# Persistent Kernels Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3, Python 3.11, PyTorch nightly

## Build (C++)
```bash
make
```

## Run (C++)
```bash
./compute_naive_loop  # naive multiple launches
./compute_persistent  # single persistent kernel
```

## Profile
```bash
nsys profile --trace=cuda --output persistent_report ./compute_persistent
ncu --set full --output persistent_ncu_report ./compute_persistent
```

## Python Extension
```bash
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu113
python persistent_example.py
```

### Sample Metrics

| Metric                | Naive (1000 iters) | Persistent |
|-----------------------|--------------------|------------|
| Kernel launches       | 1000               | 1          |
| Total time            | 500 ms             | 160 ms     |
| Shared memory reuse   | 0                  | High       |
