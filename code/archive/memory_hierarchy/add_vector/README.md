# Vector Addition Example

Performs C = A + B elementwise on the GPU and via PyTorch/Triton.

**Hardware:**  
- Grace-Blackwell GB100/GB200 (HBM3e @ 16 TB/s)  
  Fallback: Hopper H100 (HBM3 @ 3 TB/s)

**Software:**  
- CUDA 13.0, C++17  
- NVIDIA Driver ≥ 555.52.04  
- Nsight Systems 2025.2.1  
- Nsight Compute 2024.3  
- Python 3.11  
- PyTorch 2.8.0-nightly-cu130  
- Triton 3.3.0

---

## Build & Run (CUDA C++)
```bash
cd add_example
make
./add
```

## Profile
```bash
./run_nsys.sh
./run_ncu.sh
```

## Run PyTorch (with Triton)
```bash
conda create -n add-py311 python=3.11 -y
conda activate add-py311
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip install triton==3.3.1
python add.py
```

## Expected Metrics
| Implementation | Throughput (GB/s) | Kernel Time (ms) |
| --------------:| -----------------:| ----------------:|
| CUDA C++       | ~500              | ~1.2             |
| PyTorch/Triton | ~480              | ~1.3             |
