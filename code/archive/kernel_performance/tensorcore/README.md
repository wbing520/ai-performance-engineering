# TensorCore Examples

## Files
- `matmul_naive_fp32.cu`
- `matmul_tensorcore_fp16.cu`
- `matrix_multiply_amp.py`
- `Makefile`, `run_nsys.sh`, `run_ncu.sh`, `run_pytorch_profiler.sh`

## Requirements
- Hardware: Grace-Blackwell (GB200) or fallback H100
- CUDA: 13.0
- C++: C++17
- Python: 3.11
- PyTorch nightly >=2.8.0
- OpenAI Triton >=2.5.0
- Nsight Systems 2025.2.1, Nsight Compute 2024.3

## Build & Run
```bash
cd tensorcore
make
./matmul_naive_fp32
./matmul_tensorcore_fp16
python3 matrix_multiply_amp.py
```

## Profiling
```bash
./run_nsys.sh
./run_ncu.sh
./run_pytorch_profiler.sh
```