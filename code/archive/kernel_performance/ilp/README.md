# ILP Examples

## Files
- `reduce_naive.cu`
- `reduce_unrolled.cu`
- `sum_pytorch.py`
- `Makefile`, `run_nsys.sh`, `run_ncu.sh`

## Requirements
- GPU: GB200/H100
- CUDA:13.0
- C++:17
- Python:3.11, PyTorch nightly>=2.8.0, Triton>=2.5.0

## Build & Run
```bash
cd ilp
make
./reduce_naive
./reduce_unrolled
python3 sum_pytorch.py
```

## Profiling
```bash
./run_nsys.sh
```