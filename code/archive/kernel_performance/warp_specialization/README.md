# Warp Specialization

## Files
- `add_naive.cu`
- `add_specialized.cu`
- `Makefile`, `run_nsys.sh`, `run_ncu.sh`

## Requirements
- GPU: GB200/H100
- CUDA:13.0, C++17

## Build & Run
```bash
cd warp_specialization
make
./add_naive
./add_specialized
```

## Profiling
```bash
./run_nsys.sh
./run_ncu.sh
```