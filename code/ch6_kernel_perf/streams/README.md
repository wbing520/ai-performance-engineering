# Streams Overlap

## Files
- `stream_overlap.cu`
- `Makefile`, `run_nsys.sh`, `run_ncu.sh`

## Requirements
- GPU: GB200/H100
- CUDA:13.0, C++17

## Build & Run
```bash
cd streams
make
./stream_overlap
```

## Profiling
```bash
./run_nsys.sh
```