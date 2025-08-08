# CPU–GPU Pipeline Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3, Python 3.11, PyTorch nightly

## Build (C++)
```bash
make
```

## Run (C++)
```bash
./cpu_bottleneck_naive
./cpu_bottleneck_pipelined
```

## Run (Python)
```bash
python cpu_bottleneck_naive.py
python cpu_bottleneck_pipelined.py
```

## Profile
```bash
nsys profile --trace=cuda --output pipeline_report ./cpu_bottleneck_pipelined
ncu --set full --output pipeline_ncu_report ./cpu_bottleneck_pipelined
```

### Sample Metrics

| Metric                      | Naive | Pipelined |
|-----------------------------|-------|-----------|
| CPU Idle Time               | Low   | Moderate  |
| GPU Idle Time               | High  | Low       |
| Total Completion Time       | 120 ms| 80 ms     |
