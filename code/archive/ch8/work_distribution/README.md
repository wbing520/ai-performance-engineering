# Work Distribution Example

**Hardware:** Grace-Blackwell GB100/GB200 (fallback: H100)  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3

## Build
```bash
make
```

## Run
```bash
./uneven_static   # static workload
./uneven_dynamic  # dynamic queue workload
```

## Profile (nsys)
```bash
nsys profile --trace=cuda --output work_dist_report ./uneven_dynamic
```

## Profile (ncu)
```bash
ncu --set full --output work_dist_ncu_report ./uneven_dynamic
```

### Sample Metrics

| Metric                  | Static | Dynamic |
|-------------------------|--------|---------|
| Achieved Occupancy      | 40%    | 85%     |
| SM Idle Cycles          | 50%    | 5%      |
| Execution Time          | 30 ms  | 15 ms   |
