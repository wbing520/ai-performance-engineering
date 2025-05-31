# Dynamic Parallelism Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3

## Build
```bash
make
```

## Run
```bash
./dp_host_launched
./dp_device_launched
```

## Profile
```bash
nsys profile --trace=cuda --output dp_report ./dp_device_launched
ncu --set full --output dp_ncu_report ./dp_device_launched
```

### Sample Metrics

| Metric                      | Host Launch | Device Launch |
|-----------------------------|-------------|---------------|
| Kernel Launch Overhead      | High        | Low           |
| Total Execution Time        | 100 ms      | 80 ms         |
