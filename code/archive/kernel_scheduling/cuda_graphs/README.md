# CUDA Graph Batching Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3, Python 3.11, PyTorch nightly

## Build (C++)
```bash
make
```

## Run (C++)
```bash
./multi_op_naive
./multi_op_graph
```

## Run (Python)
```bash
python multi_op_naive.py
python multi_op_graph.py
```

## Profile
```bash
nsys profile --trace=cuda --output graph_report ./multi_op_graph
ncu --set full --output graph_ncu_report ./multi_op_graph
```

### Sample Metrics

| Metric                  | Naive | Graph  |
|-------------------------|-------|--------|
| Kernel Launches         | 10    | 1      |
| Total Time              | 50 ms | 20 ms  |
