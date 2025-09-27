# Profiling Metrics Manifest

This document enumerates the Nsight Compute / Nsight Systems counters required to reproduce the tables in *AI Systems Performance Engineering*. Each section lists the chapter/table and the metrics our scripts now collect.

## Chapter 4 – Distributed Networking
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- Nsight Systems timeline markers (backward/comm overlap)

## Chapter 5 – GPU Storage I/O
- Nsight Systems bandwidth traces, host/device memcpy rates (captured via `-t cuda,nvtx,osrt,nvlink`).

## Chapter 7 – Warp Divergence / Bank Conflicts
- `smsp__sass_average_branch_divergence.pct`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- Shared memory load/store sector counts for bank-conflict analysis.

## Chapter 8 – Warp Efficiency (Table 8-3)
- `gpu__time_elapsed.avg`
- `smsp__sass_average_branch_divergence.pct`
- `sm__warps_active.avg.pct_of_peak_sustained_active`

## Chapter 9 – Arithmetic Intensity
- `flop_count_sp`, `flop_count_hp`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- Roofline section export (Nsight Compute `--set full`).

## Chapters 10–11 – Pipeline & Warp Specialization
- `shared_load_sectors`, `shared_store_sectors`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`

## Chapter 13 – PyTorch Scaling
- Nsight Systems `cuda,nvtx,osrt,cublas,cudnn,nvlink`
- PyTorch profiler summary (dynamic version reporting).

## Chapter 18 – FlashMLA
- `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- Nsight Systems decode timeline (GPU metrics via `--gpu-metrics-device=all`).

## Chapter 19 – Adaptive Inference Kernels
- `shared_load_sectors`, `shared_store_sectors`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`

The enhanced `ncu_profile.sh` and `nsys_profile.sh` apply these counters globally; chapter-specific scripts can further narrow the metric set as needed.
