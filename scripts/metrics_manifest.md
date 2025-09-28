# Profiling Metrics Manifest

This manifest enumerates the profiler counters and tooling required to reproduce
the numeric tables and figures in *AI Systems Performance Engineering*. Every
chapter below lists the metrics collected by the harness after the latest updates.
When no special instrumentation is needed, the stock Nsight/PyTorch presets are
sufficient and the chapter entry notes that explicitly.

The shared defaults for Nsight Compute and Nsight Systems now live in
`metrics_config.BASE_NCU_METRICS` and
`metrics_config.BASE_NSYS_TRACE_MODULES`. Both the Python harness and the
stand-alone shell wrappers import these lists so the same counters are captured
regardless of how an example is executed.

## Chapter 1 – Goodput Measurement
- PyTorch profiler `full` preset (default) capturing CPU/GPU time, FLOPs, module
  attribution, and memory statistics.
- Nsight Compute/Systems not required for Table data (iterative throughput is
  computed in-script).

## Chapter 2 – Hardware Inventory and Monitoring
- Script emits metrics via `psutil`, `GPUtil`, and `torch.cuda` directly; no Nsight
  counters required beyond the default harness configuration.

## Chapter 3 – NUMA and Topology Awareness
- Relies on `numactl`, `nvidia-smi`, and OS telemetry. No additional profiler
  counters beyond defaults.

## Chapter 4 – Distributed Networking
- Nsight Compute: `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`.
- Nsight Systems: default CUDA/NVTX/OSRT/cuBLAS/cuDNN traces capture overlap
  markers referenced in Tables 4-1 to 4-5.

## Chapter 5 – GPU Storage I/O
- Nsight Systems trace modules include NVLink in addition to the CUDA defaults to
  reproduce Table 5-1 bandwidth numbers (trace set
  `cuda,nvtx,osrt,nvlink,cublas,cudnn`).

## Chapter 6 – CUDA Fundamentals & Resource Limits
- Tables capture architectural limits and simple timing comparisons; default Nsight
  presets suffice (no extra metrics).

## Chapter 7 – Memory Access Patterns & Tiling
- Nsight Compute: `smsp__sass_average_branch_divergence.pct`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`, `shared_load_sectors`,
  `shared_store_sectors`.
- Used for Tables 7-1 through 7-7 (coalescing, bank conflicts, TMA prefetching).

## Chapter 8 – Warp Scheduling & Occupancy
- Nsight Compute: `gpu__time_elapsed.avg`,
  `smsp__sass_average_branch_divergence.pct`, `sm__warps_active.avg.pct_of_peak_sustained_active`.
- Supports Tables 8-2 through 8-4 and divergence experiments in Table 8-3.

## Chapter 9 – Arithmetic Intensity & Roofline
- Nsight Compute: `flop_count_sp`, `flop_count_hp`,
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`.
- Harness still runs the `--set full` collection for roofline exports (Tables 9-1 and 9-2).

## Chapter 10 – Intra-Kernel Pipelining
- Nsight Compute: `shared_load_sectors`, `shared_store_sectors`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`.
- Used to compare naive, double-buffered, warp-specialized, and cluster pipelines
  (Tables 10-2 through 10-5).

## Chapter 11 – Multi-SM & Cluster Pipelines
- Same Nsight Compute counters as Chapter 10 to analyze DSMEM benefits in Table 11-1.

## Chapter 12 – CUDA Graphs & Dynamic Parallelism
- Nsight Systems: GPU metrics enabled (`--gpu-metrics-device=all`) and extended
  trace set (`cuda,nvtx,osrt,cublas,cudnn`) to capture idle gaps and launch counts in
  Tables 12-1 and 12-2.

## Chapter 13 – PyTorch Profiling & Compilation
- Nsight Systems: extended trace set including NVLink for timeline correlation.
- PyTorch profiler: runs both CLI-requested modes and the chapter’s `blackwell`
  preset to export CUDA-time/FLOP/module summaries (Tables 13-2–13-4).

## Chapter 14 – Compiler Diagnostics
- Focused on log collection (`TORCH_LOGS`, `TRITON_LOGGING`). No profiler counter
  overrides required; documentation tables (e.g., Table 14-1) are static references.

## Chapter 15 – Inference Parallelism Strategies
- Relies on architectural diagrams and scenario analysis. Default profiler settings
  are adequate.

## Chapter 16 – Inference Troubleshooting & Alerting
- Tables are driven by logging/monitoring frameworks (Prometheus, custom logs).
  Harness defaults suffice when demonstrations are profiled.

## Chapter 17 – Dynamic Routing for KV Cache
- Strategy tables (17-1 through 17-3) derive from routing heuristics; no additional
  Nsight or PyTorch counters are required beyond defaults.

## Chapter 18 – FlashMLA & Triton Kernels
- Nsight Compute: `lts__t_sectors.avg.pct_of_peak_sustained_elapsed`,
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`.
- Nsight Systems: GPU metrics enabled to capture decode timeline behavior
  (`--gpu-metrics-device=all`).

## Chapter 19 – Adaptive Precision & Parallelism
- Nsight Compute: `shared_load_sectors`, `shared_store_sectors`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__warps_active.avg.pct_of_peak_sustained_active` to analyze dynamic kernels
  (Table 19-3).

## Chapter 20 – AI-Assisted Kernel Generation
- Evaluations rely on harness default Nsight Compute/Systems collections and PyTorch
  profiler summaries; no extra counters beyond the global configuration.

The harness (`profile_harness.py`) now reads these chapter/tag requirements from
`metrics_config.py` and merges them with any command-line overrides, ensuring each
chapter’s code example collects the counters needed to rebuild the manuscript’s
profiling tables.
