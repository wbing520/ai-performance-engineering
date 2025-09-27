"""Per-chapter profiler metric configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class ProfilerOverrides:
    """Overrides for profiler invocations."""

    ncu_metrics: List[str] = field(default_factory=list)
    nsys_trace: List[str] = field(default_factory=list)
    nsys_extra_args: List[str] = field(default_factory=list)
    pytorch_modes: List[str] = field(default_factory=list)

    def merge(self, other: "ProfilerOverrides") -> None:
        self._extend_unique(self.ncu_metrics, other.ncu_metrics)
        self._extend_unique(self.nsys_trace, other.nsys_trace)
        self._extend_unique(self.nsys_extra_args, other.nsys_extra_args)
        self._extend_unique(self.pytorch_modes, other.pytorch_modes)

    @staticmethod
    def _extend_unique(target: List[str], values: Iterable[str]) -> None:
        for value in values:
            if value not in target:
                target.append(value)


def _overrides_from_lists(
    *,
    ncu_metrics: Iterable[str] | None = None,
    nsys_trace: Iterable[str] | None = None,
    nsys_extra_args: Iterable[str] | None = None,
    pytorch_modes: Iterable[str] | None = None,
) -> ProfilerOverrides:
    override = ProfilerOverrides()
    if ncu_metrics:
        override.ncu_metrics.extend(ncu_metrics)
    if nsys_trace:
        override.nsys_trace.extend(nsys_trace)
    if nsys_extra_args:
        override.nsys_extra_args.extend(nsys_extra_args)
    if pytorch_modes:
        override.pytorch_modes.extend(pytorch_modes)
    return override


CONFIG_BY_TAG: Dict[str, ProfilerOverrides] = {
    "ch04": _overrides_from_lists(
        ncu_metrics=[
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch05": _overrides_from_lists(
        nsys_trace=["cuda", "nvtx", "osrt", "nvlink", "cublas", "cudnn"],
    ),
    "ch07": _overrides_from_lists(
        ncu_metrics=[
            "smsp__sass_average_branch_divergence.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "shared_load_sectors",
            "shared_store_sectors",
        ]
    ),
    "ch08": _overrides_from_lists(
        ncu_metrics=[
            "gpu__time_elapsed.avg",
            "smsp__sass_average_branch_divergence.pct",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch09": _overrides_from_lists(
        ncu_metrics=[
            "flop_count_sp",
            "flop_count_hp",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        ]
    ),
    "ch10": _overrides_from_lists(
        ncu_metrics=[
            "shared_load_sectors",
            "shared_store_sectors",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch11": _overrides_from_lists(
        ncu_metrics=[
            "shared_load_sectors",
            "shared_store_sectors",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
    "ch13": _overrides_from_lists(
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn", "nvlink"],
        pytorch_modes=["blackwell"],
    ),
    "ch18": _overrides_from_lists(
        ncu_metrics=[
            "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
        nsys_trace=["cuda", "nvtx", "osrt", "cublas", "cudnn"],
        nsys_extra_args=["--gpu-metrics-device=all"],
    ),
    "ch19": _overrides_from_lists(
        ncu_metrics=[
            "shared_load_sectors",
            "shared_store_sectors",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
    ),
}

CONFIG_BY_EXAMPLE: Dict[str, ProfilerOverrides] = {}


def resolve_overrides(example: "Example") -> ProfilerOverrides:
    result = ProfilerOverrides()

    for tag in example.tags:
        tag_key = tag.lower()
        override = CONFIG_BY_TAG.get(tag_key)
        if override is not None:
            result.merge(override)

    example_override = CONFIG_BY_EXAMPLE.get(example.name)
    if example_override is not None:
        result.merge(example_override)

    return result


__all__ = ["ProfilerOverrides", "resolve_overrides"]
