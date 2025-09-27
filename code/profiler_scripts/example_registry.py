"""Registry describing all runnable examples for profiling harnesses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Example:
    """Metadata wrapper for an example entry."""

    name: str
    path: Path
    description: str
    default_args: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    requires_modules: List[str] = field(default_factory=list)
    optional_modules: List[str] = field(default_factory=list)
    requires_commands: List[str] = field(default_factory=list)
    requires_cuda: bool = False
    min_cuda_gpus: int = 0
    allow_cpu_fallback: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None

    def resolved_path(self, repo_root: Optional[Path] = None) -> Path:
        base = repo_root if repo_root is not None else Path.cwd()
        return (base / self.path).resolve()


def _example(
    name: str,
    path: str,
    description: str,
    *,
    default_args: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    requires_modules: Optional[Iterable[str]] = None,
    optional_modules: Optional[Iterable[str]] = None,
    requires_commands: Optional[Iterable[str]] = None,
    requires_cuda: bool = False,
    min_cuda_gpus: int = 0,
    allow_cpu_fallback: bool = True,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> Example:
    return Example(
        name=name,
        path=Path(path),
        description=description,
        default_args=list(default_args or []),
        tags=sorted(set(tags or [])),
        requires_modules=sorted(set(requires_modules or [])),
        optional_modules=sorted(set(optional_modules or [])),
        requires_commands=sorted(set(requires_commands or [])),
        requires_cuda=requires_cuda,
        min_cuda_gpus=min_cuda_gpus,
        allow_cpu_fallback=allow_cpu_fallback,
        env=dict(env or {}),
        timeout_seconds=timeout_seconds,
    )


EXAMPLES: List[Example] = [
    _example(
        name="ch1_performance_basics",
        path="code/ch1/performance_basics.py",
        description="Goodput and profiling primer from Chapter 1.",
        tags=["ch01", "profiling", "training"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=False,
        allow_cpu_fallback=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch2_hardware_info",
        path="code/ch2/hardware_info.py",
        description="Hardware topology and monitoring walkthrough.",
        tags=["ch02", "system", "monitoring"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=False,
        allow_cpu_fallback=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch3_bind_numa_affinity",
        path="code/ch3/bind_numa_affinity.py",
        description="NUMA-aware data pipeline and affinity demo.",
        tags=["ch03", "system", "numa"],
        requires_modules=["torch", "psutil"],
        requires_commands=["numactl"],
        requires_cuda=True,
        allow_cpu_fallback=False,
        timeout_seconds=600,
    ),
    _example(
        name="ch4_before_dataparallel",
        path="code/ch4/before_dataparallel.py",
        description="Baseline single-GPU training before DataParallel.",
        tags=["ch04", "distributed", "baseline"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch4_before_no_overlap",
        path="code/ch4/before_no_overlap.py",
        description="Distributed training without communication overlap.",
        tags=["ch04", "distributed", "ddp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch4_before_reinit_comm",
        path="code/ch4/before_reinit_comm.py",
        description="Communication-heavy workload prior to comm optimizations.",
        tags=["ch04", "distributed", "communication"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch4_after_ddp",
        path="code/ch4/after_ddp.py",
        description="Optimized DDP setup with improved overlap.",
        tags=["ch04", "distributed", "ddp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch4_after_overlap_ddp",
        path="code/ch4/after_overlap_ddp.py",
        description="Gradient bucketing and comm overlap example.",
        tags=["ch04", "distributed", "overlap"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch4_after_reinit_comm",
        path="code/ch4/after_reinit_comm.py",
        description="Communication-optimized training loop post refactor.",
        tags=["ch04", "distributed", "communication"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch4_barrier_straggler",
        path="code/ch4/barrier_straggler.py",
        description="Barrier synchronization and straggler analysis.",
        tags=["ch04", "distributed", "synchronization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch4_dist_allreduce",
        path="code/ch4/dist_allreduce.py",
        description="Gloo and NCCL all-reduce benchmark.",
        tags=["ch04", "distributed", "collective"],
        requires_modules=["torch"],
        requires_cuda=True,
        default_args=["--data-size", "1048576"],
        timeout_seconds=600,
    ),
    _example(
        name="ch4_nccl_benchmark",
        path="code/ch4/nccl_benchmark.py",
        description="Comprehensive NCCL collective benchmark suite.",
        tags=["ch04", "distributed", "collective"],
        requires_modules=["torch"],
        requires_cuda=True,
        default_args=[
            "--world-size", "1",
            "--max-size", "32",
            "--warmup", "1",
            "--trials", "3",
            "--operation", "allreduce",
            "--dtype", "float16",
        ],
        timeout_seconds=900,
    ),
    _example(
        name="ch4_ucx_fragmentation",
        path="code/ch4/ucx_fragmentation.py",
        description="UCX fragmentation avoidance strategies.",
        tags=["ch04", "distributed", "memory"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch5_gpudirect_storage",
        path="code/ch5/gpudirect_storage_example.py",
        description="GPUDirect Storage pipeline and monitoring demo.",
        tags=["ch05", "io", "storage"],
        requires_modules=["torch", "psutil", "numpy"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch5_storage_io_optimization",
        path="code/ch5/storage_io_optimization.py",
        description="High-throughput DataLoader and storage optimizations.",
        tags=["ch05", "io", "dataloader"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch6_add_sequential",
        path="code/ch6/add_sequential.py",
        description="Sequential kernel launch anti-pattern.",
        tags=["ch06", "kernel", "baseline"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch6_add_parallel",
        path="code/ch6/add_parallel.py",
        description="Fused parallel kernel addition example.",
        tags=["ch06", "kernel", "optimization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch7_memory_access",
        path="code/ch7/memory_access_pytorch.py",
        description="Coalesced vs uncoalesced memory access patterns.",
        tags=["ch07", "memory", "access"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch7_vectorized",
        path="code/ch7/vectorized_pytorch.py",
        description="Vectorized operations for bandwidth efficiency.",
        tags=["ch07", "memory", "vectorization"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch7_lookup",
        path="code/ch7/lookup_pytorch.py",
        description="Embedding lookup optimization strategies.",
        tags=["ch07", "memory", "embedding"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch7_matmul",
        path="code/ch7/matmul_pytorch.py",
        description="Matmul optimization with tiling examples.",
        tags=["ch07", "compute", "matmul"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch8_warp_divergence",
        path="code/ch8/warp_divergence_pytorch.py",
        description="Warp divergence and control flow efficiency study.",
        tags=["ch08", "warp", "control-flow"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=600,
    ),
    _example(
        name="ch8_occupancy",
        path="code/ch8/occupancy_pytorch.py",
        description="Occupancy tuning and launch configuration demo.",
        tags=["ch08", "warp", "occupancy"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch8_ilp",
        path="code/ch8/ilp_pytorch.py",
        description="Instruction level parallelism experiment.",
        tags=["ch08", "warp", "ilp"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch9_fusion",
        path="code/ch9/fusion_pytorch.py",
        description="Kernel fusion and compiler-assisted optimizations.",
        tags=["ch09", "compiler", "fusion"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_custom_allocator",
        path="code/ch13/custom_allocator.py",
        description="Custom CUDA memory allocator strategies.",
        tags=["ch13", "memory", "allocator"],
        requires_modules=["torch"],
        requires_cuda=True,
    ),
    _example(
        name="ch13_memory_profiling",
        path="code/ch13/memory_profiling.py",
        description="Memory profiling workflow for large models.",
        tags=["ch13", "memory", "profiling"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_fsdp",
        path="code/ch13/fsdp_example.py",
        description="Fully Sharded Data Parallel training demo.",
        tags=["ch13", "distributed", "fsdp"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch13_train_deepseek_v3",
        path="code/ch13/train_deepseek_v3.py",
        description="Transformer fine-tuning workflow inspired by DeepSeek-V3.",
        tags=["ch13", "training", "llm"],
        requires_modules=["torch", "transformers"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch14_torch_compiler_examples",
        path="code/ch14/torch_compiler_examples.py",
        description="Torch compile and dynamo optimization suite.",
        tags=["ch14", "compiler", "torch-compile"],
        requires_modules=["torch", "psutil", "GPUtil"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch14_triton_examples",
        path="code/ch14/triton_examples.py",
        description="Custom Triton kernels and PyTorch integration.",
        tags=["ch14", "triton", "compiler"],
        requires_modules=["torch", "triton"],
        requires_cuda=True,
        timeout_seconds=900,
    ),
    _example(
        name="ch15_disaggregated_inference",
        path="code/ch15/disaggregated_inference.py",
        description="Disaggregated inference and pipeline parallel patterns.",
        tags=["ch15", "inference", "pipeline"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch16_radix_attention",
        path="code/ch16/radix_attention_example.py",
        description="Radix attention and fast prefill kernels.",
        tags=["ch16", "attention", "radix"],
        requires_modules=["torch", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch16_inference_profiling",
        path="code/ch16/inference_profiling.py",
        description="Inference profiling dashboard and trace capture.",
        tags=["ch16", "inference", "profiling"],
        requires_modules=["torch", "psutil", "GPUtil", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch17_dynamic_routing",
        path="code/ch17/dynamic_routing.py",
        description="Dynamic expert routing strategies for mixture-of-experts.",
        tags=["ch17", "routing", "moe"],
        requires_modules=["torch", "yaml"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch17_early_rejection",
        path="code/ch17/early_rejection.py",
        description="Early token rejection and speculative decoding.",
        tags=["ch17", "routing", "speculative"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch18_flexdecoding",
        path="code/ch18/flexdecoding_example.py",
        description="FlexAttention powered decoding pipeline.",
        tags=["ch18", "attention", "flex"],
        requires_modules=["torch", "numpy"],
        optional_modules=["flex_attention"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch18_test_flex_attention",
        path="code/ch18/test_flex_attention.py",
        description="Environment probe for flex_attention availability.",
        tags=["ch18", "attention", "flex"],
        requires_modules=["torch"],
        requires_cuda=False,
        allow_cpu_fallback=True,
    ),
    _example(
        name="ch19_adaptive_parallelism",
        path="code/ch19/adaptive_parallelism_strategy.py",
        description="Adaptive parallelism orchestration for LLM inference.",
        tags=["ch19", "scheduling", "inference"],
        requires_modules=["torch", "psutil"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch19_token_precision_switching",
        path="code/ch19/token_precision_switching.py",
        description="Token precision switching and dynamic quantization.",
        tags=["ch19", "quantization", "inference"],
        requires_modules=["torch", "transformers", "numpy"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
    _example(
        name="ch20_ai_kernel_generator",
        path="code/ch20/ai_kernel_generator.py",
        description="AI-assisted CUDA kernel generation workflow.",
        tags=["ch20", "automation", "kernel"],
        requires_modules=["torch"],
        requires_cuda=True,
        timeout_seconds=1200,
    ),
]


EXAMPLE_BY_NAME = {example.name: example for example in EXAMPLES}


def list_example_names() -> List[str]:
    return sorted(EXAMPLE_BY_NAME)
