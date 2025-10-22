# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/kr/ckrclktybe3dei3bjaqjsd3ynpfjvoetwixdgpfcjhvv355n7hsb.py
# Topologically Sorted Source Nodes: [x, x_1, x_2, query, multi_head_attention_forward, key, value], Original ATen: [aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   key => permute_1
#   multi_head_attention_forward => clone, clone_1, clone_2
#   query => permute
#   value => permute_2
#   x => add, add_1, mul, mul_1, rsqrt, sub, var_mean
#   x_1 => add_2, add_3, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   x_2 => add_4, add_5, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
# Graph fragment:
#   %arg2_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %getitem_3 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=getitem_3]
#   %buf4 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=buf4]
#   %getitem_5 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf7 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=buf7]
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg2_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg2_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%arg2_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %getitem_1), kwargs = {})
#   %add : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %add_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg1_1), kwargs = {})
#   %permute : Tensor "f32[512, 4, 2048][2048, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_1, [1, 0, 2]), kwargs = {})
#   %clone : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %getitem_3), kwargs = {})
#   %add_2 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg0_1), kwargs = {})
#   %add_3 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg1_1), kwargs = {})
#   %permute_1 : Tensor "f32[512, 4, 2048][2048, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_3, [1, 0, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_2 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg2_1, %getitem_5), kwargs = {})
#   %add_4 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_4 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_5 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %arg0_1), kwargs = {})
#   %add_5 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg1_1), kwargs = {})
#   %permute_2 : Tensor "f32[512, 4, 2048][2048, 1048576, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_5, [1, 0, 2]), kwargs = {})
#   %clone_2 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %getitem_1,%buf1,%getitem_3,%buf4,%getitem_5,%buf7,%clone,%clone_1,%clone_2
triton_red_fused_clone_native_layer_norm_transpose_0 = async_compile.triton('triton_red_fused_clone_native_layer_norm_transpose_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_layer_norm_transpose_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 6, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 117456896}}
)
@triton.jit
def triton_red_fused_clone_native_layer_norm_transpose_0(in_ptr0, in_ptr1, in_ptr2, out_ptr6, out_ptr7, out_ptr8, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(r0_mask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(r0_mask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(r0_mask & xmask, tmp2_weight_next, tmp2_weight)
    tmp3, tmp4, tmp5 = triton_helpers.welford(tmp2_mean, tmp2_m2, tmp2_weight, 1)
    tmp2 = tmp3[:, None]
    tmp6 = tmp4[:, None]
    tmp7 = tmp5[:, None]
    x2 = (xindex % 512)
    x3 = xindex // 512
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp8 - tmp2
        tmp10 = 2048.0
        tmp11 = (tmp6 / tmp10)
        tmp12 = 1e-05
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tl.store(out_ptr6 + (r0_1 + 2048*x3 + 8192*x2), tmp19, r0_mask & xmask)
        tl.store(out_ptr7 + (r0_1 + 2048*x3 + 8192*x2), tmp19, r0_mask & xmask)
        tl.store(out_ptr8 + (r0_1 + 2048*x3 + 8192*x2), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/oe/coee6vvcunhqidcktw5x42qt7s2nlinbnfmrjzryj6gdumdodm4d.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2
#   multi_head_attention_forward => add_6, add_7, add_8, mul_6, permute_6, permute_7, permute_8, split_1, view_1, view_3, view_5, view_6, view_7, view_8
# Graph fragment:
#   %buf10 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf10]
#   %arg4_1 : Tensor "f32[6144][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %split_1 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg4_1, 2048), kwargs = {})
#   %view_1 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [512, 4, 2048]), kwargs = {})
#   %add_6 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %getitem_9), kwargs = {})
#   %view_6 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [512, 64, 128]), kwargs = {})
#   %permute_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %mul_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, 0.08838834764831845), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, 0), kwargs = {})
#   %view_3 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [512, 4, 2048]), kwargs = {})
#   %add_7 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %getitem_10), kwargs = {})
#   %view_7 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_7, [512, 64, 128]), kwargs = {})
#   %permute_7 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_1 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_7, 0), kwargs = {})
#   %view_5 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [512, 4, 2048]), kwargs = {})
#   %add_8 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %getitem_11), kwargs = {})
#   %view_8 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_8, [512, 64, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_2 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_8, 0), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default, %unsqueeze_default_1, %unsqueeze_default_2, None, False), kwargs = {scale: 1.0})
#   return %buf15
triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 50364416}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + ((x2 % 2048)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.08838834764831845
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/tv/ctvkekbhc5iregjpuqyeztfpdl44jxlfrlqcn7tezj73dp6tqhlq.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2
#   multi_head_attention_forward => add_6, add_7, add_8, mul_6, permute_6, permute_7, permute_8, split_1, view_1, view_3, view_5, view_6, view_7, view_8
# Graph fragment:
#   %buf12 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf12]
#   %arg4_1 : Tensor "f32[6144][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %split_1 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg4_1, 2048), kwargs = {})
#   %view_1 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [512, 4, 2048]), kwargs = {})
#   %add_6 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %getitem_9), kwargs = {})
#   %view_6 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [512, 64, 128]), kwargs = {})
#   %permute_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %mul_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, 0.08838834764831845), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, 0), kwargs = {})
#   %view_3 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [512, 4, 2048]), kwargs = {})
#   %add_7 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %getitem_10), kwargs = {})
#   %view_7 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_7, [512, 64, 128]), kwargs = {})
#   %permute_7 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_1 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_7, 0), kwargs = {})
#   %view_5 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [512, 4, 2048]), kwargs = {})
#   %add_8 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %getitem_11), kwargs = {})
#   %view_8 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_8, [512, 64, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_2 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_8, 0), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default, %unsqueeze_default_1, %unsqueeze_default_2, None, False), kwargs = {scale: 1.0})
#   return %buf16
triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 50364416}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (2048 + ((x0 % 2048))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/x6/cx6pn63zpogcvbjtg7z627u6ohrhleqilvjv4vv6ka3v2jwdcdbz.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#    => _scaled_dot_product_efficient_attention_default, unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2
#   multi_head_attention_forward => add_6, add_7, add_8, mul_6, permute_6, permute_7, permute_8, split_1, view_1, view_3, view_5, view_6, view_7, view_8
# Graph fragment:
#   %buf14 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf14]
#   %arg4_1 : Tensor "f32[6144][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %split_1 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%arg4_1, 2048), kwargs = {})
#   %view_1 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [512, 4, 2048]), kwargs = {})
#   %add_6 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %getitem_9), kwargs = {})
#   %view_6 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_6, [512, 64, 128]), kwargs = {})
#   %permute_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %mul_6 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_6, 0.08838834764831845), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, 0), kwargs = {})
#   %view_3 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [512, 4, 2048]), kwargs = {})
#   %add_7 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %getitem_10), kwargs = {})
#   %view_7 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_7, [512, 64, 128]), kwargs = {})
#   %permute_7 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_1 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_7, 0), kwargs = {})
#   %view_5 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [512, 4, 2048]), kwargs = {})
#   %add_8 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %getitem_11), kwargs = {})
#   %view_8 : Tensor "f32[512, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_8, [512, 64, 128]), kwargs = {})
#   %permute_8 : Tensor "f32[64, 512, 128][128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %unsqueeze_default_2 : Tensor "f32[1, 64, 512, 128][8192, 128, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_8, 0), kwargs = {})
#   %_scaled_dot_product_efficient_attention_default : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%unsqueeze_default, %unsqueeze_default_1, %unsqueeze_default_2, None, False), kwargs = {scale: 1.0})
#   return %buf17
triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 50364416}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (4096 + ((x0 % 2048))), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/cr/ccrhsboexznzkevoypre2bc2gffoi6vekveqnoz7hiqto2ulvdja.py
# Topologically Sorted Source Nodes: [, multi_head_attention_forward, attn_out, x_3, layer_norm_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor_2
#   attn_out => permute_12
#   layer_norm_3 => add_10, add_11, mul_7, mul_8, rsqrt_3, sub_4, var_mean_3
#   multi_head_attention_forward => view_10
#   x_3 => add_9
# Graph fragment:
#   %arg2_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %buf23 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf23]
#   %arg6_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %getitem_13 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf25 : Tensor "f32[4, 512, 1][512, 1, 2048]cuda:0" = PlaceHolder[target=buf25]
#   %arg7_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %add_tensor_2 : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg6_1), kwargs = {})
#   %view_10 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [512, 4, 2048]), kwargs = {})
#   %permute_12 : Tensor "f32[4, 512, 2048][2048, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [1, 0, 2]), kwargs = {})
#   %add_9 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %permute_12), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_13), kwargs = {})
#   %add_10 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[4, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_7 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_8 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %arg7_1), kwargs = {})
#   %add_11 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %arg8_1), kwargs = {})
#   return %getitem_13,%buf25,%add_11
triton_red_fused_add_addmm_native_layer_norm_transpose_view_4 = async_compile.triton('triton_red_fused_add_addmm_native_layer_norm_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_native_layer_norm_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 0, 'r0_': 67133440}}
)
@triton.jit
def triton_red_fused_add_addmm_native_layer_norm_transpose_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 2048*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 2048*x1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_2 + 2048*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r0_2 + 2048*x1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 + tmp15
        tmp17 = tmp16 - tmp6
        tmp18 = 2048.0
        tmp19 = (tmp10 / tmp18)
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = libdevice.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tl.store(out_ptr2 + (r0_2 + 2048*x3), tmp27, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/rq/crqkeelewwzh4ocy7tnigzcht6uiya7lauzj57247fogsvokw4gg.py
# Topologically Sorted Source Nodes: [, input_1, input_2], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#    => add_tensor_1
#   input_1 => view_13
#   input_2 => add_12, erf, mul_10, mul_11, mul_9
# Graph fragment:
#   %buf28 : Tensor "f32[2048, 8192][8192, 1]cuda:0" = PlaceHolder[target=buf28]
#   %arg10_1 : Tensor "f32[8192][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %add_tensor_1 : Tensor "f32[2048, 8192][8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %arg10_1), kwargs = {})
#   %view_13 : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [4, 512, 8192]), kwargs = {})
#   %mul_9 : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_10 : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_10,), kwargs = {})
#   %add_12 : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_11 : Tensor "f32[4, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %add_12), kwargs = {})
#   return %mul_11
triton_poi_fused_addmm_gelu_view_5 = async_compile.triton('triton_poi_fused_addmm_gelu_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 201359360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8192)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output/mb/cmbdsir2iu53hmek36heicebuavcwodlajwsqpfxsnxff3csehjy.py
# Topologically Sorted Source Nodes: [, multi_head_attention_forward, attn_out, x_3, input_3, x_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
# Source node to ATen node mapping:
#    => add_tensor, add_tensor_2
#   attn_out => permute_12
#   input_3 => view_15
#   multi_head_attention_forward => view_10
#   x_3 => add_9
#   x_4 => add_13
# Graph fragment:
#   %arg2_1 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %buf23 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf23]
#   %arg6_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %buf30 : Tensor "f32[2048, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf30]
#   %arg12_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_2 : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %arg6_1), kwargs = {})
#   %view_10 : Tensor "f32[512, 4, 2048][8192, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [512, 4, 2048]), kwargs = {})
#   %permute_12 : Tensor "f32[4, 512, 2048][2048, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [1, 0, 2]), kwargs = {})
#   %add_9 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %permute_12), kwargs = {})
#   %add_tensor : Tensor "f32[2048, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg12_1), kwargs = {})
#   %view_15 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [4, 512, 2048]), kwargs = {})
#   %add_13 : Tensor "f32[4, 512, 2048][1048576, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_15), kwargs = {})
#   return %add_13
triton_poi_fused_add_addmm_transpose_view_6 = async_compile.triton('triton_poi_fused_add_addmm_transpose_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=148, cc=100, major=10, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_transpose_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '5AC442E20401AAF098CB07F9147F92FCB0886743B7EB61227AD1D05246D7E7B1', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 83902464}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_transpose_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2048)
    x1 = ((xindex // 2048) % 512)
    x2 = xindex // 1048576
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*x2 + 8192*x1), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x3), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')

def partition_0(args):
    arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
    args.clear()
    assert_size_stride(arg2_1, (4, 512, 2048), (1048576, 2048, 1))
    assert_size_stride(arg0_1, (2048, ), (1, ))
    assert_size_stride(arg1_1, (2048, ), (1, ))
    assert_size_stride(arg3_1, (6144, 2048), (2048, 1))
    assert_size_stride(arg4_1, (6144, ), (1, ))
    assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg6_1, (2048, ), (1, ))
    assert_size_stride(arg7_1, (2048, ), (1, ))
    assert_size_stride(arg8_1, (2048, ), (1, ))
    assert_size_stride(arg9_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg10_1, (8192, ), (1, ))
    assert_size_stride(arg11_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg12_1, (2048, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = empty_strided_cuda((512, 4, 2048), (8192, 2048, 1), torch.float32)
        buf11 = empty_strided_cuda((512, 4, 2048), (8192, 2048, 1), torch.float32)
        buf13 = empty_strided_cuda((512, 4, 2048), (8192, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2, query, multi_head_attention_forward, key, value], Original ATen: [aten.native_layer_norm, aten.transpose, aten.clone]
        # [Provenance debug handles] triton_red_fused_clone_native_layer_norm_transpose_0:1
        stream0 = get_raw_stream(0)
        triton_red_fused_clone_native_layer_norm_transpose_0.run(arg2_1, arg0_1, arg1_1, buf9, buf11, buf13, 2048, 2048, stream=stream0)
        del arg0_1
        del arg1_1
        buf10 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:8
        extern_kernels.mm(reinterpret_tensor(buf9, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg3_1, (2048, 2048), (1, 2048), 0), out=buf10)
        buf12 = reinterpret_tensor(buf9, (2048, 2048), (2048, 1), 0); del buf9  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:9
        extern_kernels.mm(reinterpret_tensor(buf11, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg3_1, (2048, 2048), (1, 2048), 4194304), out=buf12)
        buf14 = reinterpret_tensor(buf11, (2048, 2048), (2048, 1), 0); del buf11  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:10
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg3_1, (2048, 2048), (1, 2048), 8388608), out=buf14)
        del arg3_1
        del buf13
        buf15 = reinterpret_tensor(buf10, (1, 64, 512, 128), (4194304, 128, 8192, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1:2
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_1.run(buf15, arg4_1, 4194304, stream=stream0)
        buf16 = reinterpret_tensor(buf12, (1, 64, 512, 128), (4194304, 128, 8192, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2:3
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_2.run(buf16, arg4_1, 4194304, stream=stream0)
        buf17 = reinterpret_tensor(buf14, (1, 64, 512, 128), (4194304, 128, 8192, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
        # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3:4
        stream0 = get_raw_stream(0)
        triton_poi_fused__scaled_dot_product_efficient_attention__unsafe_view_add_mul_split_transpose_unsqueeze_view_3.run(buf17, arg4_1, 4194304, stream=stream0)
        del arg4_1
        # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split, aten._unsafe_view, aten.add, aten.view, aten.transpose, aten.mul, aten.unsqueeze, aten._scaled_dot_product_efficient_attention]
        buf18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf15, buf16, buf17, None, False, scale=1.0)
        del buf15
        del buf16
        buf19 = buf18[0]
        assert_size_stride(buf19, (1, 64, 512, 128), (4194304, 128, 8192, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        assert_alignment(buf19, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
        del buf18
        buf23 = reinterpret_tensor(buf17, (2048, 2048), (2048, 1), 0); del buf17  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:11
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf23)
        del arg5_1
        buf27 = reinterpret_tensor(buf19, (4, 512, 2048), (1048576, 2048, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [, multi_head_attention_forward, attn_out, x_3, layer_norm_3], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add, aten.native_layer_norm]
        # [Provenance debug handles] triton_red_fused_add_addmm_native_layer_norm_transpose_view_4:5
        stream0 = get_raw_stream(0)
        triton_red_fused_add_addmm_native_layer_norm_transpose_view_4.run(arg2_1, buf23, arg6_1, arg7_1, arg8_1, buf27, 2048, 2048, stream=stream0)
        del arg7_1
        del arg8_1
        buf28 = empty_strided_cuda((2048, 8192), (8192, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:12
        extern_kernels.mm(reinterpret_tensor(buf27, (2048, 2048), (2048, 1), 0), reinterpret_tensor(arg9_1, (2048, 8192), (1, 2048), 0), out=buf28)
        del arg9_1
        del buf27
        buf29 = reinterpret_tensor(buf28, (4, 512, 8192), (4194304, 8192, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [, input_1, input_2], Original ATen: [aten.addmm, aten.view, aten.gelu]
        # [Provenance debug handles] triton_poi_fused_addmm_gelu_view_5:6
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_gelu_view_5.run(buf29, arg10_1, 16777216, stream=stream0)
        del arg10_1
        buf30 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        # [Provenance debug handles] extern_kernels.mm:13
        extern_kernels.mm(reinterpret_tensor(buf29, (2048, 8192), (8192, 1), 0), reinterpret_tensor(arg11_1, (8192, 2048), (1, 8192), 0), out=buf30)
        del arg11_1
        del buf29
        buf31 = reinterpret_tensor(buf30, (4, 512, 2048), (1048576, 2048, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [, multi_head_attention_forward, attn_out, x_3, input_3, x_4], Original ATen: [aten.addmm, aten.view, aten.transpose, aten.add]
        # [Provenance debug handles] triton_poi_fused_add_addmm_transpose_view_6:7
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_transpose_view_6.run(buf31, arg2_1, buf23, arg6_1, arg12_1, 4194304, stream=stream0)
        del arg12_1
        del arg2_1
        del arg6_1
        del buf23
    return (buf31, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1 = args
        args.clear()
        partition0_args = [arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1]
        del arg2_1, arg0_1, arg1_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1
        (buf31,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf31, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 512, 2048), (1048576, 2048, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
