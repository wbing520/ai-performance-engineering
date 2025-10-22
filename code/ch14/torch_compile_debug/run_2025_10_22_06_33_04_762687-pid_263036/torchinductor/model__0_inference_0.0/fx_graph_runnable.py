
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/home/ubuntu/ai-performance-engineering/code/ch14/compiled_code_output'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.max_autotune = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.0+cu130
# torch cuda version: 13.0
# torch git version: 0fabc3ba44823f257e70ce397d989c8de5e362c1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Fri_Feb_21_20:23:50_PST_2025 
# Cuda compilation tools, release 12.8, V12.8.93 
# Build cuda_12.8.r12.8/compiler.35583870_0 

# GPU Hardware Info: 
# NVIDIA B200 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1):
        var_mean = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(arg2_1, getitem_3);  getitem_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg0_1);  mul_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, arg1_1);  mul_3 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(arg2_1, getitem_5);  getitem_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, arg0_1);  mul_4 = arg0_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_5, arg1_1);  mul_5 = arg1_1 = None
        permute = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        permute_1 = torch.ops.aten.permute.default(add_3, [1, 0, 2]);  add_3 = None
        permute_2 = torch.ops.aten.permute.default(add_5, [1, 0, 2]);  add_5 = None
        split = torch.ops.aten.split.Tensor(arg3_1, 2048);  arg3_1 = None
        getitem_6 = split[0]
        getitem_7 = split[1]
        getitem_8 = split[2];  split = None
        split_1 = torch.ops.aten.split.Tensor(arg4_1, 2048);  arg4_1 = None
        getitem_9 = split_1[0]
        getitem_10 = split_1[1]
        getitem_11 = split_1[2];  split_1 = None
        permute_3 = torch.ops.aten.permute.default(getitem_6, [1, 0]);  getitem_6 = None
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view = torch.ops.aten.view.default(clone, [2048, 2048]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute_3);  view = permute_3 = None
        view_1 = torch.ops.aten.view.default(mm, [512, 4, 2048]);  mm = None
        add_6 = torch.ops.aten.add.Tensor(view_1, getitem_9);  view_1 = getitem_9 = None
        permute_4 = torch.ops.aten.permute.default(getitem_7, [1, 0]);  getitem_7 = None
        clone_1 = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2 = torch.ops.aten.view.default(clone_1, [2048, 2048]);  clone_1 = None
        mm_1 = torch.ops.aten.mm.default(view_2, permute_4);  view_2 = permute_4 = None
        view_3 = torch.ops.aten.view.default(mm_1, [512, 4, 2048]);  mm_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_3, getitem_10);  view_3 = getitem_10 = None
        permute_5 = torch.ops.aten.permute.default(getitem_8, [1, 0]);  getitem_8 = None
        clone_2 = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_4 = torch.ops.aten.view.default(clone_2, [2048, 2048]);  clone_2 = None
        mm_2 = torch.ops.aten.mm.default(view_4, permute_5);  view_4 = permute_5 = None
        view_5 = torch.ops.aten.view.default(mm_2, [512, 4, 2048]);  mm_2 = None
        add_8 = torch.ops.aten.add.Tensor(view_5, getitem_11);  view_5 = getitem_11 = None
        view_6 = torch.ops.aten.view.default(add_6, [512, 64, 128]);  add_6 = None
        permute_6 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(add_7, [512, 64, 128]);  add_7 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8 = torch.ops.aten.view.default(add_8, [512, 64, 128]);  add_8 = None
        permute_8 = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        mul_6 = torch.ops.aten.mul.Tensor(permute_6, 0.08838834764831845);  permute_6 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(permute_8, 0);  permute_8 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, False, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem_14 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem_14, 0);  getitem_14 = None
        permute_10 = torch.ops.aten.permute.default(squeeze_dim, [1, 0, 2]);  squeeze_dim = None
        clone_3 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_9 = torch.ops.aten.view.default(clone_3, [2048, 2048]);  clone_3 = None
        permute_11 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_9, permute_11);  arg6_1 = view_9 = permute_11 = None
        view_10 = torch.ops.aten.view.default(addmm, [512, 4, 2048]);  addmm = None
        permute_12 = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        add_9 = torch.ops.aten.add.Tensor(arg2_1, permute_12);  arg2_1 = permute_12 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_3[0]
        getitem_13 = var_mean_3[1];  var_mean_3 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_13);  getitem_13 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg7_1);  mul_7 = arg7_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_8, arg8_1);  mul_8 = arg8_1 = None
        view_12 = torch.ops.aten.view.default(add_11, [2048, 2048]);  add_11 = None
        permute_13 = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg10_1, view_12, permute_13);  arg10_1 = view_12 = permute_13 = None
        view_13 = torch.ops.aten.view.default(addmm_1, [4, 512, 8192]);  addmm_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_10 = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_12 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_9, add_12);  mul_9 = add_12 = None
        view_14 = torch.ops.aten.view.default(mul_11, [2048, 8192]);  mul_11 = None
        permute_14 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg12_1, view_14, permute_14);  arg12_1 = view_14 = permute_14 = None
        view_15 = torch.ops.aten.view.default(addmm_2, [4, 512, 2048]);  addmm_2 = None
        add_13 = torch.ops.aten.add.Tensor(add_9, view_15);  add_9 = view_15 = None
        return (add_13,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048,), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2048,), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf2, (4, 512, 2048), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf3, (6144, 2048), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf4, (6144,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2048, 2048), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2048,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2048,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2048,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf9, (8192, 2048), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf10, (8192,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 67108864, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2048, 8192), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf12, (2048,), is_leaf=True)  # arg12_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)