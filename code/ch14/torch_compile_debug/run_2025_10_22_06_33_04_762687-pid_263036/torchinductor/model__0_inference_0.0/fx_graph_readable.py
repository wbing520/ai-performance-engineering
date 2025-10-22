class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2048]", arg1_1: "f32[2048]", arg2_1: "f32[4, 512, 2048]", arg3_1: "f32[6144, 2048]", arg4_1: "f32[6144]", arg5_1: "f32[2048, 2048]", arg6_1: "f32[2048]", arg7_1: "f32[2048]", arg8_1: "f32[2048]", arg9_1: "f32[8192, 2048]", arg10_1: "f32[8192]", arg11_1: "f32[2048, 8192]", arg12_1: "f32[2048]"):
         # File: /home/ubuntu/ai-performance-engineering/code/ch14/inspect_compiled_code.py:46 in forward, code: attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        var_mean = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        add: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[4, 512, 2048]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  getitem_1 = None
        mul: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = None
        add_1: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[4, 512, 2048]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_3);  getitem_3 = None
        mul_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg0_1);  mul_2 = None
        add_3: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_3, arg1_1);  mul_3 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem_4: "f32[4, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_4: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[4, 512, 2048]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_5);  getitem_5 = None
        mul_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_5: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_4, arg0_1);  mul_4 = arg0_1 = None
        add_5: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_5, arg1_1);  mul_5 = arg1_1 = None
        permute: "f32[512, 4, 2048]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        permute_1: "f32[512, 4, 2048]" = torch.ops.aten.permute.default(add_3, [1, 0, 2]);  add_3 = None
        permute_2: "f32[512, 4, 2048]" = torch.ops.aten.permute.default(add_5, [1, 0, 2]);  add_5 = None
        split = torch.ops.aten.split.Tensor(arg3_1, 2048);  arg3_1 = None
        getitem_6: "f32[2048, 2048]" = split[0]
        getitem_7: "f32[2048, 2048]" = split[1]
        getitem_8: "f32[2048, 2048]" = split[2];  split = None
        split_1 = torch.ops.aten.split.Tensor(arg4_1, 2048);  arg4_1 = None
        getitem_9: "f32[2048]" = split_1[0]
        getitem_10: "f32[2048]" = split_1[1]
        getitem_11: "f32[2048]" = split_1[2];  split_1 = None
        permute_3: "f32[2048, 2048]" = torch.ops.aten.permute.default(getitem_6, [1, 0]);  getitem_6 = None
        clone: "f32[512, 4, 2048]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "f32[2048, 2048]" = torch.ops.aten.view.default(clone, [2048, 2048]);  clone = None
        mm: "f32[2048, 2048]" = torch.ops.aten.mm.default(view, permute_3);  view = permute_3 = None
        view_1: "f32[512, 4, 2048]" = torch.ops.aten.view.default(mm, [512, 4, 2048]);  mm = None
        add_6: "f32[512, 4, 2048]" = torch.ops.aten.add.Tensor(view_1, getitem_9);  view_1 = getitem_9 = None
        permute_4: "f32[2048, 2048]" = torch.ops.aten.permute.default(getitem_7, [1, 0]);  getitem_7 = None
        clone_1: "f32[512, 4, 2048]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[2048, 2048]" = torch.ops.aten.view.default(clone_1, [2048, 2048]);  clone_1 = None
        mm_1: "f32[2048, 2048]" = torch.ops.aten.mm.default(view_2, permute_4);  view_2 = permute_4 = None
        view_3: "f32[512, 4, 2048]" = torch.ops.aten.view.default(mm_1, [512, 4, 2048]);  mm_1 = None
        add_7: "f32[512, 4, 2048]" = torch.ops.aten.add.Tensor(view_3, getitem_10);  view_3 = getitem_10 = None
        permute_5: "f32[2048, 2048]" = torch.ops.aten.permute.default(getitem_8, [1, 0]);  getitem_8 = None
        clone_2: "f32[512, 4, 2048]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_4: "f32[2048, 2048]" = torch.ops.aten.view.default(clone_2, [2048, 2048]);  clone_2 = None
        mm_2: "f32[2048, 2048]" = torch.ops.aten.mm.default(view_4, permute_5);  view_4 = permute_5 = None
        view_5: "f32[512, 4, 2048]" = torch.ops.aten.view.default(mm_2, [512, 4, 2048]);  mm_2 = None
        add_8: "f32[512, 4, 2048]" = torch.ops.aten.add.Tensor(view_5, getitem_11);  view_5 = getitem_11 = None
        view_6: "f32[512, 64, 128]" = torch.ops.aten.view.default(add_6, [512, 64, 128]);  add_6 = None
        permute_6: "f32[64, 512, 128]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f32[512, 64, 128]" = torch.ops.aten.view.default(add_7, [512, 64, 128]);  add_7 = None
        permute_7: "f32[64, 512, 128]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f32[512, 64, 128]" = torch.ops.aten.view.default(add_8, [512, 64, 128]);  add_8 = None
        permute_8: "f32[64, 512, 128]" = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        mul_6: "f32[64, 512, 128]" = torch.ops.aten.mul.Tensor(permute_6, 0.08838834764831845);  permute_6 = None
        
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[1, 64, 512, 128]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_default_1: "f32[1, 64, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        unsqueeze_default_2: "f32[1, 64, 512, 128]" = torch.ops.aten.unsqueeze.default(permute_8, 0);  permute_8 = None
        _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, False, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
        getitem_14: "f32[1, 64, 512, 128]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
        squeeze_dim: "f32[64, 512, 128]" = torch.ops.aten.squeeze.dim(getitem_14, 0);  getitem_14 = None
        
         # File: /home/ubuntu/ai-performance-engineering/code/ch14/inspect_compiled_code.py:46 in forward, code: attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        permute_10: "f32[512, 64, 128]" = torch.ops.aten.permute.default(squeeze_dim, [1, 0, 2]);  squeeze_dim = None
        clone_3: "f32[512, 64, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_9: "f32[2048, 2048]" = torch.ops.aten.view.default(clone_3, [2048, 2048]);  clone_3 = None
        permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "f32[2048, 2048]" = torch.ops.aten.addmm.default(arg6_1, view_9, permute_11);  arg6_1 = view_9 = permute_11 = None
        view_10: "f32[512, 4, 2048]" = torch.ops.aten.view.default(addmm, [512, 4, 2048]);  addmm = None
        permute_12: "f32[4, 512, 2048]" = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        
         # File: /home/ubuntu/ai-performance-engineering/code/ch14/inspect_compiled_code.py:47 in forward, code: x = x + attn_out
        add_9: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(arg2_1, permute_12);  arg2_1 = permute_12 = None
        
         # File: /home/ubuntu/ai-performance-engineering/code/ch14/inspect_compiled_code.py:50 in forward, code: x = x + self.mlp(self.ln2(x))
        var_mean_3 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_12: "f32[4, 512, 1]" = var_mean_3[0]
        getitem_13: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[4, 512, 2048]" = torch.ops.aten.sub.Tensor(add_9, getitem_13);  getitem_13 = None
        mul_7: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_8: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_7, arg7_1);  mul_7 = arg7_1 = None
        add_11: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_8, arg8_1);  mul_8 = arg8_1 = None
        view_12: "f32[2048, 2048]" = torch.ops.aten.view.default(add_11, [2048, 2048]);  add_11 = None
        permute_13: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f32[2048, 8192]" = torch.ops.aten.addmm.default(arg10_1, view_12, permute_13);  arg10_1 = view_12 = permute_13 = None
        view_13: "f32[4, 512, 8192]" = torch.ops.aten.view.default(addmm_1, [4, 512, 8192]);  addmm_1 = None
        mul_9: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_10: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[4, 512, 8192]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_12: "f32[4, 512, 8192]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_11: "f32[4, 512, 8192]" = torch.ops.aten.mul.Tensor(mul_9, add_12);  mul_9 = add_12 = None
        view_14: "f32[2048, 8192]" = torch.ops.aten.view.default(mul_11, [2048, 8192]);  mul_11 = None
        permute_14: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[2048, 2048]" = torch.ops.aten.addmm.default(arg12_1, view_14, permute_14);  arg12_1 = view_14 = permute_14 = None
        view_15: "f32[4, 512, 2048]" = torch.ops.aten.view.default(addmm_2, [4, 512, 2048]);  addmm_2 = None
        add_13: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(add_9, view_15);  add_9 = view_15 = None
        return (add_13,)
        