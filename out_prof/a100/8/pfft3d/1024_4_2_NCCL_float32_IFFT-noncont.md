# Reporting for IFFT-noncont
## Parameters
| Parameter   | Value        |
|-------------|--------------|
| Function    | IFFT-noncont |
| Precision   | float32      |
| X           | 1024         |
| Y           | 1024         |
| Z           | 1024         |
| PX          | 4            |
| PY          | 2            |
| Backend     | NCCL         |
| Nodes       | 1            |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 132.32425699970918 |
| Min Time       | 23.905672073364258 |
| Max Time       | 27.81362533569336  |
| Mean Time      | 25.490297317504883 |
| Std Time       | 1.629544734954834  |
| Last Time      | 24.002347946166992 |
| Generated Code | 5.23 KB            |
| Argument Size  | 1.00 GB            |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 27.6284 |
| Run 1       | 27.8136 |
| Run 2       | 27.202  |
| Run 3       | 26.8747 |
| Run 4       | 25.5888 |
| Run 5       | 23.9642 |
| Run 6       | 23.9057 |
| Run 7       | 23.9485 |
| Run 8       | 23.9746 |
| Run 9       | 24.0023 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c64[512,1024,256]{2,1,0})->c64[256,512,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="d066e3cc0f326591d9d4e8f4cf839060"}

%fused_multiply (param_0: c64[512,1024,256]) -> c64[512,1024,256] {
  %param_0 = c64[512,1024,256]{2,1,0} parameter(0)
  %constant_7_1 = c64[] constant((9.31322575e-10, 0)), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/div" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=44}
  %broadcast.9.1 = c64[512,1024,256]{2,1,0} broadcast(c64[] %constant_7_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %multiply.5.1 = c64[512,1024,256]{2,1,0} multiply(c64[512,1024,256]{2,1,0} %param_0, c64[512,1024,256]{2,1,0} %broadcast.9.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}

%fused_broadcast () -> s8[2147483648] {
  %constant_6_1 = s8[] constant(0), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14c92533e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14c92533e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.8.1 = s8[2147483648]{0} broadcast(s8[] %constant_6_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14c92533e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14c92533e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_copy_computation (param_0.2: c64[512,1024,256]) -> c64[512,1024,256] {
  %param_0.2 = c64[512,1024,256]{2,1,0} parameter(0)
  ROOT %copy.3 = c64[512,1024,256]{2,1,0} copy(c64[512,1024,256]{2,1,0} %param_0.2)
}

ENTRY %main.22_spmd (param.1: c64[512,1024,256]) -> c64[256,512,1024] {
  %param.1 = c64[512,1024,256]{2,1,0} parameter(0), sharding={devices=[1,2,4]<=[8]}, metadata={op_name="x"}
  %wrapped_copy = c64[512,1024,256]{2,1,0} fusion(c64[512,1024,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_copy_computation
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14c92533e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14c92533e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.3.0 = c64[512,1024,256]{2,1,0} custom-call(c64[512,1024,256]{2,1,0} %wrapped_copy, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[512,1024,256]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14c92533e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14c92533e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\000\000\000\004\000\000\000\010\000\000\000\002\000\000\002\000\000\000\000\000\000\000\000\004\000\000\000\010\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\002\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  %loop_multiply_fusion = c64[512,1024,256]{2,1,0} fusion(c64[512,1024,256]{2,1,0} %custom-call.3.0), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %bitcast.20 = c64[256,512,1024]{2,1,0} bitcast(c64[512,1024,256]{2,1,0} %loop_multiply_fusion), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,2,4]<=[8]}"}) -> (tensor<512x2048x1024xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<512x2048x1024xcomplex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    return %0 : tensor<512x2048x1024xcomplex<f32>>
  }
  func.func private @do_ifft(%arg0: tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<512x2048x1024xcomplex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    return %0 : tensor<512x2048x1024xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(5.120000e+02,0.000000e+00), (2.048000e+03,0.000000e+00), (1.024000e+03,0.000000e+00)]> : tensor<3xcomplex<f32>>
    %0 = call @pfft_impl(%arg0) : (tensor<512x2048x1024xcomplex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
    %cst_1 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %2 = stablehlo.divide %cst_1, %1 : tensor<complex<f32>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    %4 = stablehlo.multiply %0, %3 : tensor<512x2048x1024xcomplex<f32>>
    return %4 : tensor<512x2048x1024xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<512x2048x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @CustomSPMDPartitioning(%arg0) {api_version = 2 : i32, backend_config = "22848141689616"} : (tensor<512x2048x1024xcomplex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    return %0 : tensor<512x2048x1024xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c64[512,2048,1024]. let
    b:c64[512,2048,1024] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c64[512,2048,1024]. let
          d:c64[512,2048,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda e:c64[3]; f:c64[512,2048,1024]. let
                g:c64[512,2048,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14c7bf533490>
                  fun_jaxpr={ lambda ; h:c64[512,2048,1024]. let
                      i:c64[512,2048,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; j:c64[512,2048,1024]. let
                            k:c64[512,2048,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] h
                    in (i,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14c7bf5337f0>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14c7bf533250>
                  symbolic_zeros=False
                ] f
                l:c64[] = reduce_prod[axes=(0,)] e
                m:c64[] = div (1+0j) l
                n:c64[512,2048,1024] = mul g m
              in (n,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
