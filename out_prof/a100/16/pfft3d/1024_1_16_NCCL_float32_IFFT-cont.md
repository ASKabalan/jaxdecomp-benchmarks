# Reporting for IFFT-cont
## Parameters
| Parameter   | Value     |
|-------------|-----------|
| Function    | IFFT-cont |
| Precision   | float32   |
| X           | 1024      |
| Y           | 1024      |
| Z           | 1024      |
| PX          | 1         |
| PY          | 16        |
| Backend     | NCCL      |
| Nodes       | 2         |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 266.1640290170908  |
| Min Time       | 185.68072509765625 |
| Max Time       | 187.66952514648438 |
| Mean Time      | 186.70765686035156 |
| Std Time       | 0.5794076323509216 |
| Last Time      | 187.39828491210938 |
| Generated Code | 5.10 KB            |
| Argument Size  | 512.00 MB          |
| Output Size    | 512.00 MB          |
| Temporary Size | 1.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 186.837 |
| Run 1       | 187.67  |
| Run 2       | 186.463 |
| Run 3       | 186.913 |
| Run 4       | 186.945 |
| Run 5       | 185.681 |
| Run 6       | 185.949 |
| Run 7       | 186.866 |
| Run 8       | 186.355 |
| Run 9       | 187.398 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c64[64,64,16384]{2,1,0})->c64[1024,64,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="0d999537d06ab32bf140f6287d7be485"}

%fused_multiply (param_0: c64[64,64,16384]) -> c64[64,64,16384] {
  %param_0 = c64[64,64,16384]{2,1,0} parameter(0)
  %constant_7_1 = c64[] constant((9.31322575e-10, 0)), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/div" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=44}
  %broadcast.9.1 = c64[64,64,16384]{2,1,0} broadcast(c64[] %constant_7_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %multiply.5.1 = c64[64,64,16384]{2,1,0} multiply(c64[64,64,16384]{2,1,0} %param_0, c64[64,64,16384]{2,1,0} %broadcast.9.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}

%fused_broadcast () -> s8[1073741824] {
  %constant_6_1 = s8[] constant(0), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14a8d296e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14a8d296e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.8.1 = s8[1073741824]{0} broadcast(s8[] %constant_6_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14a8d296e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14a8d296e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_copy_computation (param_0.2: c64[64,64,16384]) -> c64[64,64,16384] {
  %param_0.2 = c64[64,64,16384]{2,1,0} parameter(0)
  ROOT %copy.3 = c64[64,64,16384]{2,1,0} copy(c64[64,64,16384]{2,1,0} %param_0.2)
}

ENTRY %main.22_spmd (param.1: c64[64,64,16384]) -> c64[1024,64,1024] {
  %param.1 = c64[64,64,16384]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="x"}
  %wrapped_copy = c64[64,64,16384]{2,1,0} fusion(c64[64,64,16384]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_copy_computation
  %loop_broadcast_fusion = s8[1073741824]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14a8d296e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14a8d296e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.3.0 = c64[64,64,16384]{2,1,0} custom-call(c64[64,64,16384]{2,1,0} %wrapped_copy, s8[1073741824]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[64,64,16384]{2,1,0}, s8[1073741824]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14a8d296e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14a8d296e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\000\000\000@\000\000\000\004\000\000@\000\000\000\000\000\000\000\000\000\000\000\000@\000\000\000\004\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  %loop_multiply_fusion = c64[64,64,16384]{2,1,0} fusion(c64[64,64,16384]{2,1,0} %custom-call.3.0), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %bitcast.20 = c64[1024,64,1024]{2,1,0} bitcast(c64[64,64,16384]{2,1,0} %loop_multiply_fusion), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1024x16384xcomplex<f32>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<16384x64x1024xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<64x1024x16384xcomplex<f32>>) -> tensor<16384x64x1024xcomplex<f32>>
    return %0 : tensor<16384x64x1024xcomplex<f32>>
  }
  func.func private @do_ifft(%arg0: tensor<64x1024x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<64x1024x16384xcomplex<f32>>) -> tensor<16384x64x1024xcomplex<f32>>
    return %0 : tensor<16384x64x1024xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<64x1024x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(6.400000e+01,0.000000e+00), (1.024000e+03,0.000000e+00), (1.638400e+04,0.000000e+00)]> : tensor<3xcomplex<f32>>
    %0 = call @pfft_impl(%arg0) : (tensor<64x1024x16384xcomplex<f32>>) -> tensor<16384x64x1024xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
    %cst_1 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %2 = stablehlo.divide %cst_1, %1 : tensor<complex<f32>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f32>>) -> tensor<16384x64x1024xcomplex<f32>>
    %4 = stablehlo.multiply %0, %3 : tensor<16384x64x1024xcomplex<f32>>
    return %4 : tensor<16384x64x1024xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<64x1024x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @CustomSPMDPartitioning(%arg0) {api_version = 2 : i32, backend_config = "22709384036992"} : (tensor<64x1024x16384xcomplex<f32>>) -> tensor<16384x64x1024xcomplex<f32>>
    return %0 : tensor<16384x64x1024xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c64[64,1024,16384]. let
    b:c64[16384,64,1024] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c64[64,1024,16384]. let
          d:c64[16384,64,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda e:c64[3]; f:c64[64,1024,16384]. let
                g:c64[16384,64,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14a7719cf490>
                  fun_jaxpr={ lambda ; h:c64[64,1024,16384]. let
                      i:c64[16384,64,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; j:c64[64,1024,16384]. let
                            k:c64[16384,64,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] h
                    in (i,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14a7719cf640>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14a7719cf250>
                  symbolic_zeros=False
                ] f
                l:c64[] = reduce_prod[axes=(0,)] e
                m:c64[] = div (1+0j) l
                n:c64[16384,64,1024] = mul g m
              in (n,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
