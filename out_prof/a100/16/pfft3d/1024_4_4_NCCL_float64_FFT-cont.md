# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float64  |
| X           | 1024     |
| Y           | 1024     |
| Z           | 1024     |
| PX          | 4        |
| PY          | 4        |
| Backend     | NCCL     |
| Nodes       | 2        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 7509.670485043898  |
| Min Time       | 340.14170848968206 |
| Max Time       | 402.76749389158795 |
| Mean Time      | 347.8929518634686  |
| Std Time       | 18.317616892924345 |
| Last Time      | 343.0491752369562  |
| Generated Code | 5.48 KB            |
| Argument Size  | 512.00 MB          |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 402.767 |
| Run 1       | 342.214 |
| Run 2       | 340.142 |
| Run 3       | 342.284 |
| Run 4       | 341.993 |
| Run 5       | 343.11  |
| Run 6       | 340.75  |
| Run 7       | 340.443 |
| Run 8       | 342.178 |
| Run 9       | 343.049 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[256,256,1024]{2,1,0})->c128[256,256,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="a2d7006ea00546585788788def7007f7"}

%fused_broadcast () -> s8[2147483648] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x147675ef6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x147675ef60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[2147483648]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x147675ef6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x147675ef60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[256,256,1024]) -> c128[256,256,1024] {
  %param_0.1 = f64[256,256,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[256,256,1024]{2,1,0} convert(f64[256,256,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[256,256,1024]) -> c128[256,256,1024] {
  %param.1 = f64[256,256,1024]{2,1,0} parameter(0), sharding={devices=[4,4,1]<=[16]}, metadata={op_name="x"}
  %wrapped_convert = c128[256,256,1024]{2,1,0} fusion(f64[256,256,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x147675ef6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x147675ef60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %custom-call.4.0 = c128[256,256,1024]{2,1,0} custom-call(c128[256,256,1024]{2,1,0} %wrapped_convert, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[256,256,1024]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x147675ef6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x147675ef60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\004\000\000\000\004\000\000\000\004\000\000\002\000\000\000\001\000\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,4,1]<=[16]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x1024x1024xcomplex<f64>>
    return %2 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22493013635040"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %1 : tensor<1024x1024x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[1024,1024,1024]. let
    b:c128[1024,1024,1024] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[1024,1024,1024]. let
          d:c128[1024,1024,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[1024,1024,1024]. let
                f:c128[1024,1024,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x1475100f28c0>
                  fun_jaxpr={ lambda ; g:f64[1024,1024,1024]. let
                      h:c128[1024,1024,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[1024,1024,1024]. let
                            j:c128[1024,1024,1024] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[1024,1024,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x1475100f2c20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x1475100f2710>
                  symbolic_zeros=False
                ] e
                l:c128[1024,1024,1024] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
