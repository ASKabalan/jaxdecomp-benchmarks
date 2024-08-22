# Reporting for FFT-noncont
## Parameters
| Parameter   | Value       |
|-------------|-------------|
| Function    | FFT-noncont |
| Precision   | float64     |
| X           | 1024        |
| Y           | 1024        |
| Z           | 1024        |
| PX          | 8           |
| PY          | 1           |
| Backend     | NCCL        |
| Nodes       | 1           |
---
## Profiling Data
| Parameter      | Value             |
|----------------|-------------------|
| JIT Time       | 9597.58706499997  |
| Min Time       | 36.30771062489657 |
| Max Time       | 108.5465441242377 |
| Mean Time      | 43.8569631248356  |
| Std Time       | 21.56960307355621 |
| Last Time      | 36.36525849969985 |
| Generated Code | 5.48 KB           |
| Argument Size  | 1.00 GB           |
| Output Size    | 2.00 GB           |
| Temporary Size | 4.00 GB           |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 108.547  |
| Run 1       |  37.9066 |
| Run 2       |  37.4551 |
| Run 3       |  36.5092 |
| Run 4       |  36.4274 |
| Run 5       |  36.3077 |
| Run 6       |  36.4013 |
| Run 7       |  36.3396 |
| Run 8       |  36.311  |
| Run 9       |  36.3653 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[128,1024,1024]{2,1,0})->c128[128,8192,128]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="e33fe24928572ab4e3f8d5530b4943e8"}

%fused_broadcast () -> s8[4294967296] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[4294967296]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[128,1024,1024]) -> c128[128,1024,1024] {
  %param_0.1 = f64[128,1024,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[128,1024,1024]{2,1,0} convert(f64[128,1024,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[128,1024,1024]) -> c128[128,8192,128] {
  %param.1 = f64[128,1024,1024]{2,1,0} parameter(0), sharding={devices=[1,8,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c128[128,1024,1024]{2,1,0} fusion(f64[128,1024,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[4294967296]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[128,1024,1024]{2,1,0} custom-call(c128[128,1024,1024]{2,1,0} %wrapped_convert, s8[4294967296]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[128,1024,1024]{2,1,0}, s8[4294967296]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\001\000\000\004\000\000\000 \000\000\200\000\000\000\001\000\000\000\001\000\000\000\000\004\000\000\000 \000\000\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  ROOT %bitcast.16 = c128[128,8192,128]{2,1,0} bitcast(c128[128,1024,1024]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x8192x1024xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,8,1]<=[8]}"}) -> (tensor<128x8192x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<128x8192x1024xf64>) -> tensor<128x8192x1024xcomplex<f64>>
    return %0 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<128x8192x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<128x8192x1024xf64>) -> tensor<128x8192x1024xcomplex<f64>>
    return %0 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<128x8192x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<128x8192x1024xf64>) -> tensor<128x8192x1024xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<128x8192x1024xcomplex<f64>>
    return %2 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<128x8192x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<128x8192x1024xf64>) -> tensor<128x8192x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22387382891376"} : (tensor<128x8192x1024xcomplex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    return %1 : tensor<128x8192x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[128,8192,1024]. let
    b:c128[128,8192,1024] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[128,8192,1024]. let
          d:c128[128,8192,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[128,8192,1024]. let
                f:c128[128,8192,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x145c77f9e8c0>
                  fun_jaxpr={ lambda ; g:f64[128,8192,1024]. let
                      h:c128[128,8192,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[128,8192,1024]. let
                            j:c128[128,8192,1024] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[128,8192,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x145c77f9ec20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x145c77f9e710>
                  symbolic_zeros=False
                ] e
                l:c128[128,8192,1024] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
