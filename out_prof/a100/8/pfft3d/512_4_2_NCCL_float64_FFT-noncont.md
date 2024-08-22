# Reporting for FFT-noncont
## Parameters
| Parameter   | Value       |
|-------------|-------------|
| Function    | FFT-noncont |
| Precision   | float64     |
| X           | 512         |
| Y           | 512         |
| Z           | 512         |
| PX          | 4           |
| PY          | 2           |
| Backend     | NCCL        |
| Nodes       | 1           |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 8005.070153001725  |
| Min Time       | 8.668621500419249  |
| Max Time       | 81.8743386253118   |
| Mean Time      | 16.065237012389844 |
| Std Time       | 21.936498581763633 |
| Last Time      | 8.827279624711082  |
| Generated Code | 5.48 KB            |
| Argument Size  | 128.00 MB          |
| Output Size    | 256.00 MB          |
| Temporary Size | 512.00 MB          |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 81.8743  |
| Run 1       |  8.67246 |
| Run 2       |  8.66862 |
| Run 3       |  8.69647 |
| Run 4       |  8.69922 |
| Run 5       |  8.68536 |
| Run 6       |  8.80948 |
| Run 7       |  8.89078 |
| Run 8       |  8.82836 |
| Run 9       |  8.82728 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[128,256,512]{2,1,0})->c128[256,512,128]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="675b1db6ad7a58f0a24f2e4caf3f33fb"}

%fused_broadcast () -> s8[536870912] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14642cc6e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14642cc6e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[536870912]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14642cc6e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14642cc6e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[128,256,512]) -> c128[128,256,512] {
  %param_0.1 = f64[128,256,512]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[128,256,512]{2,1,0} convert(f64[128,256,512]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[128,256,512]) -> c128[256,512,128] {
  %param.1 = f64[128,256,512]{2,1,0} parameter(0), sharding={devices=[2,4,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c128[128,256,512]{2,1,0} fusion(f64[128,256,512]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[536870912]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14642cc6e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14642cc6e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[128,256,512]{2,1,0} custom-call(c128[128,256,512]{2,1,0} %wrapped_convert, s8[536870912]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[128,256,512]{2,1,0}, s8[536870912]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14642cc6e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14642cc6e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\001\000\000\002\000\000\000\004\000\000\000\001\000\000\002\000\000\000\001\000\000\000\000\002\000\000\000\004\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\002\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  ROOT %bitcast.16 = c128[256,512,128]{2,1,0} bitcast(c128[128,256,512]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14642cc6e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14642cc6e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x1024x512xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[2,4,1]<=[8]}"}) -> (tensor<256x1024x512xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<256x1024x512xf64>) -> tensor<256x1024x512xcomplex<f64>>
    return %0 : tensor<256x1024x512xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<256x1024x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<256x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<256x1024x512xf64>) -> tensor<256x1024x512xcomplex<f64>>
    return %0 : tensor<256x1024x512xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<256x1024x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<256x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<256x1024x512xf64>) -> tensor<256x1024x512xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<256x1024x512xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<256x1024x512xcomplex<f64>>
    return %2 : tensor<256x1024x512xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<256x1024x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<256x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<256x1024x512xf64>) -> tensor<256x1024x512xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22414476833504"} : (tensor<256x1024x512xcomplex<f64>>) -> tensor<256x1024x512xcomplex<f64>>
    return %1 : tensor<256x1024x512xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[256,1024,512]. let
    b:c128[256,1024,512] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[256,1024,512]. let
          d:c128[256,1024,512] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[256,1024,512]. let
                f:c128[256,1024,512] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x1462c6e6a8c0>
                  fun_jaxpr={ lambda ; g:f64[256,1024,512]. let
                      h:c128[256,1024,512] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[256,1024,512]. let
                            j:c128[256,1024,512] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[256,1024,512] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x1462c6e6ac20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x1462c6e6a710>
                  symbolic_zeros=False
                ] e
                l:c128[256,1024,512] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
