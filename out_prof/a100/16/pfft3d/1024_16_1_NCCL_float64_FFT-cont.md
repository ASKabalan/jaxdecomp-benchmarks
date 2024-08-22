# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float64  |
| X           | 1024     |
| Y           | 1024     |
| Z           | 1024     |
| PX          | 16       |
| PY          | 1        |
| Backend     | NCCL     |
| Nodes       | 2        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 6621.472556958906  |
| Min Time       | 369.9401380654308  |
| Max Time       | 441.8523374333745  |
| Mean Time      | 378.397687683173   |
| Std Time       | 21.171773852623847 |
| Last Time      | 370.71623356314376 |
| Generated Code | 5.48 KB            |
| Argument Size  | 512.00 MB          |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 441.852 |
| Run 1       | 371.867 |
| Run 2       | 370.587 |
| Run 3       | 370.946 |
| Run 4       | 373.191 |
| Run 5       | 372.614 |
| Run 6       | 370.935 |
| Run 7       | 369.94  |
| Run 8       | 371.329 |
| Run 9       | 370.716 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[64,1024,1024]{2,1,0})->c128[64,64,16384]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="4443adc16882c059a8203febd5b13f96"}

%fused_broadcast () -> s8[2147483648] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fadc0f2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fadc0f20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[2147483648]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fadc0f2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fadc0f20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[64,1024,1024]) -> c128[64,1024,1024] {
  %param_0.1 = f64[64,1024,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[64,1024,1024]{2,1,0} convert(f64[64,1024,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[64,1024,1024]) -> c128[64,64,16384] {
  %param.1 = f64[64,1024,1024]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="x"}
  %wrapped_convert = c128[64,1024,1024]{2,1,0} fusion(f64[64,1024,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fadc0f2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fadc0f20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[64,1024,1024]{2,1,0} custom-call(c128[64,1024,1024]{2,1,0} %wrapped_convert, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[64,1024,1024]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fadc0f2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fadc0f20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\004\000\000\000@\000\000@\000\000\000\001\000\000\000\001\000\000\000\000\004\000\000\000@\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c128[64,64,16384]{2,1,0} bitcast(c128[64,1024,1024]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fadc0f2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fadc0f20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x16384x1024xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<1024x64x16384xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<64x16384x1024xf64>) -> tensor<1024x64x16384xcomplex<f64>>
    return %0 : tensor<1024x64x16384xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<64x16384x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<64x16384x1024xf64>) -> tensor<1024x64x16384xcomplex<f64>>
    return %0 : tensor<1024x64x16384xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<64x16384x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<64x16384x1024xf64>) -> tensor<1024x64x16384xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<1024x64x16384xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x64x16384xcomplex<f64>>
    return %2 : tensor<1024x64x16384xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<64x16384x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<64x16384x1024xf64>) -> tensor<64x16384x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23061652598800"} : (tensor<64x16384x1024xcomplex<f64>>) -> tensor<1024x64x16384xcomplex<f64>>
    return %1 : tensor<1024x64x16384xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[64,16384,1024]. let
    b:c128[1024,64,16384] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[64,16384,1024]. let
          d:c128[1024,64,16384] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[64,16384,1024]. let
                f:c128[1024,64,16384] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14f9759528c0>
                  fun_jaxpr={ lambda ; g:f64[64,16384,1024]. let
                      h:c128[1024,64,16384] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[64,16384,1024]. let
                            j:c128[64,16384,1024] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[1024,64,16384] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14f975952c20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14f975952710>
                  symbolic_zeros=False
                ] e
                l:c128[1024,64,16384] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
