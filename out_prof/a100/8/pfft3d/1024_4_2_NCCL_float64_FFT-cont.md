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
| PY          | 2        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 8185.676755998429  |
| Min Time       | 41.14321387487507  |
| Max Time       | 110.30230487494919 |
| Mean Time      | 48.41424721239491  |
| Std Time       | 20.63254790529345  |
| Last Time      | 41.29379037522085  |
| Generated Code | 5.48 KB            |
| Argument Size  | 1.00 GB            |
| Output Size    | 2.00 GB            |
| Temporary Size | 4.00 GB            |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 110.302  |
| Run 1       |  42.2705 |
| Run 2       |  42.192  |
| Run 3       |  41.1432 |
| Run 4       |  41.3275 |
| Run 5       |  41.3191 |
| Run 6       |  41.5141 |
| Run 7       |  41.4229 |
| Run 8       |  41.3571 |
| Run 9       |  41.2938 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[256,512,1024]{2,1,0})->c128[1024,256,512]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="d9020f200de3eb49034e7d9be63049e6"}

%fused_broadcast () -> s8[4294967296] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fe7a476170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fe7a4760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[4294967296]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fe7a476170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fe7a4760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[256,512,1024]) -> c128[256,512,1024] {
  %param_0.1 = f64[256,512,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[256,512,1024]{2,1,0} convert(f64[256,512,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[256,512,1024]) -> c128[1024,256,512] {
  %param.1 = f64[256,512,1024]{2,1,0} parameter(0), sharding={devices=[2,4,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c128[256,512,1024]{2,1,0} fusion(f64[256,512,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[4294967296]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fe7a476170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fe7a4760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[256,512,1024]{2,1,0} custom-call(c128[256,512,1024]{2,1,0} %wrapped_convert, s8[4294967296]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[256,512,1024]{2,1,0}, s8[4294967296]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fe7a476170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fe7a4760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\004\000\000\000\010\000\000\000\002\000\000\002\000\000\000\001\000\000\000\000\004\000\000\000\010\000\000\000\002\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\002\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c128[1024,256,512]{2,1,0} bitcast(c128[256,512,1024]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14fe7a476170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14fe7a4760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x2048x1024xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[2,4,1]<=[8]}"}) -> (tensor<2048x1024x512xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<512x2048x1024xf64>) -> tensor<2048x1024x512xcomplex<f64>>
    return %0 : tensor<2048x1024x512xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<512x2048x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<2048x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<512x2048x1024xf64>) -> tensor<2048x1024x512xcomplex<f64>>
    return %0 : tensor<2048x1024x512xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<512x2048x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<2048x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<512x2048x1024xf64>) -> tensor<2048x1024x512xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<2048x1024x512xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<2048x1024x512xcomplex<f64>>
    return %2 : tensor<2048x1024x512xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<512x2048x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<2048x1024x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<512x2048x1024xf64>) -> tensor<512x2048x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23077202112384"} : (tensor<512x2048x1024xcomplex<f64>>) -> tensor<2048x1024x512xcomplex<f64>>
    return %1 : tensor<2048x1024x512xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[512,2048,1024]. let
    b:c128[2048,1024,512] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[512,2048,1024]. let
          d:c128[2048,1024,512] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[512,2048,1024]. let
                f:c128[2048,1024,512] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14fd1467e8c0>
                  fun_jaxpr={ lambda ; g:f64[512,2048,1024]. let
                      h:c128[2048,1024,512] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[512,2048,1024]. let
                            j:c128[512,2048,1024] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[2048,1024,512] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14fd1467ec20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14fd1467e710>
                  symbolic_zeros=False
                ] e
                l:c128[2048,1024,512] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
