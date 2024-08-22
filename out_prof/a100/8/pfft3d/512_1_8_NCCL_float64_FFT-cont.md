# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float64  |
| X           | 512      |
| Y           | 512      |
| Z           | 512      |
| PX          | 1        |
| PY          | 8        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 9788.244529001531  |
| Min Time       | 3.6189803754496097 |
| Max Time       | 74.26017024999965  |
| Mean Time      | 10.73903958754272  |
| Std Time       | 21.173734738151968 |
| Last Time      | 3.6688430000140215 |
| Generated Code | 5.48 KB            |
| Argument Size  | 128.00 MB          |
| Output Size    | 256.00 MB          |
| Temporary Size | 512.00 MB          |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 74.2602  |
| Run 1       |  3.61898 |
| Run 2       |  3.72225 |
| Run 3       |  3.68986 |
| Run 4       |  3.66156 |
| Run 5       |  3.73004 |
| Run 6       |  3.68764 |
| Run 7       |  3.64619 |
| Run 8       |  3.70486 |
| Run 9       |  3.66884 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[512,64,512]{2,1,0})->c128[64,64,4096]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="179e68ee383ac6dfe997761e7000f657"}

%fused_broadcast () -> s8[536870912] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15313317e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15313317e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[536870912]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15313317e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15313317e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[512,64,512]) -> c128[512,64,512] {
  %param_0.1 = f64[512,64,512]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[512,64,512]{2,1,0} convert(f64[512,64,512]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[512,64,512]) -> c128[64,64,4096] {
  %param.1 = f64[512,64,512]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c128[512,64,512]{2,1,0} fusion(f64[512,64,512]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[536870912]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15313317e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15313317e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[512,64,512]{2,1,0} custom-call(c128[512,64,512]{2,1,0} %wrapped_convert, s8[536870912]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[512,64,512]{2,1,0}, s8[536870912]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15313317e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15313317e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\020\000\000\000\002\000\000@\000\000\000\000\000\000\000\001\000\000\000\000\020\000\000\000\002\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c128[64,64,4096]{2,1,0} bitcast(c128[512,64,512]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15313317e170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15313317e0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x64x512xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}) -> (tensor<64x512x4096xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<4096x64x512xf64>) -> tensor<64x512x4096xcomplex<f64>>
    return %0 : tensor<64x512x4096xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<4096x64x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<64x512x4096xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<4096x64x512xf64>) -> tensor<64x512x4096xcomplex<f64>>
    return %0 : tensor<64x512x4096xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<4096x64x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<64x512x4096xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<4096x64x512xf64>) -> tensor<64x512x4096xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<64x512x4096xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<64x512x4096xcomplex<f64>>
    return %2 : tensor<64x512x4096xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<4096x64x512xf64> {mhlo.layout_mode = "default"}) -> (tensor<64x512x4096xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<4096x64x512xf64>) -> tensor<4096x64x512xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23295118291872"} : (tensor<4096x64x512xcomplex<f64>>) -> tensor<64x512x4096xcomplex<f64>>
    return %1 : tensor<64x512x4096xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[4096,64,512]. let
    b:c128[64,512,4096] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[4096,64,512]. let
          d:c128[64,512,4096] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[4096,64,512]. let
                f:c128[64,512,4096] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x152fd13928c0>
                  fun_jaxpr={ lambda ; g:f64[4096,64,512]. let
                      h:c128[64,512,4096] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[4096,64,512]. let
                            j:c128[4096,64,512] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[64,512,4096] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x152fd1392c20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x152fd1392710>
                  symbolic_zeros=False
                ] e
                l:c128[64,512,4096] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
