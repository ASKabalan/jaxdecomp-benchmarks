# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float32  |
| X           | 1024     |
| Y           | 1024     |
| Z           | 1024     |
| PX          | 1        |
| PY          | 8        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 9717.514272000699  |
| Min Time       | 13.837576866149902 |
| Max Time       | 80.75959777832031  |
| Mean Time      | 21.226072311401367 |
| Std Time       | 19.84757423400879  |
| Last Time      | 13.837576866149902 |
| Generated Code | 5.35 KB            |
| Argument Size  | 512.00 MB          |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 80.7596 |
| Run 1       | 14.7004 |
| Run 2       | 14.7681 |
| Run 3       | 14.7376 |
| Run 4       | 14.7283 |
| Run 5       | 14.9298 |
| Run 6       | 14.8983 |
| Run 7       | 14.8511 |
| Run 8       | 14.05   |
| Run 9       | 13.8376 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[1024,128,1024]{2,1,0})->c64[128,128,8192]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="9254dae378dc1c10ab9e5f6ad25653d3"}

%fused_broadcast () -> s8[2147483648] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x149180eb6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x149180eb60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[2147483648]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x149180eb6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x149180eb60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[1024,128,1024]) -> c64[1024,128,1024] {
  %param_0.1 = f32[1024,128,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[1024,128,1024]{2,1,0} convert(f32[1024,128,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[1024,128,1024]) -> c64[128,128,8192] {
  %param.1 = f32[1024,128,1024]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c64[1024,128,1024]{2,1,0} fusion(f32[1024,128,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x149180eb6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x149180eb60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[1024,128,1024]{2,1,0} custom-call(c64[1024,128,1024]{2,1,0} %wrapped_convert, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[1024,128,1024]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x149180eb6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x149180eb60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000 \000\000\000\004\000\000\200\000\000\000\000\000\000\000\000\000\000\000\000 \000\000\000\004\000\000\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[128,128,8192]{2,1,0} bitcast(c64[1024,128,1024]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x149180eb6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x149180eb60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x128x1024xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}) -> (tensor<128x1024x8192xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<8192x128x1024xf32>) -> tensor<128x1024x8192xcomplex<f32>>
    return %0 : tensor<128x1024x8192xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<8192x128x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<128x1024x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<8192x128x1024xf32>) -> tensor<128x1024x8192xcomplex<f32>>
    return %0 : tensor<128x1024x8192xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<8192x128x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<128x1024x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<8192x128x1024xf32>) -> tensor<128x1024x8192xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<128x1024x8192xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<128x1024x8192xcomplex<f32>>
    return %2 : tensor<128x1024x8192xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<8192x128x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<128x1024x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<8192x128x1024xf32>) -> tensor<8192x128x1024xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22609162061808"} : (tensor<8192x128x1024xcomplex<f32>>) -> tensor<128x1024x8192xcomplex<f32>>
    return %1 : tensor<128x1024x8192xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[8192,128,1024]. let
    b:c64[128,1024,8192] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[8192,128,1024]. let
          d:c64[128,1024,8192] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[8192,128,1024]. let
                f:c64[128,1024,8192] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14901b0aa7a0>
                  fun_jaxpr={ lambda ; g:f32[8192,128,1024]. let
                      h:c64[128,1024,8192] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[8192,128,1024]. let
                            j:c64[8192,128,1024] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[128,1024,8192] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14901b0aab00>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14901b0aa5f0>
                  symbolic_zeros=False
                ] e
                l:c64[128,1024,8192] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
