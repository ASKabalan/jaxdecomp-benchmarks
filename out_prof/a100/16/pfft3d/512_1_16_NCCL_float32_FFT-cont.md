# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float32  |
| X           | 512      |
| Y           | 512      |
| Z           | 512      |
| PX          | 1        |
| PY          | 16       |
| Backend     | NCCL     |
| Nodes       | 2        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 6354.13932800293   |
| Min Time       | 24.015687942504883 |
| Max Time       | 97.34410858154297  |
| Mean Time      | 31.726665496826172 |
| Std Time       | 21.873605728149414 |
| Last Time      | 24.015687942504883 |
| Generated Code | 6.23 KB            |
| Argument Size  | 32.00 MB           |
| Output Size    | 64.00 MB           |
| Temporary Size | 128.00 MB          |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 97.3441 |
| Run 1       | 24.6434 |
| Run 2       | 24.5884 |
| Run 3       | 24.2958 |
| Run 4       | 24.4516 |
| Run 5       | 24.7786 |
| Run 6       | 24.5874 |
| Run 7       | 24.1395 |
| Run 8       | 24.4221 |
| Run 9       | 24.0157 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[512,32,512]{2,1,0})->c64[32,32,8192]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="356ced164d1e6848af3b85e908e61e80"}

%fused_broadcast () -> s8[134217728] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x153e28e56170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x153e28e560e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[134217728]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x153e28e56170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x153e28e560e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[512,32,512]) -> c64[512,32,512] {
  %param_0.1 = f32[512,32,512]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[512,32,512]{2,1,0} convert(f32[512,32,512]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[512,32,512]) -> c64[32,32,8192] {
  %param.1 = f32[512,32,512]{2,1,0} parameter(0), sharding={devices=[16,1,1]<=[16]}, metadata={op_name="x"}
  %wrapped_convert = c64[512,32,512]{2,1,0} fusion(f32[512,32,512]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[134217728]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x153e28e56170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x153e28e560e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[512,32,512]{2,1,0} custom-call(c64[512,32,512]{2,1,0} %wrapped_convert, s8[134217728]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[512,32,512]{2,1,0}, s8[134217728]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x153e28e56170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x153e28e560e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000 \000\000\000\002\000\000 \000\000\000\000\000\000\000\000\000\000\000\000 \000\000\000\002\000\000 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[32,32,8192]{2,1,0} bitcast(c64[512,32,512]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x153e28e56170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x153e28e560e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x32x512xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[16,1,1]<=[16]}"}) -> (tensor<32x512x8192xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<8192x32x512xf32>) -> tensor<32x512x8192xcomplex<f32>>
    return %0 : tensor<32x512x8192xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<8192x32x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x512x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<8192x32x512xf32>) -> tensor<32x512x8192xcomplex<f32>>
    return %0 : tensor<32x512x8192xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<8192x32x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x512x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<8192x32x512xf32>) -> tensor<32x512x8192xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<32x512x8192xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<32x512x8192xcomplex<f32>>
    return %2 : tensor<32x512x8192xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<8192x32x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x512x8192xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<8192x32x512xf32>) -> tensor<8192x32x512xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23350712404144"} : (tensor<8192x32x512xcomplex<f32>>) -> tensor<32x512x8192xcomplex<f32>>
    return %1 : tensor<32x512x8192xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[8192,32,512]. let
    b:c64[32,512,8192] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[8192,32,512]. let
          d:c64[32,512,8192] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[8192,32,512]. let
                f:c64[32,512,8192] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x153cc2e2e7a0>
                  fun_jaxpr={ lambda ; g:f32[8192,32,512]. let
                      h:c64[32,512,8192] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[8192,32,512]. let
                            j:c64[8192,32,512] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[32,512,8192] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x153cc2e2eb00>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x153cc2e2e5f0>
                  symbolic_zeros=False
                ] e
                l:c64[32,512,8192] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
