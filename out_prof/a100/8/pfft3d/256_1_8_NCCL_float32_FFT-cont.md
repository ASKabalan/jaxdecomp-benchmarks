# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float32  |
| X           | 256      |
| Y           | 256      |
| Z           | 256      |
| PX          | 1        |
| PY          | 8        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value               |
|----------------|---------------------|
| JIT Time       | 9820.810455003084   |
| Min Time       | 0.38428786396980286 |
| Max Time       | 69.42445373535156   |
| Mean Time      | 7.316117763519287   |
| Std Time       | 20.702791213989258  |
| Last Time      | 0.38870859146118164 |
| Generated Code | 5.10 KB             |
| Argument Size  | 8.00 MB             |
| Output Size    | 16.00 MB            |
| Temporary Size | 32.00 MB            |
---
## Iteration Runs
| Iteration   |      Time |
|-------------|-----------|
| Run 0       | 69.4245   |
| Run 1       |  0.409472 |
| Run 2       |  0.463477 |
| Run 3       |  0.428267 |
| Run 4       |  0.432615 |
| Run 5       |  0.397385 |
| Run 6       |  0.432427 |
| Run 7       |  0.384288 |
| Run 8       |  0.400089 |
| Run 9       |  0.388709 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[256,32,256]{2,1,0})->c64[32,32,2048]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="2b2f4befcee4fd09292a9382db0ac94c"}

%fused_broadcast () -> s8[33554432] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x152ac4bba170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x152ac4bba0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[33554432]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x152ac4bba170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x152ac4bba0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[256,32,256]) -> c64[256,32,256] {
  %param_0.1 = f32[256,32,256]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[256,32,256]{2,1,0} convert(f32[256,32,256]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[256,32,256]) -> c64[32,32,2048] {
  %param.1 = f32[256,32,256]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c64[256,32,256]{2,1,0} fusion(f32[256,32,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[33554432]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x152ac4bba170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x152ac4bba0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[256,32,256]{2,1,0} custom-call(c64[256,32,256]{2,1,0} %wrapped_convert, s8[33554432]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[256,32,256]{2,1,0}, s8[33554432]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x152ac4bba170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x152ac4bba0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\010\000\000\000\001\000\000 \000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[32,32,2048]{2,1,0} bitcast(c64[256,32,256]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x152ac4bba170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x152ac4bba0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x32x256xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}) -> (tensor<32x256x2048xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<2048x32x256xf32>) -> tensor<32x256x2048xcomplex<f32>>
    return %0 : tensor<32x256x2048xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<2048x32x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x256x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<2048x32x256xf32>) -> tensor<32x256x2048xcomplex<f32>>
    return %0 : tensor<32x256x2048xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<2048x32x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x256x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<2048x32x256xf32>) -> tensor<32x256x2048xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<32x256x2048xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<32x256x2048xcomplex<f32>>
    return %2 : tensor<32x256x2048xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<2048x32x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<32x256x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<2048x32x256xf32>) -> tensor<2048x32x256xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23267429812256"} : (tensor<2048x32x256xcomplex<f32>>) -> tensor<32x256x2048xcomplex<f32>>
    return %1 : tensor<32x256x2048xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[2048,32,256]. let
    b:c64[32,256,2048] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[2048,32,256]. let
          d:c64[32,256,2048] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[2048,32,256]. let
                f:c64[32,256,2048] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x15295edb67a0>
                  fun_jaxpr={ lambda ; g:f32[2048,32,256]. let
                      h:c64[32,256,2048] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[2048,32,256]. let
                            j:c64[2048,32,256] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[32,256,2048] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x15295edb6b00>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x15295edb65f0>
                  symbolic_zeros=False
                ] e
                l:c64[32,256,2048] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
