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
| PY          | 4        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 983.7563679320738  |
| Min Time       | 1.0061802864074707 |
| Max Time       | 74.8157958984375   |
| Mean Time      | 8.439569473266602  |
| Std Time       | 22.125423431396484 |
| Last Time      | 1.0735962390899658 |
| Generated Code | 5.48 KB            |
| Argument Size  | 16.00 MB           |
| Output Size    | 32.00 MB           |
| Temporary Size | 64.00 MB           |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 74.8158  |
| Run 1       |  1.10762 |
| Run 2       |  1.06391 |
| Run 3       |  1.08421 |
| Run 4       |  1.04121 |
| Run 5       |  1.07615 |
| Run 6       |  1.06266 |
| Run 7       |  1.00618 |
| Run 8       |  1.06437 |
| Run 9       |  1.0736  |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[256,64,256]{2,1,0})->c64[64,64,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=4, frontend_attributes={fingerprint_before_lhs="a0b1f71e1062a89c7b51af42dd9792bc"}

%fused_broadcast () -> s8[67108864] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x151e500fe950> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x151e500fe8c0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[67108864]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x151e500fe950> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x151e500fe8c0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[256,64,256]) -> c64[256,64,256] {
  %param_0.1 = f32[256,64,256]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[256,64,256]{2,1,0} convert(f32[256,64,256]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[256,64,256]) -> c64[64,64,1024] {
  %param.1 = f32[256,64,256]{2,1,0} parameter(0), sharding={devices=[4,1,1]<=[4]}, metadata={op_name="x"}
  %wrapped_convert = c64[256,64,256]{2,1,0} fusion(f32[256,64,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[67108864]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x151e500fe950> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x151e500fe8c0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[256,64,256]{2,1,0} custom-call(c64[256,64,256]{2,1,0} %wrapped_convert, s8[67108864]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[256,64,256]{2,1,0}, s8[67108864]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x151e500fe950> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x151e500fe8c0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\004\000\000\000\001\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\001\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[64,64,1024]{2,1,0} bitcast(c64[256,64,256]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x151e500fe950> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x151e500fe8c0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/gpfsdswork/projects/rech/tkc/commun/venv/v100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x64x256xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,1,1]<=[4]}"}) -> (tensor<64x256x1024xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<1024x64x256xf32>) -> tensor<64x256x1024xcomplex<f32>>
    return %0 : tensor<64x256x1024xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<1024x64x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<64x256x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<1024x64x256xf32>) -> tensor<64x256x1024xcomplex<f32>>
    return %0 : tensor<64x256x1024xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<1024x64x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<64x256x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<1024x64x256xf32>) -> tensor<64x256x1024xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<64x256x1024xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<64x256x1024xcomplex<f32>>
    return %2 : tensor<64x256x1024xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<1024x64x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<64x256x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1024x64x256xf32>) -> tensor<1024x64x256xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "23214137755008"} : (tensor<1024x64x256xcomplex<f32>>) -> tensor<64x256x1024xcomplex<f32>>
    return %1 : tensor<64x256x1024xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[1024,64,256]. let
    b:c64[64,256,1024] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[1024,64,256]. let
          d:c64[64,256,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[1024,64,256]. let
                f:c64[64,256,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x151cf66817e0>
                  fun_jaxpr={ lambda ; g:f32[1024,64,256]. let
                      h:c64[64,256,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[1024,64,256]. let
                            j:c64[1024,64,256] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[64,256,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x151cf6681b40>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x151cf6681630>
                  symbolic_zeros=False
                ] e
                l:c64[64,256,1024] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
