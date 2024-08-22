# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float32  |
| X           | 2048     |
| Y           | 2048     |
| Z           | 2048     |
| PX          | 2        |
| PY          | 4        |
| Backend     | NCCL     |
| Nodes       | 1        |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 8323.504990999936  |
| Min Time       | 167.60458374023438 |
| Max Time       | 250.06431579589844 |
| Mean Time      | 176.47007751464844 |
| Std Time       | 24.532989501953125 |
| Last Time      | 168.1676483154297  |
| Generated Code | 5.35 KB            |
| Argument Size  | 4.00 GB            |
| Output Size    | 8.00 GB            |
| Temporary Size | 16.00 GB           |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 250.064 |
| Run 1       | 167.605 |
| Run 2       | 168.503 |
| Run 3       | 168.456 |
| Run 4       | 168.652 |
| Run 5       | 168.384 |
| Run 6       | 168.431 |
| Run 7       | 168.064 |
| Run 8       | 168.374 |
| Run 9       | 168.168 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[1024,512,2048]{2,1,0})->c64[256,1024,4096]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="1f224a3a5a3eab15146f1648de1a59be"}

%fused_broadcast () -> s8[17179869184] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14aaf07b6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14aaf07b60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[17179869184]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14aaf07b6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14aaf07b60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[1024,512,2048]) -> c64[1024,512,2048] {
  %param_0.1 = f32[1024,512,2048]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[1024,512,2048]{2,1,0} convert(f32[1024,512,2048]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[1024,512,2048]) -> c64[256,1024,4096] {
  %param.1 = f32[1024,512,2048]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c64[1024,512,2048]{2,1,0} fusion(f32[1024,512,2048]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[17179869184]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14aaf07b6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14aaf07b60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[1024,512,2048]{2,1,0} custom-call(c64[1024,512,2048]{2,1,0} %wrapped_convert, s8[17179869184]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[1024,512,2048]{2,1,0}, s8[17179869184]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14aaf07b6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14aaf07b60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\010\000\000\000\004\000\000\000\020\000\000\002\000\000\000\000\000\000\000\000\010\000\000\000\004\000\000\000\020\000\000\000\000\000\000\000\000\000\000\000\000\000\000\002\000\000\000\004\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[256,1024,4096]{2,1,0} bitcast(c64[1024,512,2048]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14aaf07b6170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14aaf07b60e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x1024x2048xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,2,1]<=[8]}"}) -> (tensor<1024x2048x4096xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<4096x1024x2048xf32>) -> tensor<1024x2048x4096xcomplex<f32>>
    return %0 : tensor<1024x2048x4096xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<4096x1024x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x2048x4096xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<4096x1024x2048xf32>) -> tensor<1024x2048x4096xcomplex<f32>>
    return %0 : tensor<1024x2048x4096xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<4096x1024x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x2048x4096xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<4096x1024x2048xf32>) -> tensor<1024x2048x4096xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<1024x2048x4096xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x2048x4096xcomplex<f32>>
    return %2 : tensor<1024x2048x4096xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<4096x1024x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x2048x4096xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<4096x1024x2048xf32>) -> tensor<4096x1024x2048xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22718407854208"} : (tensor<4096x1024x2048xcomplex<f32>>) -> tensor<1024x2048x4096xcomplex<f32>>
    return %1 : tensor<1024x2048x4096xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[4096,1024,2048]. let
    b:c64[1024,2048,4096] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[4096,1024,2048]. let
          d:c64[1024,2048,4096] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[4096,1024,2048]. let
                f:c64[1024,2048,4096] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14a98a9927a0>
                  fun_jaxpr={ lambda ; g:f32[4096,1024,2048]. let
                      h:c64[1024,2048,4096] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[4096,1024,2048]. let
                            j:c64[4096,1024,2048] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[1024,2048,4096] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14a98a992b00>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14a98a9925f0>
                  symbolic_zeros=False
                ] e
                l:c64[1024,2048,4096] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
