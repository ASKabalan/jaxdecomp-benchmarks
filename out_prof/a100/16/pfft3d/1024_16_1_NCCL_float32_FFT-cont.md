# Reporting for FFT-cont
## Parameters
| Parameter   | Value    |
|-------------|----------|
| Function    | FFT-cont |
| Precision   | float32  |
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
| JIT Time       | 6473.510376992635  |
| Min Time       | 185.37222290039062 |
| Max Time       | 257.021728515625   |
| Mean Time      | 193.0948028564453  |
| Std Time       | 21.31307029724121  |
| Last Time      | 186.41326904296875 |
| Generated Code | 5.23 KB            |
| Argument Size  | 256.00 MB          |
| Output Size    | 512.00 MB          |
| Temporary Size | 1.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 257.022 |
| Run 1       | 185.642 |
| Run 2       | 185.372 |
| Run 3       | 186.913 |
| Run 4       | 186.186 |
| Run 5       | 185.958 |
| Run 6       | 185.583 |
| Run 7       | 185.937 |
| Run 8       | 185.922 |
| Run 9       | 186.413 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[64,1024,1024]{2,1,0})->c64[64,64,16384]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="d1f744fc1273af671c7db78bb4e5da16"}

%fused_broadcast () -> s8[1073741824] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14d8aca2a170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14d8aca2a0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[1073741824]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14d8aca2a170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14d8aca2a0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f32[64,1024,1024]) -> c64[64,1024,1024] {
  %param_0.1 = f32[64,1024,1024]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c64[64,1024,1024]{2,1,0} convert(f32[64,1024,1024]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f32[64,1024,1024]) -> c64[64,64,16384] {
  %param.1 = f32[64,1024,1024]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="x"}
  %wrapped_convert = c64[64,1024,1024]{2,1,0} fusion(f32[64,1024,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[1073741824]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14d8aca2a170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14d8aca2a0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c64[64,1024,1024]{2,1,0} custom-call(c64[64,1024,1024]{2,1,0} %wrapped_convert, s8[1073741824]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c64[64,1024,1024]{2,1,0}, s8[1073741824]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14d8aca2a170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14d8aca2a0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\001\000\000\004\000\000\000@\000\000@\000\000\000\001\000\000\000\000\000\000\000\000\004\000\000\000@\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  ROOT %bitcast.16 = c64[64,64,16384]{2,1,0} bitcast(c64[64,1024,1024]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14d8aca2a170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14d8aca2a0e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x16384x1024xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<1024x64x16384xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<64x16384x1024xf32>) -> tensor<1024x64x16384xcomplex<f32>>
    return %0 : tensor<1024x64x16384xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<64x16384x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<64x16384x1024xf32>) -> tensor<1024x64x16384xcomplex<f32>>
    return %0 : tensor<1024x64x16384xcomplex<f32>>
  }
  func.func private @_do_pfft(%arg0: tensor<64x16384x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<64x16384x1024xf32>) -> tensor<1024x64x16384xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f32>>) -> tensor<1024x64x16384xcomplex<f32>>
    %2 = stablehlo.multiply %0, %1 : tensor<1024x64x16384xcomplex<f32>>
    return %2 : tensor<1024x64x16384xcomplex<f32>>
  }
  func.func private @pfft_impl(%arg0: tensor<64x16384x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x16384xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<64x16384x1024xf32>) -> tensor<64x16384x1024xcomplex<f32>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22914838152960"} : (tensor<64x16384x1024xcomplex<f32>>) -> tensor<1024x64x16384xcomplex<f32>>
    return %1 : tensor<1024x64x16384xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[64,16384,1024]. let
    b:c64[1024,64,16384] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[64,16384,1024]. let
          d:c64[1024,64,16384] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f32[64,16384,1024]. let
                f:c64[1024,64,16384] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14d746c1a7a0>
                  fun_jaxpr={ lambda ; g:f32[64,16384,1024]. let
                      h:c64[1024,64,16384] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f32[64,16384,1024]. let
                            j:c64[64,16384,1024] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[1024,64,16384] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14d746c1ab00>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14d746c1a5f0>
                  symbolic_zeros=False
                ] e
                l:c64[1024,64,16384] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
