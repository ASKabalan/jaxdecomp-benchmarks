# Reporting for FFT-noncont
## Parameters
| Parameter   | Value       |
|-------------|-------------|
| Function    | FFT-noncont |
| Precision   | float64     |
| X           | 256         |
| Y           | 256         |
| Z           | 256         |
| PX          | 8           |
| PY          | 1           |
| Backend     | NCCL        |
| Nodes       | 1           |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 9637.364291997073  |
| Min Time       | 1.3144828753866022 |
| Max Time       | 63.461269748586346 |
| Mean Time      | 7.538689237162544  |
| Std Time       | 18.64086492778688  |
| Last Time      | 1.3158152491996589 |
| Generated Code | 5.60 KB            |
| Argument Size  | 16.00 MB           |
| Output Size    | 32.00 MB           |
| Temporary Size | 64.00 MB           |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 63.4613  |
| Run 1       |  1.32951 |
| Run 2       |  1.32071 |
| Run 3       |  1.32136 |
| Run 4       |  1.36273 |
| Run 5       |  1.31561 |
| Run 6       |  1.31448 |
| Run 7       |  1.32122 |
| Run 8       |  1.32418 |
| Run 9       |  1.31582 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[32,256,256]{2,1,0})->c128[32,2048,32]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="ce4f2514d1e127053b4c711f373cfda7"}

%fused_broadcast () -> s8[67108864] {
  %constant_2_1 = s8[] constant(0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14590d676170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14590d6760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.4.1 = s8[67108864]{0} broadcast(s8[] %constant_2_1), dimensions={}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14590d676170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14590d6760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_convert_computation (param_0.1: f64[32,256,256]) -> c128[32,256,256] {
  %param_0.1 = f64[32,256,256]{2,1,0} parameter(0)
  ROOT %convert.6.1 = c128[32,256,256]{2,1,0} convert(f64[32,256,256]{2,1,0} %param_0.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
}

ENTRY %main.16_spmd (param.1: f64[32,256,256]) -> c128[32,2048,32] {
  %param.1 = f64[32,256,256]{2,1,0} parameter(0), sharding={devices=[1,8,1]<=[8]}, metadata={op_name="x"}
  %wrapped_convert = c128[32,256,256]{2,1,0} fusion(f64[32,256,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=554}
  %loop_broadcast_fusion = s8[67108864]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14590d676170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14590d6760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.4.0 = c128[32,256,256]{2,1,0} custom-call(c128[32,256,256]{2,1,0} %wrapped_convert, s8[67108864]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[32,256,256]{2,1,0}, s8[67108864]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14590d676170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14590d6760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\001\000\000\001\000\000\000\010\000\000 \000\000\000\001\000\000\000\001\000\000\000\000\001\000\000\000\010\000\000 \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  ROOT %bitcast.16 = c128[32,2048,32]{2,1,0} bitcast(c128[32,256,256]{2,1,0} %custom-call.4.0), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14590d676170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14590d6760e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.FFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x2048x256xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,8,1]<=[8]}"}) -> (tensor<32x2048x256xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<32x2048x256xf64>) -> tensor<32x2048x256xcomplex<f64>>
    return %0 : tensor<32x2048x256xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<32x2048x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<32x2048x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<32x2048x256xf64>) -> tensor<32x2048x256xcomplex<f64>>
    return %0 : tensor<32x2048x256xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<32x2048x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<32x2048x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @pfft_impl(%arg0) : (tensor<32x2048x256xf64>) -> tensor<32x2048x256xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<complex<f64>>) -> tensor<32x2048x256xcomplex<f64>>
    %2 = stablehlo.multiply %0, %1 : tensor<32x2048x256xcomplex<f64>>
    return %2 : tensor<32x2048x256xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<32x2048x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<32x2048x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<32x2048x256xf64>) -> tensor<32x2048x256xcomplex<f64>>
    %1 = stablehlo.custom_call @CustomSPMDPartitioning(%0) {api_version = 2 : i32, backend_config = "22366703775664"} : (tensor<32x2048x256xcomplex<f64>>) -> tensor<32x2048x256xcomplex<f64>>
    return %1 : tensor<32x2048x256xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[32,2048,256]. let
    b:c128[32,2048,256] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[32,2048,256]. let
          d:c128[32,2048,256] = pjit[
            name=_do_pfft
            jaxpr={ lambda ; e:f64[32,2048,256]. let
                f:c128[32,2048,256] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x1457a767a8c0>
                  fun_jaxpr={ lambda ; g:f64[32,2048,256]. let
                      h:c128[32,2048,256] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; i:f64[32,2048,256]. let
                            j:c128[32,2048,256] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[32,2048,256] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.FFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] g
                    in (h,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x1457a767ac20>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x1457a767a710>
                  symbolic_zeros=False
                ] e
                l:c128[32,2048,256] = mul f (1+0j)
              in (l,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
