# Reporting for IFFT-cont
## Parameters
| Parameter   | Value     |
|-------------|-----------|
| Function    | IFFT-cont |
| Precision   | float64   |
| X           | 1024      |
| Y           | 1024      |
| Z           | 1024      |
| PX          | 1         |
| PY          | 16        |
| Backend     | NCCL      |
| Nodes       | 2         |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 452.4426150601357  |
| Min Time       | 371.10794850741513 |
| Max Time       | 374.19359262275975 |
| Mean Time      | 372.37165822734823 |
| Std Time       | 0.9364528188184775 |
| Last Time      | 371.12565487768734 |
| Generated Code | 5.35 KB            |
| Argument Size  | 1.00 GB            |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 372.509 |
| Run 1       | 371.995 |
| Run 2       | 371.108 |
| Run 3       | 373.118 |
| Run 4       | 372.897 |
| Run 5       | 371.552 |
| Run 6       | 374.194 |
| Run 7       | 373.119 |
| Run 8       | 372.099 |
| Run 9       | 371.126 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[64,64,16384]{2,1,0})->c128[1024,64,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="d8ff501d62e06d6f3b2f3050b18f6fd1"}

%fused_multiply (param_0: c128[64,64,16384]) -> c128[64,64,16384] {
  %param_0 = c128[64,64,16384]{2,1,0} parameter(0)
  %constant_7_1 = c128[] constant((9.3132257461547852e-10, 0)), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/div" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=44}
  %broadcast.9.1 = c128[64,64,16384]{2,1,0} broadcast(c128[] %constant_7_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %multiply.5.1 = c128[64,64,16384]{2,1,0} multiply(c128[64,64,16384]{2,1,0} %param_0, c128[64,64,16384]{2,1,0} %broadcast.9.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}

%fused_broadcast () -> s8[2147483648] {
  %constant_6_1 = s8[] constant(0), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14bd3ff86170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14bd3ff860e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.8.1 = s8[2147483648]{0} broadcast(s8[] %constant_6_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14bd3ff86170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14bd3ff860e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_copy_computation (param_0.2: c128[64,64,16384]) -> c128[64,64,16384] {
  %param_0.2 = c128[64,64,16384]{2,1,0} parameter(0)
  ROOT %copy.3 = c128[64,64,16384]{2,1,0} copy(c128[64,64,16384]{2,1,0} %param_0.2)
}

ENTRY %main.22_spmd (param.1: c128[64,64,16384]) -> c128[1024,64,1024] {
  %param.1 = c128[64,64,16384]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="x"}
  %wrapped_copy = c128[64,64,16384]{2,1,0} fusion(c128[64,64,16384]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_copy_computation
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14bd3ff86170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14bd3ff860e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.3.0 = c128[64,64,16384]{2,1,0} custom-call(c128[64,64,16384]{2,1,0} %wrapped_copy, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[64,64,16384]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x14bd3ff86170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x14bd3ff860e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, True]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\001\000\000\000@\000\000\000\004\000\000@\000\000\000\000\000\000\000\001\000\000\000\000@\000\000\000\004\000\000@\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\020\000\000\000\001\000\000\000\004\000\000\000\001\001\001\000\003\000\000\000"
  %loop_multiply_fusion = c128[64,64,16384]{2,1,0} fusion(c128[64,64,16384]{2,1,0} %custom-call.3.0), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %bitcast.20 = c128[1024,64,1024]{2,1,0} bitcast(c128[64,64,16384]{2,1,0} %loop_multiply_fusion), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1024x16384xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<16384x64x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<64x1024x16384xcomplex<f64>>) -> tensor<16384x64x1024xcomplex<f64>>
    return %0 : tensor<16384x64x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<64x1024x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<64x1024x16384xcomplex<f64>>) -> tensor<16384x64x1024xcomplex<f64>>
    return %0 : tensor<16384x64x1024xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<64x1024x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(6.400000e+01,0.000000e+00), (1.024000e+03,0.000000e+00), (1.638400e+04,0.000000e+00)]> : tensor<3xcomplex<f64>>
    %0 = call @pfft_impl(%arg0) : (tensor<64x1024x16384xcomplex<f64>>) -> tensor<16384x64x1024xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
    %cst_1 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.divide %cst_1, %1 : tensor<complex<f64>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f64>>) -> tensor<16384x64x1024xcomplex<f64>>
    %4 = stablehlo.multiply %0, %3 : tensor<16384x64x1024xcomplex<f64>>
    return %4 : tensor<16384x64x1024xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<64x1024x16384xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<16384x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @CustomSPMDPartitioning(%arg0) {api_version = 2 : i32, backend_config = "22731370859536"} : (tensor<64x1024x16384xcomplex<f64>>) -> tensor<16384x64x1024xcomplex<f64>>
    return %0 : tensor<16384x64x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[64,1024,16384]. let
    b:c128[16384,64,1024] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[64,1024,16384]. let
          d:c128[16384,64,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda e:c128[3]; f:c128[64,1024,16384]. let
                g:c128[16384,64,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x14ac8f3d35b0>
                  fun_jaxpr={ lambda ; h:c128[64,1024,16384]. let
                      i:c128[16384,64,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; j:c128[64,1024,16384]. let
                            k:c128[16384,64,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                              local_transpose=True
                            ] j
                          in (k,) }
                      ] h
                    in (i,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x14ac8f3d3760>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x14ac8f3d3370>
                  symbolic_zeros=False
                ] f
                l:c128[] = reduce_prod[axes=(0,)] e
                m:c128[] = div (1+0j) l
                n:c128[16384,64,1024] = mul g m
              in (n,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
