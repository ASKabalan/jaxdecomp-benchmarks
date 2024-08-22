# Reporting for IFFT-noncont
## Parameters
| Parameter   | Value        |
|-------------|--------------|
| Function    | IFFT-noncont |
| Precision   | float64      |
| X           | 1024         |
| Y           | 1024         |
| Z           | 1024         |
| PX          | 8            |
| PY          | 1            |
| Backend     | NCCL         |
| Nodes       | 1            |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 120.83049099965137 |
| Min Time       | 39.47742437458146  |
| Max Time       | 44.24266162504864  |
| Mean Time      | 40.92623221267786  |
| Std Time       | 1.9882674805545582 |
| Last Time      | 39.47742437458146  |
| Generated Code | 5.35 KB            |
| Argument Size  | 2.00 GB            |
| Output Size    | 2.00 GB            |
| Temporary Size | 4.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 43.7746 |
| Run 1       | 43.8504 |
| Run 2       | 44.2427 |
| Run 3       | 39.6331 |
| Run 4       | 39.6232 |
| Run 5       | 39.6042 |
| Run 6       | 39.6103 |
| Run 7       | 39.6085 |
| Run 8       | 39.838  |
| Run 9       | 39.4774 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[128,8192,128]{2,1,0})->c128[128,1024,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="7f0c8b2c56d4b376ed85ec1f4c133bac"}

%fused_multiply (param_0: c128[128,8192,128]) -> c128[128,8192,128] {
  %param_0 = c128[128,8192,128]{2,1,0} parameter(0)
  %constant_7_1 = c128[] constant((9.3132257461547852e-10, 0)), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/div" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=44}
  %broadcast.9.1 = c128[128,8192,128]{2,1,0} broadcast(c128[] %constant_7_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %multiply.5.1 = c128[128,8192,128]{2,1,0} multiply(c128[128,8192,128]{2,1,0} %param_0, c128[128,8192,128]{2,1,0} %broadcast.9.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}

%fused_broadcast () -> s8[4294967296] {
  %constant_6_1 = s8[] constant(0), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.8.1 = s8[4294967296]{0} broadcast(s8[] %constant_6_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_copy_computation (param_0.2: c128[128,8192,128]) -> c128[128,8192,128] {
  %param_0.2 = c128[128,8192,128]{2,1,0} parameter(0)
  ROOT %copy.3 = c128[128,8192,128]{2,1,0} copy(c128[128,8192,128]{2,1,0} %param_0.2)
}

ENTRY %main.22_spmd (param.1: c128[128,8192,128]) -> c128[128,1024,1024] {
  %param.1 = c128[128,8192,128]{2,1,0} parameter(0), sharding={devices=[1,1,8]<=[8]}, metadata={op_name="x"}
  %wrapped_copy = c128[128,8192,128]{2,1,0} fusion(c128[128,8192,128]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_copy_computation
  %loop_broadcast_fusion = s8[4294967296]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.3.0 = c128[128,8192,128]{2,1,0} custom-call(c128[128,8192,128]{2,1,0} %wrapped_copy, s8[4294967296]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[128,8192,128]{2,1,0}, s8[4294967296]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x145dddb92170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x145dddb920e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\000\000\000\004\000\000\000 \000\000\200\000\000\000\001\000\000\000\001\000\000\000\000\004\000\000\000 \000\000\200\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\010\000\000\000\001\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  %loop_multiply_fusion = c128[128,8192,128]{2,1,0} fusion(c128[128,8192,128]{2,1,0} %custom-call.3.0), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %bitcast.20 = c128[128,1024,1024]{2,1,0} bitcast(c128[128,8192,128]{2,1,0} %loop_multiply_fusion), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,1,8]<=[8]}"}) -> (tensor<128x8192x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<128x8192x1024xcomplex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    return %0 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<128x8192x1024xcomplex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    return %0 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(1.280000e+02,0.000000e+00), (8.192000e+03,0.000000e+00), (1.024000e+03,0.000000e+00)]> : tensor<3xcomplex<f64>>
    %0 = call @pfft_impl(%arg0) : (tensor<128x8192x1024xcomplex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
    %cst_1 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.divide %cst_1, %1 : tensor<complex<f64>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    %4 = stablehlo.multiply %0, %3 : tensor<128x8192x1024xcomplex<f64>>
    return %4 : tensor<128x8192x1024xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x8192x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @CustomSPMDPartitioning(%arg0) {api_version = 2 : i32, backend_config = "22387356600016"} : (tensor<128x8192x1024xcomplex<f64>>) -> tensor<128x8192x1024xcomplex<f64>>
    return %0 : tensor<128x8192x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[128,8192,1024]. let
    b:c128[128,8192,1024] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[128,8192,1024]. let
          d:c128[128,8192,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda e:c128[3]; f:c128[128,8192,1024]. let
                g:c128[128,8192,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x145c77f9f130>
                  fun_jaxpr={ lambda ; h:c128[128,8192,1024]. let
                      i:c128[128,8192,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; j:c128[128,8192,1024]. let
                            k:c128[128,8192,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] h
                    in (i,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x145c77f9fb50>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x145c77f9eef0>
                  symbolic_zeros=False
                ] f
                l:c128[] = reduce_prod[axes=(0,)] e
                m:c128[] = div (1+0j) l
                n:c128[128,8192,1024] = mul g m
              in (n,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
