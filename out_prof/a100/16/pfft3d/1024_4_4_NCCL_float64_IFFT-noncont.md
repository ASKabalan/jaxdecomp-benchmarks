# Reporting for IFFT-noncont
## Parameters
| Parameter   | Value        |
|-------------|--------------|
| Function    | IFFT-noncont |
| Precision   | float64      |
| X           | 1024         |
| Y           | 1024         |
| Z           | 1024         |
| PX          | 4            |
| PY          | 4            |
| Backend     | NCCL         |
| Nodes       | 2            |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 455.37258707918227 |
| Min Time       | 339.1440428895294  |
| Max Time       | 340.1214557379717  |
| Mean Time      | 339.772074935172   |
| Std Time       | 0.2977502659530114 |
| Last Time      | 339.49493605905445 |
| Generated Code | 5.35 KB            |
| Argument Size  | 1.00 GB            |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 340.121 |
| Run 1       | 339.583 |
| Run 2       | 339.648 |
| Run 3       | 339.909 |
| Run 4       | 339.144 |
| Run 5       | 340     |
| Run 6       | 340.038 |
| Run 7       | 340.082 |
| Run 8       | 339.701 |
| Run 9       | 339.495 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[1024,256,256]{2,1,0})->c128[256,256,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="a917785bcd751c3c244059ead154d3f9"}

%fused_multiply (param_0: c128[1024,256,256]) -> c128[1024,256,256] {
  %param_0 = c128[1024,256,256]{2,1,0} parameter(0)
  %constant_7_1 = c128[] constant((9.3132257461547852e-10, 0)), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/div" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=44}
  %broadcast.11.1 = c128[1024,256,256]{2,1,0} broadcast(c128[] %constant_7_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %multiply.5.1 = c128[1024,256,256]{2,1,0} multiply(c128[1024,256,256]{2,1,0} %param_0, c128[1024,256,256]{2,1,0} %broadcast.11.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}

%fused_broadcast () -> s8[2147483648] {
  %constant_6_1 = s8[] constant(0), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15143a0b2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15143a0b20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  ROOT %broadcast.10.1 = s8[2147483648]{0} broadcast(s8[] %constant_6_1), dimensions={}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15143a0b2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15143a0b20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
}

%wrapped_copy_computation (param_0.2: c128[1024,256,256]) -> c128[1024,256,256] {
  %param_0.2 = c128[1024,256,256]{2,1,0} parameter(0)
  ROOT %copy.3 = c128[1024,256,256]{2,1,0} copy(c128[1024,256,256]{2,1,0} %param_0.2)
}

ENTRY %main.23_spmd (param.1: c128[1024,256,256]) -> c128[256,256,1024] {
  %param.1 = c128[1024,256,256]{2,1,0} parameter(0), sharding={devices=[1,4,4]<=[16]}, metadata={op_name="x"}
  %wrapped_copy = c128[1024,256,256]{2,1,0} fusion(c128[1024,256,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_copy_computation
  %loop_broadcast_fusion = s8[2147483648]{0} fusion(), kind=kLoop, calls=%fused_broadcast, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15143a0b2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15143a0b20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}
  %custom-call.3.0 = c128[1024,256,256]{2,1,0} custom-call(c128[1024,256,256]{2,1,0} %wrapped_copy, s8[2147483648]{0} %loop_broadcast_fusion), custom_call_target="pfft3d", operand_layout_constraints={c128[1024,256,256]{2,1,0}, s8[2147483648]{0}}, custom_call_has_side_effect=true, output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/jit(pfft_impl)/custom_partitioning[partition=<function FFTPrimitive.partition at 0x15143a0b2170> propagate_user_sharding=None infer_sharding_from_operands=<function FFTPrimitive.infer_sharding_from_operands at 0x15143a0b20e0> decode_shardings=True in_tree=PyTreeDef((*,)) out_tree=PyTreeDef(*) static_args=[jaxlib.xla_extension.FftType.IFFT, False, False]]" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/_src/fft.py" source_line=556}, backend_config="\000\000\000\000\000\004\000\000\000\004\000\000\000\004\000\000\002\000\000\000\001\000\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000\000\000\000\000\000\000\000\000\000\000\000\004\000\000\000\004\000\000\000\004\000\000\000\000\000\000\000\003\000\000\000"
  %loop_multiply_fusion = c128[1024,256,256]{2,1,0} fusion(c128[1024,256,256]{2,1,0} %custom-call.3.0), kind=kLoop, calls=%fused_multiply, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
  ROOT %bitcast.20 = c128[256,256,1024]{2,1,0} bitcast(c128[1024,256,256]{2,1,0} %loop_multiply_fusion), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(_do_pfft)/mul" source_file="/lustre/fswork/projects/rech/tkc/commun/venv/a100/lib/python3.10/site-packages/jaxdecomp/fft.py" source_line=84}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,4,4]<=[16]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @_do_pfft(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @_do_pfft(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(1.024000e+03,0.000000e+00)> : tensor<3xcomplex<f64>>
    %0 = call @pfft_impl(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %1 = stablehlo.reduce(%cst init: %cst_0) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
    %cst_1 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.divide %cst_1, %1 : tensor<complex<f64>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %4 = stablehlo.multiply %0, %3 : tensor<1024x1024x1024xcomplex<f64>>
    return %4 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @pfft_impl(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @CustomSPMDPartitioning(%arg0) {api_version = 2 : i32, backend_config = "23170611808336"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[1024,1024,1024]. let
    b:c128[1024,1024,1024] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[1024,1024,1024]. let
          d:c128[1024,1024,1024] = pjit[
            name=_do_pfft
            jaxpr={ lambda e:c128[3]; f:c128[1024,1024,1024]. let
                g:c128[1024,1024,1024] = custom_vjp_call_jaxpr[
                  bwd=<function CustomVJPCallPrimitive.bind.<locals>.<lambda> at 0x1512d409f5b0>
                  fun_jaxpr={ lambda ; h:c128[1024,1024,1024]. let
                      i:c128[1024,1024,1024] = pjit[
                        name=pfft_impl
                        jaxpr={ lambda ; j:c128[1024,1024,1024]. let
                            k:c128[1024,1024,1024] = fft_wrapper[
                              adjoint=False
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                              local_transpose=False
                            ] j
                          in (k,) }
                      ] h
                    in (i,) }
                  fwd_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x1512d409f910>
                  num_consts=0
                  out_trees=<function transformation_with_aux.<locals>.<lambda> at 0x1512d409f370>
                  symbolic_zeros=False
                ] f
                l:c128[] = reduce_prod[axes=(0,)] e
                m:c128[] = div (1+0j) l
                n:c128[1024,1024,1024] = mul g m
              in (n,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
