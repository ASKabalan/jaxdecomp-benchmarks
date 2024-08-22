# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 1024    |
| Y           | 1024    |
| Z           | 1024    |
| PX          | 1       |
| PY          | 16      |
| Backend     | NCCL    |
| Nodes       | 2       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 557.6739970128983  |
| Min Time       | 325.3461346102995  |
| Max Time       | 335.85794331884244 |
| Mean Time      | 327.7709444446373  |
| Std Time       | 2.8740057556216976 |
| Last Time      | 326.29825625917874 |
| Generated Code | 16.84 KB           |
| Argument Size  | 1.00 GB            |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 335.858 |
| Run 1       | 327.724 |
| Run 2       | 327.091 |
| Run 3       | 328.791 |
| Run 4       | 325.566 |
| Run 5       | 325.346 |
| Run 6       | 326.747 |
| Run 7       | 327.834 |
| Run 8       | 326.454 |
| Run 9       | 326.298 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[64,1024,1024]{2,1,0})->c128[64,1024,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="30722a8e6f167e7a33df89f27e51b137"}

%fused_transpose (param_0.1: c128[64,1024,1024]) -> c128[1024,64,16,64] {
  %param_0.1 = c128[64,1024,1024]{1,0,2} parameter(0)
  %bitcast.34.1 = c128[16,64,64,1024]{3,2,1,0} bitcast(c128[64,1024,1024]{1,0,2} %param_0.1)
  ROOT %transpose.22.1 = c128[1024,64,16,64]{3,2,1,0} transpose(c128[16,64,64,1024]{3,2,1,0} %bitcast.34.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.3: c128[64,1024,1024]) -> c128[1024,64,1024] {
  %param_0.3 = c128[64,1024,1024]{2,1,0} parameter(0)
  ROOT %transpose.20.1 = c128[1024,64,1024]{2,1,0} transpose(c128[64,1024,1024]{2,1,0} %param_0.3), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.4: c128[1024,64,1024]) -> c128[64,1024,1024] {
  %param_0.4 = c128[1024,64,1024]{2,1,0} parameter(0)
  ROOT %transpose.23.1 = c128[64,1024,1024]{2,1,0} transpose(c128[1024,64,1024]{2,1,0} %param_0.4), dimensions={1,2,0}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

ENTRY %main.34_spmd (param.1: c128[64,1024,1024]) -> c128[64,1024,1024] {
  %param.1 = c128[64,1024,1024]{2,1,0} parameter(0), sharding={devices=[16,1,1]<=[16]}, metadata={op_name="arr"}
  %fft.15.0 = c128[64,1024,1024]{2,1,0} fft(c128[64,1024,1024]{2,1,0} %param.1), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[1024,64,1024]{2,1,0} fusion(c128[64,1024,1024]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.31.0 = c128[64,1024,1024]{1,0,2} bitcast(c128[1024,64,1024]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[64,1024,1024]{1,0,2}), c128[64,1024,1024]{1,0,2}) all-to-all-start(c128[64,1024,1024]{1,0,2} %bitcast.31.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[64,1024,1024]{1,0,2} all-to-all-done(((c128[64,1024,1024]{1,0,2}), c128[64,1024,1024]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion = c128[1024,64,16,64]{3,2,1,0} fusion(c128[64,1024,1024]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.38.0 = c128[1024,64,1024]{2,1,0} bitcast(c128[1024,64,16,64]{3,2,1,0} %loop_transpose_fusion)
  %fft.16.0 = c128[1024,64,1024]{2,1,0} fft(c128[1024,64,1024]{2,1,0} %bitcast.38.0), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[64,1024,1024]{2,1,0} fusion(c128[1024,64,1024]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  ROOT %fft.17.0 = c128[64,1024,1024]{2,1,0} fft(c128[64,1024,1024]{2,1,0} %wrapped_transpose.1), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[16,1,1]<=[16]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[16,1,1]<=[16]}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[16,1,1]<=[16]}"} : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %4 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<64x1024x1024xcomplex<f64>>) -> (tensor<64x1024x1024xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi64>, split_count = 16 : i64, split_dimension = 2 : i64}> : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<1024x1024x64xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<1024x1024x64xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %5 = call @fft_1(%4) : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    return %5 : tensor<64x1024x1024xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    return %0 : tensor<64x1024x1024xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<1024x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    return %0 : tensor<1024x64x1024xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    return %0 : tensor<64x1024x1024xcomplex<f64>>
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
            name=ifft3d
            jaxpr={ lambda ; e:c128[1024,1024,1024]. let
                f:c128[1024,1024,1024] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:c128[64,1024,1024]. let
                      h:c128[64,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[64,1024,1024]. let
                            j:c128[64,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[1024,1024,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[1024,64,1024] = transpose[permutation=(1, 2, 0)] k
                      m:c128[1024,64,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[1024,64,1024]. let
                            o:c128[1024,64,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[1024,64,1024] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[64,1024,1024] = transpose[permutation=(1, 2, 0)] p
                      r:c128[64,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[64,1024,1024]. let
                            t:c128[64,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 16, 'y': 1)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
