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
| PY          | 8       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 294.8156940001354  |
| Min Time       | 36.60894249969715  |
| Max Time       | 38.69749849991422  |
| Mean Time      | 37.31555655003831  |
| Std Time       | 0.8178687915375978 |
| Last Time      | 36.66545899977791  |
| Generated Code | 16.97 KB           |
| Argument Size  | 2.00 GB            |
| Output Size    | 2.00 GB            |
| Temporary Size | 4.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 38.6975 |
| Run 1       | 38.385  |
| Run 2       | 38.3313 |
| Run 3       | 37.6857 |
| Run 4       | 36.6425 |
| Run 5       | 36.7556 |
| Run 6       | 36.6089 |
| Run 7       | 36.6779 |
| Run 8       | 36.7057 |
| Run 9       | 36.6655 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[128,1024,1024]{2,1,0})->c128[128,1024,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="fdc10cf236445ca1496bad31c8fc4909"}

%fused_transpose (param_0.1: c128[128,1024,1024]) -> c128[1024,128,8,128] {
  %param_0.1 = c128[128,1024,1024]{1,0,2} parameter(0)
  %bitcast.34.1 = c128[8,128,128,1024]{3,2,1,0} bitcast(c128[128,1024,1024]{1,0,2} %param_0.1)
  ROOT %transpose.22.1 = c128[1024,128,8,128]{3,2,1,0} transpose(c128[8,128,128,1024]{3,2,1,0} %bitcast.34.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.3: c128[128,1024,1024]) -> c128[1024,128,1024] {
  %param_0.3 = c128[128,1024,1024]{2,1,0} parameter(0)
  ROOT %transpose.20.1 = c128[1024,128,1024]{2,1,0} transpose(c128[128,1024,1024]{2,1,0} %param_0.3), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.4: c128[1024,128,1024]) -> c128[128,1024,1024] {
  %param_0.4 = c128[1024,128,1024]{2,1,0} parameter(0)
  ROOT %transpose.23.1 = c128[128,1024,1024]{2,1,0} transpose(c128[1024,128,1024]{2,1,0} %param_0.4), dimensions={1,2,0}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

ENTRY %main.34_spmd (param.1: c128[128,1024,1024]) -> c128[128,1024,1024] {
  %param.1 = c128[128,1024,1024]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="arr"}
  %fft.15.0 = c128[128,1024,1024]{2,1,0} fft(c128[128,1024,1024]{2,1,0} %param.1), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[1024,128,1024]{2,1,0} fusion(c128[128,1024,1024]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.31.0 = c128[128,1024,1024]{1,0,2} bitcast(c128[1024,128,1024]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[128,1024,1024]{1,0,2}), c128[128,1024,1024]{1,0,2}) all-to-all-start(c128[128,1024,1024]{1,0,2} %bitcast.31.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[128,1024,1024]{1,0,2} all-to-all-done(((c128[128,1024,1024]{1,0,2}), c128[128,1024,1024]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion = c128[1024,128,8,128]{3,2,1,0} fusion(c128[128,1024,1024]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.38.0 = c128[1024,128,1024]{2,1,0} bitcast(c128[1024,128,8,128]{3,2,1,0} %loop_transpose_fusion)
  %fft.16.0 = c128[1024,128,1024]{2,1,0} fft(c128[1024,128,1024]{2,1,0} %bitcast.38.0), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[128,1024,1024]{2,1,0} fusion(c128[1024,128,1024]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  ROOT %fft.17.0 = c128[128,1024,1024]{2,1,0} fft(c128[128,1024,1024]{2,1,0} %wrapped_transpose.1), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %4 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<128x1024x1024xcomplex<f64>>) -> (tensor<128x1024x1024xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, split_count = 8 : i64, split_dimension = 2 : i64}> : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<1024x1024x128xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<1024x1024x128xcomplex<f64>>) -> tensor<1024x128x1024xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<1024x128x1024xcomplex<f64>>) -> tensor<1024x128x1024xcomplex<f64>>
    %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<1024x128x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    %5 = call @fft_1(%4) : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    return %5 : tensor<128x1024x1024xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<128x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    return %0 : tensor<128x1024x1024xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<1024x128x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x128x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<1024x128x1024xcomplex<f64>>) -> tensor<1024x128x1024xcomplex<f64>>
    return %0 : tensor<1024x128x1024xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<128x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<128x1024x1024xcomplex<f64>>) -> tensor<128x1024x1024xcomplex<f64>>
    return %0 : tensor<128x1024x1024xcomplex<f64>>
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
                  jaxpr={ lambda ; g:c128[128,1024,1024]. let
                      h:c128[128,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[128,1024,1024]. let
                            j:c128[128,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[1024,1024,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[1024,128,1024] = transpose[permutation=(1, 2, 0)] k
                      m:c128[1024,128,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[1024,128,1024]. let
                            o:c128[1024,128,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[1024,128,1024] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[128,1024,1024] = transpose[permutation=(1, 2, 0)] p
                      r:c128[128,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[128,1024,1024]. let
                            t:c128[128,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 8, 'y': 1)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
