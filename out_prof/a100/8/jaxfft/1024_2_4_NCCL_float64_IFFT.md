# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 1024    |
| Y           | 1024    |
| Z           | 1024    |
| PX          | 2       |
| PY          | 4       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 739.8843490009313  |
| Min Time       | 53.72349499930351  |
| Max Time       | 55.67243475070427  |
| Mean Time      | 54.1093487750004   |
| Std Time       | 0.6775053974521335 |
| Last Time      | 53.7751743759145   |
| Generated Code | 18.78 KB           |
| Argument Size  | 2.00 GB            |
| Output Size    | 2.00 GB            |
| Temporary Size | 4.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 55.6724 |
| Run 1       | 55.2185 |
| Run 2       | 53.7713 |
| Run 3       | 53.7272 |
| Run 4       | 53.7313 |
| Run 5       | 53.7508 |
| Run 6       | 53.8908 |
| Run 7       | 53.8324 |
| Run 8       | 53.7235 |
| Run 9       | 53.7752 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[256,512,1024]{2,1,0})->c128[256,512,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="ba31737c735a40f999547908c34ab358"}

%fused_transpose (param_0.1: c128[512,256,1024]) -> c128[256,512,2,512] {
  %param_0.1 = c128[512,256,1024]{1,0,2} parameter(0)
  %bitcast.61.1 = c128[2,512,512,256]{3,2,1,0} bitcast(c128[512,256,1024]{1,0,2} %param_0.1)
  ROOT %transpose.31.1 = c128[256,512,2,512]{3,2,1,0} transpose(c128[2,512,512,256]{3,2,1,0} %bitcast.61.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

%fused_transpose.1 (param_0.3: c128[256,512,1024]) -> c128[512,256,4,256] {
  %param_0.3 = c128[256,512,1024]{1,0,2} parameter(0)
  %bitcast.49.1 = c128[4,256,256,512]{3,2,1,0} bitcast(c128[256,512,1024]{1,0,2} %param_0.3)
  ROOT %transpose.29.1 = c128[512,256,4,256]{3,2,1,0} transpose(c128[4,256,256,512]{3,2,1,0} %bitcast.49.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.6: c128[256,512,1024]) -> c128[1024,256,512] {
  %param_0.6 = c128[256,512,1024]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c128[1024,256,512]{2,1,0} transpose(c128[256,512,1024]{2,1,0} %param_0.6), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.7: c128[512,256,1024]) -> c128[1024,512,256] {
  %param_0.7 = c128[512,256,1024]{2,1,0} parameter(0)
  ROOT %transpose.30.1 = c128[1024,512,256]{2,1,0} transpose(c128[512,256,1024]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
}

ENTRY %main.39_spmd (param.1: c128[256,512,1024]) -> c128[256,512,1024] {
  %param.1 = c128[256,512,1024]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}, metadata={op_name="arr"}
  %fft.15.0 = c128[256,512,1024]{2,1,0} fft(c128[256,512,1024]{2,1,0} %param.1), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[1024,256,512]{2,1,0} fusion(c128[256,512,1024]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.46.0 = c128[256,512,1024]{1,0,2} bitcast(c128[1024,256,512]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[256,512,1024]{1,0,2}), c128[256,512,1024]{1,0,2}) all-to-all-start(c128[256,512,1024]{1,0,2} %bitcast.46.0), channel_id=1, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[256,512,1024]{1,0,2} all-to-all-done(((c128[256,512,1024]{1,0,2}), c128[256,512,1024]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion.1 = c128[512,256,4,256]{3,2,1,0} fusion(c128[256,512,1024]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.53.0 = c128[512,256,1024]{2,1,0} bitcast(c128[512,256,4,256]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c128[512,256,1024]{2,1,0} fft(c128[512,256,1024]{2,1,0} %bitcast.53.0), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[1024,512,256]{2,1,0} fusion(c128[512,256,1024]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %bitcast.58.0 = c128[512,256,1024]{1,0,2} bitcast(c128[1024,512,256]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c128[512,256,1024]{1,0,2}), c128[512,256,1024]{1,0,2}) all-to-all-start(c128[512,256,1024]{1,0,2} %bitcast.58.0), channel_id=2, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c128[512,256,1024]{1,0,2} all-to-all-done(((c128[512,256,1024]{1,0,2}), c128[512,256,1024]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %loop_transpose_fusion = c128[256,512,2,512]{3,2,1,0} fusion(c128[512,256,1024]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %bitcast.65.0 = c128[256,512,1024]{2,1,0} bitcast(c128[256,512,2,512]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c128[256,512,1024]{2,1,0} fft(c128[256,512,1024]{2,1,0} %bitcast.65.0), fft_type=IFFT, fft_length={1024}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,2,1]<=[8]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<256x512x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %4 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<256x512x1024xcomplex<f64>>) -> (tensor<256x512x1024xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<256x512x1024xcomplex<f64>>) -> tensor<1024x512x256xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<1024x512x256xcomplex<f64>>) -> tensor<512x256x1024xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<512x256x1024xcomplex<f64>>) -> tensor<512x256x1024xcomplex<f64>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<512x256x1024xcomplex<f64>>) -> tensor<1024x256x512xcomplex<f64>>
    %5 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<1024x256x512xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    %6 = call @fft_1(%5) : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    return %6 : tensor<256x512x1024xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<256x512x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x512x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    return %0 : tensor<256x512x1024xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<512x256x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<512x256x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<512x256x1024xcomplex<f64>>) -> tensor<512x256x1024xcomplex<f64>>
    return %0 : tensor<512x256x1024xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<256x512x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x512x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [1024] : (tensor<256x512x1024xcomplex<f64>>) -> tensor<256x512x1024xcomplex<f64>>
    return %0 : tensor<256x512x1024xcomplex<f64>>
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
                  jaxpr={ lambda ; g:c128[256,512,1024]. let
                      h:c128[256,512,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[256,512,1024]. let
                            j:c128[256,512,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[1024,512,256] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[512,256,1024] = transpose[permutation=(1, 2, 0)] k
                      m:c128[512,256,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[512,256,1024]. let
                            o:c128[512,256,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[1024,256,512] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[256,512,1024] = transpose[permutation=(1, 2, 0)] p
                      r:c128[256,512,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[256,512,1024]. let
                            t:c128[256,512,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 4, 'y': 2)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
