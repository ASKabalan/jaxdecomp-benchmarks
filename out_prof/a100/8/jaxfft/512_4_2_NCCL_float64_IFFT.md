# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 512     |
| Y           | 512     |
| Z           | 512     |
| PX          | 4       |
| PY          | 2       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value               |
|----------------|---------------------|
| JIT Time       | 412.8390939986275   |
| Min Time       | 6.4926245004244265  |
| Max Time       | 6.622409625379078   |
| Mean Time      | 6.525863525257591   |
| Std Time       | 0.03570677162851736 |
| Last Time      | 6.496769624391163   |
| Generated Code | 18.78 KB            |
| Argument Size  | 256.00 MB           |
| Output Size    | 256.00 MB           |
| Temporary Size | 512.00 MB           |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 6.62241 |
| Run 1       | 6.51032 |
| Run 2       | 6.50391 |
| Run 3       | 6.52627 |
| Run 4       | 6.50797 |
| Run 5       | 6.53701 |
| Run 6       | 6.54214 |
| Run 7       | 6.51923 |
| Run 8       | 6.49262 |
| Run 9       | 6.49677 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[256,128,512]{2,1,0})->c128[256,128,512]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="5d2f69d50c3a1d640f9bd8687c7667ce"}

%fused_transpose (param_0.1: c128[128,256,512]) -> c128[256,128,4,128] {
  %param_0.1 = c128[128,256,512]{1,0,2} parameter(0)
  %bitcast.61.1 = c128[4,128,128,256]{3,2,1,0} bitcast(c128[128,256,512]{1,0,2} %param_0.1)
  ROOT %transpose.31.1 = c128[256,128,4,128]{3,2,1,0} transpose(c128[4,128,128,256]{3,2,1,0} %bitcast.61.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

%fused_transpose.1 (param_0.3: c128[256,128,512]) -> c128[128,256,2,256] {
  %param_0.3 = c128[256,128,512]{1,0,2} parameter(0)
  %bitcast.49.1 = c128[2,256,256,128]{3,2,1,0} bitcast(c128[256,128,512]{1,0,2} %param_0.3)
  ROOT %transpose.29.1 = c128[128,256,2,256]{3,2,1,0} transpose(c128[2,256,256,128]{3,2,1,0} %bitcast.49.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.6: c128[256,128,512]) -> c128[512,256,128] {
  %param_0.6 = c128[256,128,512]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c128[512,256,128]{2,1,0} transpose(c128[256,128,512]{2,1,0} %param_0.6), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.7: c128[128,256,512]) -> c128[512,128,256] {
  %param_0.7 = c128[128,256,512]{2,1,0} parameter(0)
  ROOT %transpose.30.1 = c128[512,128,256]{2,1,0} transpose(c128[128,256,512]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
}

ENTRY %main.39_spmd (param.1: c128[256,128,512]) -> c128[256,128,512] {
  %param.1 = c128[256,128,512]{2,1,0} parameter(0), sharding={devices=[2,4,1]<=[8]}, metadata={op_name="arr"}
  %fft.15.0 = c128[256,128,512]{2,1,0} fft(c128[256,128,512]{2,1,0} %param.1), fft_type=IFFT, fft_length={512}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[512,256,128]{2,1,0} fusion(c128[256,128,512]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.46.0 = c128[256,128,512]{1,0,2} bitcast(c128[512,256,128]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[256,128,512]{1,0,2}), c128[256,128,512]{1,0,2}) all-to-all-start(c128[256,128,512]{1,0,2} %bitcast.46.0), channel_id=1, replica_groups={{0,4},{1,5},{2,6},{3,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[256,128,512]{1,0,2} all-to-all-done(((c128[256,128,512]{1,0,2}), c128[256,128,512]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion.1 = c128[128,256,2,256]{3,2,1,0} fusion(c128[256,128,512]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.53.0 = c128[128,256,512]{2,1,0} bitcast(c128[128,256,2,256]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c128[128,256,512]{2,1,0} fft(c128[128,256,512]{2,1,0} %bitcast.53.0), fft_type=IFFT, fft_length={512}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[512,128,256]{2,1,0} fusion(c128[128,256,512]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %bitcast.58.0 = c128[128,256,512]{1,0,2} bitcast(c128[512,128,256]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c128[128,256,512]{1,0,2}), c128[128,256,512]{1,0,2}) all-to-all-start(c128[128,256,512]{1,0,2} %bitcast.58.0), channel_id=2, replica_groups={{0,1,2,3},{4,5,6,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c128[128,256,512]{1,0,2} all-to-all-done(((c128[128,256,512]{1,0,2}), c128[128,256,512]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %loop_transpose_fusion = c128[256,128,4,128]{3,2,1,0} fusion(c128[128,256,512]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %bitcast.65.0 = c128[256,128,512]{2,1,0} bitcast(c128[256,128,4,128]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c128[256,128,512]{2,1,0} fft(c128[256,128,512]{2,1,0} %bitcast.65.0), fft_type=IFFT, fft_length={512}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(512,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<512x512x512xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[2,4,1]<=[8]}"}) -> (tensor<512x512x512xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<512x512x512xcomplex<f64>>) -> tensor<512x512x512xcomplex<f64>>
    return %0 : tensor<512x512x512xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<512x512x512xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<512x512x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<512x512x512xcomplex<f64>>) -> tensor<512x512x512xcomplex<f64>>
    return %0 : tensor<512x512x512xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<512x512x512xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<512x512x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4,1]<=[8]}"} : (tensor<512x512x512xcomplex<f64>>) -> tensor<512x512x512xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<512x512x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,4,1]<=[8]}"} : (tensor<256x128x512xcomplex<f64>>) -> tensor<512x512x512xcomplex<f64>>
    return %4 : tensor<512x512x512xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<256x128x512xcomplex<f64>>) -> (tensor<256x128x512xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<256x128x512xcomplex<f64>>) -> tensor<512x128x256xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<512x128x256xcomplex<f64>>) -> tensor<128x256x512xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<128x256x512xcomplex<f64>>) -> tensor<128x256x512xcomplex<f64>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<128x256x512xcomplex<f64>>) -> tensor<512x256x128xcomplex<f64>>
    %5 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<512x256x128xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    %6 = call @fft_1(%5) : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    return %6 : tensor<256x128x512xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<256x128x512xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x128x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [512] : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    return %0 : tensor<256x128x512xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<128x256x512xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x256x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [512] : (tensor<128x256x512xcomplex<f64>>) -> tensor<128x256x512xcomplex<f64>>
    return %0 : tensor<128x256x512xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<256x128x512xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x128x512xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [512] : (tensor<256x128x512xcomplex<f64>>) -> tensor<256x128x512xcomplex<f64>>
    return %0 : tensor<256x128x512xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[512,512,512]. let
    b:c128[512,512,512] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[512,512,512]. let
          d:c128[512,512,512] = pjit[
            name=ifft3d
            jaxpr={ lambda ; e:c128[512,512,512]. let
                f:c128[512,512,512] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:c128[256,128,512]. let
                      h:c128[256,128,512] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[256,128,512]. let
                            j:c128[256,128,512] = fft[
                              fft_lengths=(512,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[512,128,256] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[128,256,512] = transpose[permutation=(1, 2, 0)] k
                      m:c128[128,256,512] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[128,256,512]. let
                            o:c128[128,256,512] = fft[
                              fft_lengths=(512,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[512,256,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[256,128,512] = transpose[permutation=(1, 2, 0)] p
                      r:c128[256,128,512] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[256,128,512]. let
                            t:c128[256,128,512] = fft[
                              fft_lengths=(512,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 2, 'y': 4)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
