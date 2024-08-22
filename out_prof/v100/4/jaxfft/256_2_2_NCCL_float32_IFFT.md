# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float32 |
| X           | 256     |
| Y           | 256     |
| Z           | 256     |
| PX          | 2       |
| PY          | 2       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value               |
|----------------|---------------------|
| JIT Time       | 295.50328000914305  |
| Min Time       | 2.6375012397766113  |
| Max Time       | 3.01678204536438    |
| Mean Time      | 2.882990837097168   |
| Std Time       | 0.11371580511331558 |
| Last Time      | 3.01678204536438    |
| Generated Code | 7.72 KB             |
| Argument Size  | 32.00 MB            |
| Output Size    | 32.00 MB            |
| Temporary Size | 64.00 MB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 2.7824  |
| Run 1       | 2.6375  |
| Run 2       | 2.80156 |
| Run 3       | 2.85659 |
| Run 4       | 2.99122 |
| Run 5       | 2.96634 |
| Run 6       | 2.99784 |
| Run 7       | 2.86019 |
| Run 8       | 2.91947 |
| Run 9       | 3.01678 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c64[128,128,256]{2,1,0})->c64[128,128,256]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=4, frontend_attributes={fingerprint_before_lhs="02e48e0970b1b10c6052c091e9cd4c05"}

%fused_transpose (param_0.1: c64[128,128,256]) -> c64[128,128,2,128] {
  %param_0.1 = c64[128,128,256]{1,0,2} parameter(0)
  %bitcast.61.1 = c64[2,128,128,128]{3,2,1,0} bitcast(c64[128,128,256]{1,0,2} %param_0.1)
  ROOT %transpose.31.1 = c64[128,128,2,128]{3,2,1,0} transpose(c64[2,128,128,128]{3,2,1,0} %bitcast.61.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

%fused_transpose.1 (param_0.3: c64[128,128,256]) -> c64[128,128,2,128] {
  %param_0.3 = c64[128,128,256]{1,0,2} parameter(0)
  %bitcast.49.1 = c64[2,128,128,128]{3,2,1,0} bitcast(c64[128,128,256]{1,0,2} %param_0.3)
  ROOT %transpose.29.1 = c64[128,128,2,128]{3,2,1,0} transpose(c64[2,128,128,128]{3,2,1,0} %bitcast.49.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.6: c64[128,128,256]) -> c64[256,128,128] {
  %param_0.6 = c64[128,128,256]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c64[256,128,128]{2,1,0} transpose(c64[128,128,256]{2,1,0} %param_0.6), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.7: c64[128,128,256]) -> c64[256,128,128] {
  %param_0.7 = c64[128,128,256]{2,1,0} parameter(0)
  ROOT %transpose.30.1 = c64[256,128,128]{2,1,0} transpose(c64[128,128,256]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

ENTRY %main.39_spmd (param.1: c64[128,128,256]) -> c64[128,128,256] {
  %param.1 = c64[128,128,256]{2,1,0} parameter(0), sharding={devices=[2,2,1]<=[4]}, metadata={op_name="arr"}
  %fft.15.0 = c64[128,128,256]{2,1,0} fft(c64[128,128,256]{2,1,0} %param.1), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c64[256,128,128]{2,1,0} fusion(c64[128,128,256]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.46.0 = c64[128,128,256]{1,0,2} bitcast(c64[256,128,128]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c64[128,128,256]{1,0,2}), c64[128,128,256]{1,0,2}) all-to-all-start(c64[128,128,256]{1,0,2} %bitcast.46.0), channel_id=1, replica_groups={{0,2},{1,3}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c64[128,128,256]{1,0,2} all-to-all-done(((c64[128,128,256]{1,0,2}), c64[128,128,256]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion.1 = c64[128,128,2,128]{3,2,1,0} fusion(c64[128,128,256]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45 deduplicated_name="loop_transpose_fusion"}
  %bitcast.53.0 = c64[128,128,256]{2,1,0} bitcast(c64[128,128,2,128]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c64[128,128,256]{2,1,0} fft(c64[128,128,256]{2,1,0} %bitcast.53.0), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose.1 = c64[256,128,128]{2,1,0} fusion(c64[128,128,256]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.58.0 = c64[128,128,256]{1,0,2} bitcast(c64[256,128,128]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c64[128,128,256]{1,0,2}), c64[128,128,256]{1,0,2}) all-to-all-start(c64[128,128,256]{1,0,2} %bitcast.58.0), channel_id=2, replica_groups={{0,1},{2,3}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c64[128,128,256]{1,0,2} all-to-all-done(((c64[128,128,256]{1,0,2}), c64[128,128,256]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %loop_transpose_fusion = c64[128,128,2,128]{3,2,1,0} fusion(c64[128,128,256]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49 deduplicated_name="loop_transpose_fusion"}
  %bitcast.65.0 = c64[128,128,256]{2,1,0} bitcast(c64[128,128,2,128]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c64[128,128,256]{2,1,0} fft(c64[128,128,256]{2,1,0} %bitcast.65.0), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[2,2,1]<=[4]}"}) -> (tensor<256x256x256xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<256x256x256xcomplex<f32>>) -> tensor<256x256x256xcomplex<f32>>
    return %0 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @do_ifft(%arg0: tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<256x256x256xcomplex<f32>>) -> tensor<256x256x256xcomplex<f32>>
    return %0 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @ifft3d(%arg0: tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,2,1]<=[4]}"} : (tensor<256x256x256xcomplex<f32>>) -> tensor<256x256x256xcomplex<f32>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x256x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %2 = call @shmap_body(%1) : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,2,1]<=[4]}"} : (tensor<128x128x256xcomplex<f32>>) -> tensor<256x256x256xcomplex<f32>>
    return %4 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @shmap_body(%arg0: tensor<128x128x256xcomplex<f32>>) -> (tensor<128x128x256xcomplex<f32>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<128x128x256xcomplex<f32>>) -> tensor<256x128x128xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<256x128x128xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %3 = call @fft_0(%2) : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<128x128x256xcomplex<f32>>) -> tensor<256x128x128xcomplex<f32>>
    %5 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<256x128x128xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    %6 = call @fft_1(%5) : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    return %6 : tensor<128x128x256xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    return %0 : tensor<128x128x256xcomplex<f32>>
  }
  func.func private @fft_0(%arg0: tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    return %0 : tensor<128x128x256xcomplex<f32>>
  }
  func.func private @fft_1(%arg0: tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<128x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<128x128x256xcomplex<f32>>) -> tensor<128x128x256xcomplex<f32>>
    return %0 : tensor<128x128x256xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c64[256,256,256]. let
    b:c64[256,256,256] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c64[256,256,256]. let
          d:c64[256,256,256] = pjit[
            name=ifft3d
            jaxpr={ lambda ; e:c64[256,256,256]. let
                f:c64[256,256,256] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:c64[128,128,256]. let
                      h:c64[128,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c64[128,128,256]. let
                            j:c64[128,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c64[256,128,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c64[128,128,256] = transpose[permutation=(1, 2, 0)] k
                      m:c64[128,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c64[128,128,256]. let
                            o:c64[128,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c64[256,128,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c64[128,128,256] = transpose[permutation=(1, 2, 0)] p
                      r:c64[128,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c64[128,128,256]. let
                            t:c64[128,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 2, 'y': 2)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
