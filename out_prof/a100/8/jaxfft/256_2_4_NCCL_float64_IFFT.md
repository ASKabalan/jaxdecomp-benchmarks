# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 256     |
| Y           | 256     |
| Z           | 256     |
| PX          | 2       |
| PY          | 4       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value               |
|----------------|---------------------|
| JIT Time       | 579.9643430000287   |
| Min Time       | 1.0028052502093487  |
| Max Time       | 1.7696662498565274  |
| Mean Time      | 1.1041142503472656  |
| Std Time       | 0.22248751346733148 |
| Last Time      | 1.0028052502093487  |
| Generated Code | 18.78 KB            |
| Argument Size  | 32.00 MB            |
| Output Size    | 32.00 MB            |
| Temporary Size | 64.00 MB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 1.76967 |
| Run 1       | 1.03286 |
| Run 2       | 1.04853 |
| Run 3       | 1.04786 |
| Run 4       | 1.0126  |
| Run 5       | 1.03927 |
| Run 6       | 1.01579 |
| Run 7       | 1.05538 |
| Run 8       | 1.01639 |
| Run 9       | 1.00281 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[64,128,256]{2,1,0})->c128[64,128,256]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="c3a05872a564241d6a1a2b6ce441bf4b"}

%fused_transpose (param_0.1: c128[128,64,256]) -> c128[64,128,2,128] {
  %param_0.1 = c128[128,64,256]{1,0,2} parameter(0)
  %bitcast.61.1 = c128[2,128,128,64]{3,2,1,0} bitcast(c128[128,64,256]{1,0,2} %param_0.1)
  ROOT %transpose.31.1 = c128[64,128,2,128]{3,2,1,0} transpose(c128[2,128,128,64]{3,2,1,0} %bitcast.61.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

%fused_transpose.1 (param_0.3: c128[64,128,256]) -> c128[128,64,4,64] {
  %param_0.3 = c128[64,128,256]{1,0,2} parameter(0)
  %bitcast.49.1 = c128[4,64,64,128]{3,2,1,0} bitcast(c128[64,128,256]{1,0,2} %param_0.3)
  ROOT %transpose.29.1 = c128[128,64,4,64]{3,2,1,0} transpose(c128[4,64,64,128]{3,2,1,0} %bitcast.49.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.6: c128[64,128,256]) -> c128[256,64,128] {
  %param_0.6 = c128[64,128,256]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c128[256,64,128]{2,1,0} transpose(c128[64,128,256]{2,1,0} %param_0.6), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.7: c128[128,64,256]) -> c128[256,128,64] {
  %param_0.7 = c128[128,64,256]{2,1,0} parameter(0)
  ROOT %transpose.30.1 = c128[256,128,64]{2,1,0} transpose(c128[128,64,256]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
}

ENTRY %main.39_spmd (param.1: c128[64,128,256]) -> c128[64,128,256] {
  %param.1 = c128[64,128,256]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}, metadata={op_name="arr"}
  %fft.15.0 = c128[64,128,256]{2,1,0} fft(c128[64,128,256]{2,1,0} %param.1), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[256,64,128]{2,1,0} fusion(c128[64,128,256]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.46.0 = c128[64,128,256]{1,0,2} bitcast(c128[256,64,128]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[64,128,256]{1,0,2}), c128[64,128,256]{1,0,2}) all-to-all-start(c128[64,128,256]{1,0,2} %bitcast.46.0), channel_id=1, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[64,128,256]{1,0,2} all-to-all-done(((c128[64,128,256]{1,0,2}), c128[64,128,256]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion.1 = c128[128,64,4,64]{3,2,1,0} fusion(c128[64,128,256]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.53.0 = c128[128,64,256]{2,1,0} bitcast(c128[128,64,4,64]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c128[128,64,256]{2,1,0} fft(c128[128,64,256]{2,1,0} %bitcast.53.0), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[256,128,64]{2,1,0} fusion(c128[128,64,256]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %bitcast.58.0 = c128[128,64,256]{1,0,2} bitcast(c128[256,128,64]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c128[128,64,256]{1,0,2}), c128[128,64,256]{1,0,2}) all-to-all-start(c128[128,64,256]{1,0,2} %bitcast.58.0), channel_id=2, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c128[128,64,256]{1,0,2} all-to-all-done(((c128[128,64,256]{1,0,2}), c128[128,64,256]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %loop_transpose_fusion = c128[64,128,2,128]{3,2,1,0} fusion(c128[128,64,256]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %bitcast.65.0 = c128[64,128,256]{2,1,0} bitcast(c128[64,128,2,128]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c128[64,128,256]{2,1,0} fft(c128[64,128,256]{2,1,0} %bitcast.65.0), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,2,1]<=[8]}"}) -> (tensor<256x256x256xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x256x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<64x128x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %4 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<64x128x256xcomplex<f64>>) -> (tensor<64x128x256xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<64x128x256xcomplex<f64>>) -> tensor<256x128x64xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<256x128x64xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<128x64x256xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<128x64x256xcomplex<f64>>) -> tensor<256x64x128xcomplex<f64>>
    %5 = stablehlo.transpose %4, dims = [1, 2, 0] : (tensor<256x64x128xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %6 = call @fft_1(%5) : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %6 : tensor<64x128x256xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %0 : tensor<64x128x256xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<128x64x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x64x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<128x64x256xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    return %0 : tensor<128x64x256xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %0 : tensor<64x128x256xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[256,256,256]. let
    b:c128[256,256,256] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[256,256,256]. let
          d:c128[256,256,256] = pjit[
            name=ifft3d
            jaxpr={ lambda ; e:c128[256,256,256]. let
                f:c128[256,256,256] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:c128[64,128,256]. let
                      h:c128[64,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[64,128,256]. let
                            j:c128[64,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[256,128,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[128,64,256] = transpose[permutation=(1, 2, 0)] k
                      m:c128[128,64,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[128,64,256]. let
                            o:c128[128,64,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[256,64,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[64,128,256] = transpose[permutation=(1, 2, 0)] p
                      r:c128[64,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[64,128,256]. let
                            t:c128[64,128,256] = fft[
                              fft_lengths=(256,)
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
