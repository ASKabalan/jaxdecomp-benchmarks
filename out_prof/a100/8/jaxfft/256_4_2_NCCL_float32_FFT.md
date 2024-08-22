# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
| Precision   | float32 |
| X           | 256     |
| Y           | 256     |
| Z           | 256     |
| PX          | 4       |
| PY          | 2       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 7141.248693002126  |
| Min Time       | 0.4787384867668152 |
| Max Time       | 121.82028198242188 |
| Mean Time      | 12.631946563720703 |
| Std Time       | 36.396114349365234 |
| Last Time      | 0.4966415762901306 |
| Generated Code | 16.47 KB           |
| Argument Size  | 8.00 MB            |
| Output Size    | 16.00 MB           |
| Temporary Size | 32.00 MB           |
---
## Iteration Runs
| Iteration   |       Time |
|-------------|------------|
| Run 0       | 121.82     |
| Run 1       |   0.508742 |
| Run 2       |   0.489809 |
| Run 3       |   0.478738 |
| Run 4       |   0.519229 |
| Run 5       |   0.485652 |
| Run 6       |   0.521009 |
| Run 7       |   0.516718 |
| Run 8       |   0.482641 |
| Run 9       |   0.496642 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[128,64,256]{2,1,0})->c64[128,64,256]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="4d6a18a13a121c751cd2b66df35e18c2"}

%fused_transpose (param_0.1: c64[64,128,256]) -> c64[128,64,2,128] {
  %param_0.1 = c64[64,128,256]{1,0,2} parameter(0)
  %bitcast.63.1 = c64[2,128,64,128]{3,2,1,0} bitcast(c64[64,128,256]{1,0,2} %param_0.1)
  ROOT %transpose.29.1 = c64[128,64,2,128]{3,2,1,0} transpose(c64[2,128,64,128]{3,2,1,0} %bitcast.63.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

%fused_transpose.1 (param_0.3: c64[128,64,256]) -> c64[64,128,4,64] {
  %param_0.3 = c64[128,64,256]{1,0,2} parameter(0)
  %bitcast.51.1 = c64[4,64,128,64]{3,2,1,0} bitcast(c64[128,64,256]{1,0,2} %param_0.3)
  ROOT %transpose.26.1 = c64[64,128,4,64]{3,2,1,0} transpose(c64[4,64,128,64]{3,2,1,0} %bitcast.51.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_convert_computation (param_0.6: f32[128,64,256]) -> c64[128,64,256] {
  %param_0.6 = f32[128,64,256]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c64[128,64,256]{2,1,0} convert(f32[128,64,256]{2,1,0} %param_0.6), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.7: c64[128,64,256]) -> c64[256,128,64] {
  %param_0.7 = c64[128,64,256]{2,1,0} parameter(0)
  ROOT %transpose.24.1 = c64[256,128,64]{2,1,0} transpose(c64[128,64,256]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation.1 (param_0.8: c64[64,128,256]) -> c64[256,64,128] {
  %param_0.8 = c64[64,128,256]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c64[256,64,128]{2,1,0} transpose(c64[64,128,256]{2,1,0} %param_0.8), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
}

ENTRY %main.40_spmd (param.1: f32[128,64,256]) -> c64[128,64,256] {
  %param.1 = f32[128,64,256]{2,1,0} parameter(0), sharding={devices=[2,4,1]<=[8]}, metadata={op_name="arr"}
  %wrapped_convert = c64[128,64,256]{2,1,0} fusion(f32[128,64,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c64[128,64,256]{2,1,0} fft(c64[128,64,256]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c64[256,128,64]{2,1,0} fusion(c64[128,64,256]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %bitcast.48.0 = c64[128,64,256]{1,0,2} bitcast(c64[256,128,64]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c64[128,64,256]{1,0,2}), c64[128,64,256]{1,0,2}) all-to-all-start(c64[128,64,256]{1,0,2} %bitcast.48.0), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c64[128,64,256]{1,0,2} all-to-all-done(((c64[128,64,256]{1,0,2}), c64[128,64,256]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %loop_transpose_fusion.1 = c64[64,128,4,64]{3,2,1,0} fusion(c64[128,64,256]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %bitcast.55.0 = c64[64,128,256]{2,1,0} bitcast(c64[64,128,4,64]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c64[64,128,256]{2,1,0} fft(c64[64,128,256]{2,1,0} %bitcast.55.0), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c64[256,64,128]{2,1,0} fusion(c64[64,128,256]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %bitcast.60.0 = c64[64,128,256]{1,0,2} bitcast(c64[256,64,128]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c64[64,128,256]{1,0,2}), c64[64,128,256]{1,0,2}) all-to-all-start(c64[64,128,256]{1,0,2} %bitcast.60.0), channel_id=2, replica_groups={{0,4},{1,5},{2,6},{3,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c64[64,128,256]{1,0,2} all-to-all-done(((c64[64,128,256]{1,0,2}), c64[64,128,256]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %loop_transpose_fusion = c64[128,64,2,128]{3,2,1,0} fusion(c64[64,128,256]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %bitcast.67.0 = c64[128,64,256]{2,1,0} bitcast(c64[128,64,2,128]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c64[128,64,256]{2,1,0} fft(c64[128,64,256]{2,1,0} %bitcast.67.0), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256x256xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[2,4,1]<=[8]}"}) -> (tensor<256x256x256xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<256x256x256xf32>) -> tensor<256x256x256xcomplex<f32>>
    return %0 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<256x256x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<256x256x256xf32>) -> tensor<256x256x256xcomplex<f32>>
    return %0 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @fft3d(%arg0: tensor<256x256x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4,1]<=[8]}"} : (tensor<256x256x256xf32>) -> tensor<256x256x256xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x256x256xf32>) -> tensor<128x64x256xf32>
    %2 = call @shmap_body(%1) : (tensor<128x64x256xf32>) -> tensor<128x64x256xcomplex<f32>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x64x256xcomplex<f32>>) -> tensor<128x64x256xcomplex<f32>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,4,1]<=[8]}"} : (tensor<128x64x256xcomplex<f32>>) -> tensor<256x256x256xcomplex<f32>>
    return %4 : tensor<256x256x256xcomplex<f32>>
  }
  func.func private @shmap_body(%arg0: tensor<128x64x256xf32>) -> (tensor<128x64x256xcomplex<f32>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<128x64x256xf32>) -> tensor<128x64x256xcomplex<f32>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<128x64x256xcomplex<f32>>) -> tensor<128x256x64xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<128x256x64xcomplex<f32>>) -> tensor<64x128x256xcomplex<f32>>
    %3 = call @fft_0(%2) : (tensor<64x128x256xcomplex<f32>>) -> tensor<64x128x256xcomplex<f32>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<64x128x256xcomplex<f32>>) -> tensor<64x256x128xcomplex<f32>>
    %5 = stablehlo.transpose %4, dims = [2, 0, 1] : (tensor<64x256x128xcomplex<f32>>) -> tensor<128x64x256xcomplex<f32>>
    %6 = call @fft_1(%5) : (tensor<128x64x256xcomplex<f32>>) -> tensor<128x64x256xcomplex<f32>>
    return %6 : tensor<128x64x256xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<128x64x256xf32> {mhlo.layout_mode = "default"}) -> (tensor<128x64x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<128x64x256xf32>) -> tensor<128x64x256xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [256] : (tensor<128x64x256xcomplex<f32>>) -> tensor<128x64x256xcomplex<f32>>
    return %1 : tensor<128x64x256xcomplex<f32>>
  }
  func.func private @fft_0(%arg0: tensor<64x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<64x128x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [256] : (tensor<64x128x256xcomplex<f32>>) -> tensor<64x128x256xcomplex<f32>>
    return %0 : tensor<64x128x256xcomplex<f32>>
  }
  func.func private @fft_1(%arg0: tensor<128x64x256xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<128x64x256xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [256] : (tensor<128x64x256xcomplex<f32>>) -> tensor<128x64x256xcomplex<f32>>
    return %0 : tensor<128x64x256xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[256,256,256]. let
    b:c64[256,256,256] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[256,256,256]. let
          d:c64[256,256,256] = pjit[
            name=fft3d
            jaxpr={ lambda ; e:f32[256,256,256]. let
                f:c64[256,256,256] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:f32[128,64,256]. let
                      h:c64[128,64,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f32[128,64,256]. let
                            j:c64[128,64,256] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[128,64,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c64[128,256,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c64[64,128,256] = transpose[permutation=(2, 0, 1)] l
                      n:c64[64,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c64[64,128,256]. let
                            p:c64[64,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c64[64,256,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c64[128,64,256] = transpose[permutation=(2, 0, 1)] q
                      s:c64[128,64,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c64[128,64,256]. let
                            u:c64[128,64,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] t
                          in (u,) }
                      ] r
                    in (s,) }
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
