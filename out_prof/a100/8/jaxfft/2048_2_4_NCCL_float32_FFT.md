# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
| Precision   | float32 |
| X           | 2048    |
| Y           | 2048    |
| Z           | 2048    |
| PX          | 2       |
| PY          | 4       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 6847.770221997052  |
| Min Time       | 124.0154037475586  |
| Max Time       | 245.03164672851562 |
| Mean Time      | 136.28048706054688 |
| Std Time       | 36.25050354003906  |
| Last Time      | 124.19169616699219 |
| Generated Code | 16.59 KB           |
| Argument Size  | 4.00 GB            |
| Output Size    | 8.00 GB            |
| Temporary Size | 16.00 GB           |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 245.032 |
| Run 1       | 124.015 |
| Run 2       | 124.203 |
| Run 3       | 124.344 |
| Run 4       | 124.074 |
| Run 5       | 124.259 |
| Run 6       | 124.206 |
| Run 7       | 124.2   |
| Run 8       | 124.279 |
| Run 9       | 124.192 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[512,1024,2048]{2,1,0})->c64[512,1024,2048]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="58a76c19e4f0381898bf6127b1657aef"}

%fused_transpose (param_0.1: c64[1024,512,2048]) -> c64[512,1024,4,512] {
  %param_0.1 = c64[1024,512,2048]{1,0,2} parameter(0)
  %bitcast.63.1 = c64[4,512,1024,512]{3,2,1,0} bitcast(c64[1024,512,2048]{1,0,2} %param_0.1)
  ROOT %transpose.29.1 = c64[512,1024,4,512]{3,2,1,0} transpose(c64[4,512,1024,512]{3,2,1,0} %bitcast.63.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

%fused_transpose.1 (param_0.3: c64[512,1024,2048]) -> c64[1024,512,2,1024] {
  %param_0.3 = c64[512,1024,2048]{1,0,2} parameter(0)
  %bitcast.51.1 = c64[2,1024,512,1024]{3,2,1,0} bitcast(c64[512,1024,2048]{1,0,2} %param_0.3)
  ROOT %transpose.26.1 = c64[1024,512,2,1024]{3,2,1,0} transpose(c64[2,1024,512,1024]{3,2,1,0} %bitcast.51.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_convert_computation (param_0.6: f32[512,1024,2048]) -> c64[512,1024,2048] {
  %param_0.6 = f32[512,1024,2048]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c64[512,1024,2048]{2,1,0} convert(f32[512,1024,2048]{2,1,0} %param_0.6), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.7: c64[512,1024,2048]) -> c64[2048,512,1024] {
  %param_0.7 = c64[512,1024,2048]{2,1,0} parameter(0)
  ROOT %transpose.24.1 = c64[2048,512,1024]{2,1,0} transpose(c64[512,1024,2048]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation.1 (param_0.8: c64[1024,512,2048]) -> c64[2048,1024,512] {
  %param_0.8 = c64[1024,512,2048]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c64[2048,1024,512]{2,1,0} transpose(c64[1024,512,2048]{2,1,0} %param_0.8), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
}

ENTRY %main.40_spmd (param.1: f32[512,1024,2048]) -> c64[512,1024,2048] {
  %param.1 = f32[512,1024,2048]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}, metadata={op_name="arr"}
  %wrapped_convert = c64[512,1024,2048]{2,1,0} fusion(f32[512,1024,2048]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c64[512,1024,2048]{2,1,0} fft(c64[512,1024,2048]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c64[2048,512,1024]{2,1,0} fusion(c64[512,1024,2048]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %bitcast.48.0 = c64[512,1024,2048]{1,0,2} bitcast(c64[2048,512,1024]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c64[512,1024,2048]{1,0,2}), c64[512,1024,2048]{1,0,2}) all-to-all-start(c64[512,1024,2048]{1,0,2} %bitcast.48.0), channel_id=1, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c64[512,1024,2048]{1,0,2} all-to-all-done(((c64[512,1024,2048]{1,0,2}), c64[512,1024,2048]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %loop_transpose_fusion.1 = c64[1024,512,2,1024]{3,2,1,0} fusion(c64[512,1024,2048]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %bitcast.55.0 = c64[1024,512,2048]{2,1,0} bitcast(c64[1024,512,2,1024]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c64[1024,512,2048]{2,1,0} fft(c64[1024,512,2048]{2,1,0} %bitcast.55.0), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c64[2048,1024,512]{2,1,0} fusion(c64[1024,512,2048]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %bitcast.60.0 = c64[1024,512,2048]{1,0,2} bitcast(c64[2048,1024,512]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c64[1024,512,2048]{1,0,2}), c64[1024,512,2048]{1,0,2}) all-to-all-start(c64[1024,512,2048]{1,0,2} %bitcast.60.0), channel_id=2, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c64[1024,512,2048]{1,0,2} all-to-all-done(((c64[1024,512,2048]{1,0,2}), c64[1024,512,2048]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %loop_transpose_fusion = c64[512,1024,4,512]{3,2,1,0} fusion(c64[1024,512,2048]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %bitcast.67.0 = c64[512,1024,2048]{2,1,0} bitcast(c64[512,1024,4,512]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c64[512,1024,2048]{2,1,0} fft(c64[512,1024,2048]{2,1,0} %bitcast.67.0), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,2,1]<=[8]}"}) -> (tensor<2048x2048x2048xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %0 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %0 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @fft3d(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x2048x2048xf32>) -> tensor<512x1024x2048xf32>
    %2 = call @shmap_body(%1) : (tensor<512x1024x2048xf32>) -> tensor<512x1024x2048xcomplex<f32>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<512x1024x2048xcomplex<f32>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %4 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @shmap_body(%arg0: tensor<512x1024x2048xf32>) -> (tensor<512x1024x2048xcomplex<f32>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<512x1024x2048xf32>) -> tensor<512x1024x2048xcomplex<f32>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<512x2048x1024xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<512x2048x1024xcomplex<f32>>) -> tensor<1024x512x2048xcomplex<f32>>
    %3 = call @fft_0(%2) : (tensor<1024x512x2048xcomplex<f32>>) -> tensor<1024x512x2048xcomplex<f32>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<1024x512x2048xcomplex<f32>>) -> tensor<1024x2048x512xcomplex<f32>>
    %5 = stablehlo.transpose %4, dims = [2, 0, 1] : (tensor<1024x2048x512xcomplex<f32>>) -> tensor<512x1024x2048xcomplex<f32>>
    %6 = call @fft_1(%5) : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<512x1024x2048xcomplex<f32>>
    return %6 : tensor<512x1024x2048xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<512x1024x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x1024x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<512x1024x2048xf32>) -> tensor<512x1024x2048xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [2048] : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<512x1024x2048xcomplex<f32>>
    return %1 : tensor<512x1024x2048xcomplex<f32>>
  }
  func.func private @fft_0(%arg0: tensor<1024x512x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<1024x512x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [2048] : (tensor<1024x512x2048xcomplex<f32>>) -> tensor<1024x512x2048xcomplex<f32>>
    return %0 : tensor<1024x512x2048xcomplex<f32>>
  }
  func.func private @fft_1(%arg0: tensor<512x1024x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<512x1024x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [2048] : (tensor<512x1024x2048xcomplex<f32>>) -> tensor<512x1024x2048xcomplex<f32>>
    return %0 : tensor<512x1024x2048xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[2048,2048,2048]. let
    b:c64[2048,2048,2048] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[2048,2048,2048]. let
          d:c64[2048,2048,2048] = pjit[
            name=fft3d
            jaxpr={ lambda ; e:f32[2048,2048,2048]. let
                f:c64[2048,2048,2048] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:f32[512,1024,2048]. let
                      h:c64[512,1024,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f32[512,1024,2048]. let
                            j:c64[512,1024,2048] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[512,1024,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c64[512,2048,1024] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c64[1024,512,2048] = transpose[permutation=(2, 0, 1)] l
                      n:c64[1024,512,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c64[1024,512,2048]. let
                            p:c64[1024,512,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c64[1024,2048,512] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c64[512,1024,2048] = transpose[permutation=(2, 0, 1)] q
                      s:c64[512,1024,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c64[512,1024,2048]. let
                            u:c64[512,1024,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] t
                          in (u,) }
                      ] r
                    in (s,) }
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
