# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
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
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 7269.384495997656  |
| Min Time       | 0.8030069998312683 |
| Max Time       | 142.88905600096768 |
| Mean Time      | 15.048817800106917 |
| Std Time       | 42.61341964135292  |
| Last Time      | 0.8030069998312683 |
| Generated Code | 21.34 KB           |
| Argument Size  | 16.00 MB           |
| Output Size    | 32.00 MB           |
| Temporary Size | 64.00 MB           |
---
## Iteration Runs
| Iteration   |       Time |
|-------------|------------|
| Run 0       | 142.889    |
| Run 1       |   0.850583 |
| Run 2       |   0.836917 |
| Run 3       |   0.857936 |
| Run 4       |   0.841023 |
| Run 5       |   0.819289 |
| Run 6       |   0.892098 |
| Run 7       |   0.87075  |
| Run 8       |   0.827519 |
| Run 9       |   0.803007 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[64,128,256]{2,1,0})->c128[64,128,256]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="ac82ab573eef6406f0c5cf79a15b4508"}

%fused_transpose (param_0.1: c128[128,64,256]) -> c128[64,128,4,64] {
  %param_0.1 = c128[128,64,256]{1,0,2} parameter(0)
  %bitcast.63.1 = c128[4,64,128,64]{3,2,1,0} bitcast(c128[128,64,256]{1,0,2} %param_0.1)
  ROOT %transpose.29.1 = c128[64,128,4,64]{3,2,1,0} transpose(c128[4,64,128,64]{3,2,1,0} %bitcast.63.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

%fused_transpose.1 (param_0.3: c128[64,128,256]) -> c128[128,64,2,128] {
  %param_0.3 = c128[64,128,256]{1,0,2} parameter(0)
  %bitcast.51.1 = c128[2,128,64,128]{3,2,1,0} bitcast(c128[64,128,256]{1,0,2} %param_0.3)
  ROOT %transpose.26.1 = c128[128,64,2,128]{3,2,1,0} transpose(c128[2,128,64,128]{3,2,1,0} %bitcast.51.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_convert_computation (param_0.6: f64[64,128,256]) -> c128[64,128,256] {
  %param_0.6 = f64[64,128,256]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c128[64,128,256]{2,1,0} convert(f64[64,128,256]{2,1,0} %param_0.6), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.7: c128[64,128,256]) -> c128[256,64,128] {
  %param_0.7 = c128[64,128,256]{2,1,0} parameter(0)
  ROOT %transpose.24.1 = c128[256,64,128]{2,1,0} transpose(c128[64,128,256]{2,1,0} %param_0.7), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation.1 (param_0.8: c128[128,64,256]) -> c128[256,128,64] {
  %param_0.8 = c128[128,64,256]{2,1,0} parameter(0)
  ROOT %transpose.28.1 = c128[256,128,64]{2,1,0} transpose(c128[128,64,256]{2,1,0} %param_0.8), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
}

ENTRY %main.40_spmd (param.1: f64[64,128,256]) -> c128[64,128,256] {
  %param.1 = f64[64,128,256]{2,1,0} parameter(0), sharding={devices=[4,2,1]<=[8]}, metadata={op_name="arr"}
  %wrapped_convert = c128[64,128,256]{2,1,0} fusion(f64[64,128,256]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c128[64,128,256]{2,1,0} fft(c128[64,128,256]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c128[256,64,128]{2,1,0} fusion(c128[64,128,256]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %bitcast.48.0 = c128[64,128,256]{1,0,2} bitcast(c128[256,64,128]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[64,128,256]{1,0,2}), c128[64,128,256]{1,0,2}) all-to-all-start(c128[64,128,256]{1,0,2} %bitcast.48.0), channel_id=1, replica_groups={{0,1},{2,3},{4,5},{6,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[64,128,256]{1,0,2} all-to-all-done(((c128[64,128,256]{1,0,2}), c128[64,128,256]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %loop_transpose_fusion.1 = c128[128,64,2,128]{3,2,1,0} fusion(c128[64,128,256]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %bitcast.55.0 = c128[128,64,256]{2,1,0} bitcast(c128[128,64,2,128]{3,2,1,0} %loop_transpose_fusion.1)
  %fft.16.0 = c128[128,64,256]{2,1,0} fft(c128[128,64,256]{2,1,0} %bitcast.55.0), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c128[256,128,64]{2,1,0} fusion(c128[128,64,256]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %bitcast.60.0 = c128[128,64,256]{1,0,2} bitcast(c128[256,128,64]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start.1 = ((c128[128,64,256]{1,0,2}), c128[128,64,256]{1,0,2}) all-to-all-start(c128[128,64,256]{1,0,2} %bitcast.60.0), channel_id=2, replica_groups={{0,2,4,6},{1,3,5,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done.1 = c128[128,64,256]{1,0,2} all-to-all-done(((c128[128,64,256]{1,0,2}), c128[128,64,256]{1,0,2}) %all-to-all-start.1), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %loop_transpose_fusion = c128[64,128,4,64]{3,2,1,0} fusion(c128[128,64,256]{1,0,2} %all-to-all-done.1), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %bitcast.67.0 = c128[64,128,256]{2,1,0} bitcast(c128[64,128,4,64]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c128[64,128,256]{2,1,0} fft(c128[64,128,256]{2,1,0} %bitcast.67.0), fft_type=FFT, fft_length={256}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256x256xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[4,2,1]<=[8]}"}) -> (tensor<256x256x256xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<256x256x256xf64>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<256x256x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<256x256x256xf64>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @fft3d(%arg0: tensor<256x256x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<256x256x256xf64>) -> tensor<256x256x256xf64>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x256x256xf64>) -> tensor<64x128x256xf64>
    %2 = call @shmap_body(%1) : (tensor<64x128x256xf64>) -> tensor<64x128x256xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2,1]<=[8]}"} : (tensor<64x128x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %4 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<64x128x256xf64>) -> (tensor<64x128x256xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<64x128x256xf64>) -> tensor<64x128x256xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, split_count = 2 : i64, split_dimension = 2 : i64}> : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x256x128xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<64x256x128xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<128x64x256xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    %4 = "stablehlo.all_to_all"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 2 : i64}> : (tensor<128x64x256xcomplex<f64>>) -> tensor<128x256x64xcomplex<f64>>
    %5 = stablehlo.transpose %4, dims = [2, 0, 1] : (tensor<128x256x64xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    %6 = call @fft_1(%5) : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %6 : tensor<64x128x256xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<64x128x256xf64> {mhlo.layout_mode = "default"}) -> (tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<64x128x256xf64>) -> tensor<64x128x256xcomplex<f64>>
    %1 = stablehlo.fft %0, type =  FFT, length = [256] : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %1 : tensor<64x128x256xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<128x64x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x64x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [256] : (tensor<128x64x256xcomplex<f64>>) -> tensor<128x64x256xcomplex<f64>>
    return %0 : tensor<128x64x256xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x128x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [256] : (tensor<64x128x256xcomplex<f64>>) -> tensor<64x128x256xcomplex<f64>>
    return %0 : tensor<64x128x256xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[256,256,256]. let
    b:c128[256,256,256] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[256,256,256]. let
          d:c128[256,256,256] = pjit[
            name=fft3d
            jaxpr={ lambda ; e:f64[256,256,256]. let
                f:c128[256,256,256] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:f64[64,128,256]. let
                      h:c128[64,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f64[64,128,256]. let
                            j:c128[64,128,256] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[64,128,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c128[64,256,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c128[128,64,256] = transpose[permutation=(2, 0, 1)] l
                      n:c128[128,64,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c128[128,64,256]. let
                            p:c128[128,64,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c128[128,256,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c128[64,128,256] = transpose[permutation=(2, 0, 1)] q
                      s:c128[64,128,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c128[64,128,256]. let
                            u:c128[64,128,256] = fft[
                              fft_lengths=(256,)
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
