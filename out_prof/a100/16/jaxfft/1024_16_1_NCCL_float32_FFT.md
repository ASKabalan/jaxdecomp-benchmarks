# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
| Precision   | float32 |
| X           | 1024    |
| Y           | 1024    |
| Z           | 1024    |
| PX          | 16      |
| PY          | 1       |
| Backend     | NCCL    |
| Nodes       | 2       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 2017.6418659975752 |
| Min Time       | 160.54769897460938 |
| Max Time       | 275.78363037109375 |
| Mean Time      | 172.66360473632812 |
| Std Time       | 34.37966537475586  |
| Last Time      | 161.19992065429688 |
| Generated Code | 14.91 KB           |
| Argument Size  | 256.00 MB          |
| Output Size    | 512.00 MB          |
| Temporary Size | 1.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 275.784 |
| Run 1       | 160.987 |
| Run 2       | 161.506 |
| Run 3       | 162.996 |
| Run 4       | 160.926 |
| Run 5       | 161.242 |
| Run 6       | 160.842 |
| Run 7       | 160.548 |
| Run 8       | 160.607 |
| Run 9       | 161.2   |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[1024,64,1024]{2,1,0})->c64[1024,64,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="ce5efd36b61139ae811d485bab6f5277"}

%fused_transpose (param_0.1: c64[1024,64,1024]) -> c64[64,1024,16,64] {
  %param_0.1 = c64[1024,64,1024]{1,0,2} parameter(0)
  %bitcast.36.1 = c64[16,64,1024,64]{3,2,1,0} bitcast(c64[1024,64,1024]{1,0,2} %param_0.1)
  ROOT %transpose.19.1 = c64[64,1024,16,64]{3,2,1,0} transpose(c64[16,64,1024,64]{3,2,1,0} %bitcast.36.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_convert_computation (param_0.3: f32[1024,64,1024]) -> c64[1024,64,1024] {
  %param_0.3 = f32[1024,64,1024]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c64[1024,64,1024]{2,1,0} convert(f32[1024,64,1024]{2,1,0} %param_0.3), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.4: c64[1024,64,1024]) -> c64[1024,1024,64] {
  %param_0.4 = c64[1024,64,1024]{2,1,0} parameter(0)
  ROOT %transpose.17.1 = c64[1024,1024,64]{2,1,0} transpose(c64[1024,64,1024]{2,1,0} %param_0.4), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation.1 (param_0.5: c64[64,1024,1024]) -> c64[1024,64,1024] {
  %param_0.5 = c64[64,1024,1024]{2,1,0} parameter(0)
  ROOT %transpose.21.1 = c64[1024,64,1024]{2,1,0} transpose(c64[64,1024,1024]{2,1,0} %param_0.5), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

ENTRY %main.35_spmd (param.1: f32[1024,64,1024]) -> c64[1024,64,1024] {
  %param.1 = f32[1024,64,1024]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="arr"}
  %wrapped_convert = c64[1024,64,1024]{2,1,0} fusion(f32[1024,64,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c64[1024,64,1024]{2,1,0} fft(c64[1024,64,1024]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c64[1024,1024,64]{2,1,0} fusion(c64[1024,64,1024]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %bitcast.33.0 = c64[1024,64,1024]{1,0,2} bitcast(c64[1024,1024,64]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c64[1024,64,1024]{1,0,2}), c64[1024,64,1024]{1,0,2}) all-to-all-start(c64[1024,64,1024]{1,0,2} %bitcast.33.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c64[1024,64,1024]{1,0,2} all-to-all-done(((c64[1024,64,1024]{1,0,2}), c64[1024,64,1024]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %loop_transpose_fusion = c64[64,1024,16,64]{3,2,1,0} fusion(c64[1024,64,1024]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %bitcast.40.0 = c64[64,1024,1024]{2,1,0} bitcast(c64[64,1024,16,64]{3,2,1,0} %loop_transpose_fusion)
  %fft.16.0 = c64[64,1024,1024]{2,1,0} fft(c64[64,1024,1024]{2,1,0} %bitcast.40.0), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c64[1024,64,1024]{2,1,0} fusion(c64[64,1024,1024]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  ROOT %fft.17.0 = c64[1024,64,1024]{2,1,0} fft(c64[1024,64,1024]{2,1,0} %wrapped_transpose.1), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<1024x1024x1024xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xcomplex<f32>>
    return %0 : tensor<1024x1024x1024xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<1024x1024x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xcomplex<f32>>
    return %0 : tensor<1024x1024x1024xcomplex<f32>>
  }
  func.func private @fft3d(%arg0: tensor<1024x1024x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,16,1]<=[16]}"} : (tensor<1024x1024x1024xf32>) -> tensor<1024x1024x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024x1024xf32>) -> tensor<1024x64x1024xf32>
    %2 = call @shmap_body(%1) : (tensor<1024x64x1024xf32>) -> tensor<1024x64x1024xcomplex<f32>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x64x1024xcomplex<f32>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,16,1]<=[16]}"} : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x1024x1024xcomplex<f32>>
    return %4 : tensor<1024x1024x1024xcomplex<f32>>
  }
  func.func private @shmap_body(%arg0: tensor<1024x64x1024xf32>) -> (tensor<1024x64x1024xcomplex<f32>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<1024x64x1024xf32>) -> tensor<1024x64x1024xcomplex<f32>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi64>, split_count = 16 : i64, split_dimension = 2 : i64}> : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x1024x64xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<1024x1024x64xcomplex<f32>>) -> tensor<64x1024x1024xcomplex<f32>>
    %3 = call @fft_0(%2) : (tensor<64x1024x1024xcomplex<f32>>) -> tensor<64x1024x1024xcomplex<f32>>
    %4 = stablehlo.transpose %3, dims = [2, 0, 1] : (tensor<64x1024x1024xcomplex<f32>>) -> tensor<1024x64x1024xcomplex<f32>>
    %5 = call @fft_1(%4) : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x64x1024xcomplex<f32>>
    return %5 : tensor<1024x64x1024xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<1024x64x1024xf32> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1024x64x1024xf32>) -> tensor<1024x64x1024xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [1024] : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x64x1024xcomplex<f32>>
    return %1 : tensor<1024x64x1024xcomplex<f32>>
  }
  func.func private @fft_0(%arg0: tensor<64x1024x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<64x1024x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [1024] : (tensor<64x1024x1024xcomplex<f32>>) -> tensor<64x1024x1024xcomplex<f32>>
    return %0 : tensor<64x1024x1024xcomplex<f32>>
  }
  func.func private @fft_1(%arg0: tensor<1024x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x1024xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [1024] : (tensor<1024x64x1024xcomplex<f32>>) -> tensor<1024x64x1024xcomplex<f32>>
    return %0 : tensor<1024x64x1024xcomplex<f32>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f32[1024,1024,1024]. let
    b:c64[1024,1024,1024] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f32[1024,1024,1024]. let
          d:c64[1024,1024,1024] = pjit[
            name=fft3d
            jaxpr={ lambda ; e:f32[1024,1024,1024]. let
                f:c64[1024,1024,1024] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:f32[1024,64,1024]. let
                      h:c64[1024,64,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f32[1024,64,1024]. let
                            j:c64[1024,64,1024] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[1024,64,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c64[1024,1024,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c64[64,1024,1024] = transpose[permutation=(2, 0, 1)] l
                      n:c64[64,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c64[64,1024,1024]. let
                            p:c64[64,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c64[64,1024,1024] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c64[1024,64,1024] = transpose[permutation=(2, 0, 1)] q
                      s:c64[1024,64,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c64[1024,64,1024]. let
                            u:c64[1024,64,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] t
                          in (u,) }
                      ] r
                    in (s,) }
                  mesh=Mesh('z': 1, 'y': 16)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
