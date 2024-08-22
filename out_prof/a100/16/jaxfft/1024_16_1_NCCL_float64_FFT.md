# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
| Precision   | float64 |
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
| JIT Time       | 2148.6293739872053 |
| Min Time       | 321.67074900644366 |
| Max Time       | 455.82285361160757 |
| Mean Time      | 336.5306530628004  |
| Std Time       | 39.78562037871443  |
| Last Time      | 322.8569419225096  |
| Generated Code | 19.41 KB           |
| Argument Size  | 512.00 MB          |
| Output Size    | 1.00 GB            |
| Temporary Size | 2.00 GB            |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 455.823 |
| Run 1       | 325.075 |
| Run 2       | 322.111 |
| Run 3       | 323.312 |
| Run 4       | 325.535 |
| Run 5       | 321.876 |
| Run 6       | 324.665 |
| Run 7       | 321.671 |
| Run 8       | 322.381 |
| Run 9       | 322.857 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f64[1024,64,1024]{2,1,0})->c128[1024,64,1024]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="6f553b06cd5c5e094bb672613d83b6cd"}

%fused_transpose (param_0.1: c128[1024,64,1024]) -> c128[64,1024,16,64] {
  %param_0.1 = c128[1024,64,1024]{1,0,2} parameter(0)
  %bitcast.36.1 = c128[16,64,1024,64]{3,2,1,0} bitcast(c128[1024,64,1024]{1,0,2} %param_0.1)
  ROOT %transpose.19.1 = c128[64,1024,16,64]{3,2,1,0} transpose(c128[16,64,1024,64]{3,2,1,0} %bitcast.36.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_convert_computation (param_0.3: f64[1024,64,1024]) -> c128[1024,64,1024] {
  %param_0.3 = f64[1024,64,1024]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c128[1024,64,1024]{2,1,0} convert(f64[1024,64,1024]{2,1,0} %param_0.3), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.4: c128[1024,64,1024]) -> c128[1024,1024,64] {
  %param_0.4 = c128[1024,64,1024]{2,1,0} parameter(0)
  ROOT %transpose.17.1 = c128[1024,1024,64]{2,1,0} transpose(c128[1024,64,1024]{2,1,0} %param_0.4), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation.1 (param_0.5: c128[64,1024,1024]) -> c128[1024,64,1024] {
  %param_0.5 = c128[64,1024,1024]{2,1,0} parameter(0)
  ROOT %transpose.21.1 = c128[1024,64,1024]{2,1,0} transpose(c128[64,1024,1024]{2,1,0} %param_0.5), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

ENTRY %main.35_spmd (param.1: f64[1024,64,1024]) -> c128[1024,64,1024] {
  %param.1 = f64[1024,64,1024]{2,1,0} parameter(0), sharding={devices=[1,16,1]<=[16]}, metadata={op_name="arr"}
  %wrapped_convert = c128[1024,64,1024]{2,1,0} fusion(f64[1024,64,1024]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex128 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c128[1024,64,1024]{2,1,0} fft(c128[1024,64,1024]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c128[1024,1024,64]{2,1,0} fusion(c128[1024,64,1024]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %bitcast.33.0 = c128[1024,64,1024]{1,0,2} bitcast(c128[1024,1024,64]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[1024,64,1024]{1,0,2}), c128[1024,64,1024]{1,0,2}) all-to-all-start(c128[1024,64,1024]{1,0,2} %bitcast.33.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[1024,64,1024]{1,0,2} all-to-all-done(((c128[1024,64,1024]{1,0,2}), c128[1024,64,1024]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %loop_transpose_fusion = c128[64,1024,16,64]{3,2,1,0} fusion(c128[1024,64,1024]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %bitcast.40.0 = c128[64,1024,1024]{2,1,0} bitcast(c128[64,1024,16,64]{3,2,1,0} %loop_transpose_fusion)
  %fft.16.0 = c128[64,1024,1024]{2,1,0} fft(c128[64,1024,1024]{2,1,0} %bitcast.40.0), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c128[1024,64,1024]{2,1,0} fusion(c128[64,1024,1024]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  ROOT %fft.17.0 = c128[1024,64,1024]{2,1,0} fft(c128[1024,64,1024]{2,1,0} %wrapped_transpose.1), fft_type=FFT, fft_length={1024}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(1024,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,16,1]<=[16]}"}) -> (tensor<1024x1024x1024xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @do_fft(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %0 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @fft3d(%arg0: tensor<1024x1024x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,16,1]<=[16]}"} : (tensor<1024x1024x1024xf64>) -> tensor<1024x1024x1024xf64>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024x1024xf64>) -> tensor<1024x64x1024xf64>
    %2 = call @shmap_body(%1) : (tensor<1024x64x1024xf64>) -> tensor<1024x64x1024xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,16,1]<=[16]}"} : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x1024x1024xcomplex<f64>>
    return %4 : tensor<1024x1024x1024xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<1024x64x1024xf64>) -> (tensor<1024x64x1024xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<1024x64x1024xf64>) -> tensor<1024x64x1024xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi64>, split_count = 16 : i64, split_dimension = 2 : i64}> : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x1024x64xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [2, 0, 1] : (tensor<1024x1024x64xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    %4 = stablehlo.transpose %3, dims = [2, 0, 1] : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    %5 = call @fft_1(%4) : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    return %5 : tensor<1024x64x1024xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<1024x64x1024xf64> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<1024x64x1024xf64>) -> tensor<1024x64x1024xcomplex<f64>>
    %1 = stablehlo.fft %0, type =  FFT, length = [1024] : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    return %1 : tensor<1024x64x1024xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<64x1024x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [1024] : (tensor<64x1024x1024xcomplex<f64>>) -> tensor<64x1024x1024xcomplex<f64>>
    return %0 : tensor<64x1024x1024xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<1024x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<1024x64x1024xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [1024] : (tensor<1024x64x1024xcomplex<f64>>) -> tensor<1024x64x1024xcomplex<f64>>
    return %0 : tensor<1024x64x1024xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:f64[1024,1024,1024]. let
    b:c128[1024,1024,1024] = pjit[
      name=do_fft
      jaxpr={ lambda ; c:f64[1024,1024,1024]. let
          d:c128[1024,1024,1024] = pjit[
            name=fft3d
            jaxpr={ lambda ; e:f64[1024,1024,1024]. let
                f:c128[1024,1024,1024] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:f64[1024,64,1024]. let
                      h:c128[1024,64,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f64[1024,64,1024]. let
                            j:c128[1024,64,1024] = convert_element_type[
                              new_dtype=complex128
                              weak_type=False
                            ] i
                            k:c128[1024,64,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c128[1024,1024,64] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c128[64,1024,1024] = transpose[permutation=(2, 0, 1)] l
                      n:c128[64,1024,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c128[64,1024,1024]. let
                            p:c128[64,1024,1024] = fft[
                              fft_lengths=(1024,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c128[64,1024,1024] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c128[1024,64,1024] = transpose[permutation=(2, 0, 1)] q
                      s:c128[1024,64,1024] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c128[1024,64,1024]. let
                            u:c128[1024,64,1024] = fft[
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
