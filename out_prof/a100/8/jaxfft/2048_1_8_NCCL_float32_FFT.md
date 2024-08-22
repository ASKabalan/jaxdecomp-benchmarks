# Reporting for FFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | FFT     |
| Precision   | float32 |
| X           | 2048    |
| Y           | 2048    |
| Z           | 2048    |
| PX          | 1       |
| PY          | 8       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 10760.330134999094 |
| Min Time       | 97.26461029052734  |
| Max Time       | 212.88363647460938 |
| Mean Time      | 108.93145751953125 |
| Std Time       | 34.65079879760742  |
| Last Time      | 97.35255432128906  |
| Generated Code | 14.91 KB           |
| Argument Size  | 4.00 GB            |
| Output Size    | 8.00 GB            |
| Temporary Size | 16.00 GB           |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 212.884  |
| Run 1       |  97.3303 |
| Run 2       |  97.4565 |
| Run 3       |  97.4171 |
| Run 4       |  97.3224 |
| Run 5       |  97.5019 |
| Run 6       |  97.2646 |
| Run 7       |  97.4457 |
| Run 8       |  97.3398 |
| Run 9       |  97.3526 |
---
## Compiled Code
```hlo
HloModule jit_do_fft, is_scheduled=true, entry_computation_layout={(f32[256,2048,2048]{2,1,0})->c64[256,2048,2048]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="d5dc178d15de3bd56184dae0c6b7fb23"}

%fused_transpose (param_0.1: c64[2048,256,2048]) -> c64[256,2048,8,256] {
  %param_0.1 = c64[2048,256,2048]{1,0,2} parameter(0)
  %bitcast.39.1 = c64[8,256,2048,256]{3,2,1,0} bitcast(c64[2048,256,2048]{1,0,2} %param_0.1)
  ROOT %transpose.21.1 = c64[256,2048,8,256]{3,2,1,0} transpose(c64[8,256,2048,256]{3,2,1,0} %bitcast.39.1), dimensions={1,2,0,3}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
}

%wrapped_convert_computation (param_0.3: f32[256,2048,2048]) -> c64[256,2048,2048] {
  %param_0.3 = f32[256,2048,2048]{2,1,0} parameter(0)
  ROOT %convert.7.1 = c64[256,2048,2048]{2,1,0} convert(f32[256,2048,2048]{2,1,0} %param_0.3), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
}

%wrapped_transpose_computation (param_0.4: c64[256,2048,2048]) -> c64[2048,256,2048] {
  %param_0.4 = c64[256,2048,2048]{2,1,0} parameter(0)
  ROOT %transpose.18.1 = c64[2048,256,2048]{2,1,0} transpose(c64[256,2048,2048]{2,1,0} %param_0.4), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
}

%wrapped_transpose_computation.1 (param_0.5: c64[2048,256,2048]) -> c64[2048,2048,256] {
  %param_0.5 = c64[2048,256,2048]{2,1,0} parameter(0)
  ROOT %transpose.19.1 = c64[2048,2048,256]{2,1,0} transpose(c64[2048,256,2048]{2,1,0} %param_0.5), dimensions={2,0,1}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
}

ENTRY %main.35_spmd (param.1: f32[256,2048,2048]) -> c64[256,2048,2048] {
  %param.1 = f32[256,2048,2048]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="arr"}
  %wrapped_convert = c64[256,2048,2048]{2,1,0} fusion(f32[256,2048,2048]{2,1,0} %param.1), kind=kLoop, calls=%wrapped_convert_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/convert_element_type[new_dtype=complex64 weak_type=False sharding=None]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %fft.15.0 = c64[256,2048,2048]{2,1,0} fft(c64[256,2048,2048]{2,1,0} %wrapped_convert), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=63}
  %wrapped_transpose = c64[2048,256,2048]{2,1,0} fusion(c64[256,2048,2048]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/transpose[permutation=(2, 0, 1)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=37}
  %fft.16.0 = c64[2048,256,2048]{2,1,0} fft(c64[2048,256,2048]{2,1,0} %wrapped_transpose), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %wrapped_transpose.1 = c64[2048,2048,256]{2,1,0} fusion(c64[2048,256,2048]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=65}
  %bitcast.36.0 = c64[2048,256,2048]{1,0,2} bitcast(c64[2048,2048,256]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start = ((c64[2048,256,2048]{1,0,2}), c64[2048,256,2048]{1,0,2}) all-to-all-start(c64[2048,256,2048]{1,0,2} %bitcast.36.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={2}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c64[2048,256,2048]{1,0,2} all-to-all-done(((c64[2048,256,2048]{1,0,2}), c64[2048,256,2048]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %loop_transpose_fusion = c64[256,2048,8,256]{3,2,1,0} fusion(c64[2048,256,2048]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=1 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=41}
  %bitcast.43.0 = c64[256,2048,2048]{2,1,0} bitcast(c64[256,2048,8,256]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c64[256,2048,2048]{2,1,0} fft(c64[256,2048,2048]{2,1,0} %bitcast.43.0), fft_type=FFT, fft_length={2048}, metadata={op_name="jit(do_fft)/jit(main)/jit(do_fft)/jit(fft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.FFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=67}
}


```

---
## Lowered Code
```hlo
module @jit_do_fft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[8,1,1]<=[8]}"}) -> (tensor<2048x2048x2048xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_fft(%arg0) : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %0 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @do_fft(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = call @fft3d(%arg0) : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %0 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @fft3d(%arg0: tensor<2048x2048x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<2048x2048x2048xf32>) -> tensor<2048x2048x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x2048x2048xf32>) -> tensor<256x2048x2048xf32>
    %2 = call @shmap_body(%1) : (tensor<256x2048x2048xf32>) -> tensor<256x2048x2048xcomplex<f32>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<256x2048x2048xcomplex<f32>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,1,1]<=[8]}"} : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<2048x2048x2048xcomplex<f32>>
    return %4 : tensor<2048x2048x2048xcomplex<f32>>
  }
  func.func private @shmap_body(%arg0: tensor<256x2048x2048xf32>) -> (tensor<256x2048x2048xcomplex<f32>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<256x2048x2048xf32>) -> tensor<256x2048x2048xcomplex<f32>>
    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<2048x256x2048xcomplex<f32>>
    %2 = call @fft_0(%1) : (tensor<2048x256x2048xcomplex<f32>>) -> tensor<2048x256x2048xcomplex<f32>>
    %3 = "stablehlo.all_to_all"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 1 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, split_count = 8 : i64, split_dimension = 2 : i64}> : (tensor<2048x256x2048xcomplex<f32>>) -> tensor<2048x2048x256xcomplex<f32>>
    %4 = stablehlo.transpose %3, dims = [2, 0, 1] : (tensor<2048x2048x256xcomplex<f32>>) -> tensor<256x2048x2048xcomplex<f32>>
    %5 = call @fft_1(%4) : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<256x2048x2048xcomplex<f32>>
    return %5 : tensor<256x2048x2048xcomplex<f32>>
  }
  func.func private @fft(%arg0: tensor<256x2048x2048xf32> {mhlo.layout_mode = "default"}) -> (tensor<256x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.convert %arg0 : (tensor<256x2048x2048xf32>) -> tensor<256x2048x2048xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [2048] : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<256x2048x2048xcomplex<f32>>
    return %1 : tensor<256x2048x2048xcomplex<f32>>
  }
  func.func private @fft_0(%arg0: tensor<2048x256x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<2048x256x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [2048] : (tensor<2048x256x2048xcomplex<f32>>) -> tensor<2048x256x2048xcomplex<f32>>
    return %0 : tensor<2048x256x2048xcomplex<f32>>
  }
  func.func private @fft_1(%arg0: tensor<256x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) -> (tensor<256x2048x2048xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  FFT, length = [2048] : (tensor<256x2048x2048xcomplex<f32>>) -> tensor<256x2048x2048xcomplex<f32>>
    return %0 : tensor<256x2048x2048xcomplex<f32>>
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
                  jaxpr={ lambda ; g:f32[256,2048,2048]. let
                      h:c64[256,2048,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:f32[256,2048,2048]. let
                            j:c64[256,2048,2048] = convert_element_type[
                              new_dtype=complex64
                              weak_type=False
                            ] i
                            k:c64[256,2048,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] j
                          in (k,) }
                      ] g
                      l:c64[256,2048,2048] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] h
                      m:c64[2048,256,2048] = transpose[permutation=(2, 0, 1)] l
                      n:c64[2048,256,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; o:c64[2048,256,2048]. let
                            p:c64[2048,256,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] o
                          in (p,) }
                      ] m
                      q:c64[2048,2048,256] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=1
                        split_axis=2
                        tiled=True
                      ] n
                      r:c64[256,2048,2048] = transpose[permutation=(2, 0, 1)] q
                      s:c64[256,2048,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; t:c64[256,2048,2048]. let
                            u:c64[256,2048,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.FFT
                            ] t
                          in (u,) }
                      ] r
                    in (s,) }
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
