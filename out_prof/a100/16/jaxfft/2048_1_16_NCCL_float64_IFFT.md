# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 2048    |
| Y           | 2048    |
| Z           | 2048    |
| PX          | 1       |
| PY          | 16      |
| Backend     | NCCL    |
| Nodes       | 2       |
---
## Profiling Data
| Parameter      | Value              |
|----------------|--------------------|
| JIT Time       | 2850.206745089963  |
| Min Time       | 2621.42280642729   |
| Max Time       | 2627.8562919906108 |
| Mean Time      | 2624.49056432597   |
| Std Time       | 2.269627328695906  |
| Last Time      | 2622.463675266772  |
| Generated Code | 16.97 KB           |
| Argument Size  | 8.00 GB            |
| Output Size    | 8.00 GB            |
| Temporary Size | 16.00 GB           |
---
## Iteration Runs
| Iteration   |    Time |
|-------------|---------|
| Run 0       | 2626.51 |
| Run 1       | 2627.19 |
| Run 2       | 2627.86 |
| Run 3       | 2626.72 |
| Run 4       | 2621.79 |
| Run 5       | 2624.29 |
| Run 6       | 2621.42 |
| Run 7       | 2623.76 |
| Run 8       | 2622.91 |
| Run 9       | 2622.46 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[128,2048,2048]{2,1,0})->c128[128,2048,2048]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=16, frontend_attributes={fingerprint_before_lhs="4f0b158bd9d6b8288169c67c4314b51b"}

%fused_transpose (param_0.1: c128[128,2048,2048]) -> c128[2048,128,16,128] {
  %param_0.1 = c128[128,2048,2048]{1,0,2} parameter(0)
  %bitcast.34.1 = c128[16,128,128,2048]{3,2,1,0} bitcast(c128[128,2048,2048]{1,0,2} %param_0.1)
  ROOT %transpose.22.1 = c128[2048,128,16,128]{3,2,1,0} transpose(c128[16,128,128,2048]{3,2,1,0} %bitcast.34.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation (param_0.3: c128[128,2048,2048]) -> c128[2048,128,2048] {
  %param_0.3 = c128[128,2048,2048]{2,1,0} parameter(0)
  ROOT %transpose.20.1 = c128[2048,128,2048]{2,1,0} transpose(c128[128,2048,2048]{2,1,0} %param_0.3), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}

%wrapped_transpose_computation.1 (param_0.4: c128[2048,128,2048]) -> c128[128,2048,2048] {
  %param_0.4 = c128[2048,128,2048]{2,1,0} parameter(0)
  ROOT %transpose.23.1 = c128[128,2048,2048]{2,1,0} transpose(c128[2048,128,2048]{2,1,0} %param_0.4), dimensions={1,2,0}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

ENTRY %main.34_spmd (param.1: c128[128,2048,2048]) -> c128[128,2048,2048] {
  %param.1 = c128[128,2048,2048]{2,1,0} parameter(0), sharding={devices=[16,1,1]<=[16]}, metadata={op_name="arr"}
  %fft.15.0 = c128[128,2048,2048]{2,1,0} fft(c128[128,2048,2048]{2,1,0} %param.1), fft_type=IFFT, fft_length={2048}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[2048,128,2048]{2,1,0} fusion(c128[128,2048,2048]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %bitcast.31.0 = c128[128,2048,2048]{1,0,2} bitcast(c128[2048,128,2048]{2,1,0} %wrapped_transpose)
  %all-to-all-start = ((c128[128,2048,2048]{1,0,2}), c128[128,2048,2048]{1,0,2}) all-to-all-start(c128[128,2048,2048]{1,0,2} %bitcast.31.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[128,2048,2048]{1,0,2} all-to-all-done(((c128[128,2048,2048]{1,0,2}), c128[128,2048,2048]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %loop_transpose_fusion = c128[2048,128,16,128]{3,2,1,0} fusion(c128[128,2048,2048]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'z\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %bitcast.38.0 = c128[2048,128,2048]{2,1,0} bitcast(c128[2048,128,16,128]{3,2,1,0} %loop_transpose_fusion)
  %fft.16.0 = c128[2048,128,2048]{2,1,0} fft(c128[2048,128,2048]{2,1,0} %bitcast.38.0), fft_type=IFFT, fft_length={2048}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[128,2048,2048]{2,1,0} fusion(c128[2048,128,2048]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  ROOT %fft.17.0 = c128[128,2048,2048]{2,1,0} fft(c128[128,2048,2048]{2,1,0} %wrapped_transpose.1), fft_type=IFFT, fft_length={2048}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(2048,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 16 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2048x2048x2048xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[16,1,1]<=[16]}"}) -> (tensor<2048x2048x2048xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<2048x2048x2048xcomplex<f64>>) -> tensor<2048x2048x2048xcomplex<f64>>
    return %0 : tensor<2048x2048x2048xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<2048x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<2048x2048x2048xcomplex<f64>>) -> tensor<2048x2048x2048xcomplex<f64>>
    return %0 : tensor<2048x2048x2048xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<2048x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<2048x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[16,1,1]<=[16]}"} : (tensor<2048x2048x2048xcomplex<f64>>) -> tensor<2048x2048x2048xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[16,1,1]<=[16]}"} : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<2048x2048x2048xcomplex<f64>>
    return %4 : tensor<2048x2048x2048xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<128x2048x2048xcomplex<f64>>) -> (tensor<128x2048x2048xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    %1 = "stablehlo.all_to_all"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi64>, split_count = 16 : i64, split_dimension = 2 : i64}> : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<2048x2048x128xcomplex<f64>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0] : (tensor<2048x2048x128xcomplex<f64>>) -> tensor<2048x128x2048xcomplex<f64>>
    %3 = call @fft_0(%2) : (tensor<2048x128x2048xcomplex<f64>>) -> tensor<2048x128x2048xcomplex<f64>>
    %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<2048x128x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    %5 = call @fft_1(%4) : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    return %5 : tensor<128x2048x2048xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<128x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [2048] : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    return %0 : tensor<128x2048x2048xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<2048x128x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<2048x128x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [2048] : (tensor<2048x128x2048xcomplex<f64>>) -> tensor<2048x128x2048xcomplex<f64>>
    return %0 : tensor<2048x128x2048xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<128x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<128x2048x2048xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [2048] : (tensor<128x2048x2048xcomplex<f64>>) -> tensor<128x2048x2048xcomplex<f64>>
    return %0 : tensor<128x2048x2048xcomplex<f64>>
  }
}

```

---
## JAXPR
```haskel
{ lambda ; a:c128[2048,2048,2048]. let
    b:c128[2048,2048,2048] = pjit[
      name=do_ifft
      jaxpr={ lambda ; c:c128[2048,2048,2048]. let
          d:c128[2048,2048,2048] = pjit[
            name=ifft3d
            jaxpr={ lambda ; e:c128[2048,2048,2048]. let
                f:c128[2048,2048,2048] = shard_map[
                  auto=frozenset()
                  check_rep=True
                  in_names=({0: ('z',), 1: ('y',)},)
                  jaxpr={ lambda ; g:c128[128,2048,2048]. let
                      h:c128[128,2048,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[128,2048,2048]. let
                            j:c128[128,2048,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[2048,2048,128] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[2048,128,2048] = transpose[permutation=(1, 2, 0)] k
                      m:c128[2048,128,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[2048,128,2048]. let
                            o:c128[2048,128,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[2048,128,2048] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[128,2048,2048] = transpose[permutation=(1, 2, 0)] p
                      r:c128[128,2048,2048] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[128,2048,2048]. let
                            t:c128[128,2048,2048] = fft[
                              fft_lengths=(2048,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 16, 'y': 1)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
