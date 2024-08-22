# Reporting for IFFT
## Parameters
| Parameter   | Value   |
|-------------|---------|
| Function    | IFFT    |
| Precision   | float64 |
| X           | 256     |
| Y           | 256     |
| Z           | 256     |
| PX          | 8       |
| PY          | 1       |
| Backend     | NCCL    |
| Nodes       | 1       |
---
## Profiling Data
| Parameter      | Value               |
|----------------|---------------------|
| JIT Time       | 239.14791099741706  |
| Min Time       | 0.7995358755579218  |
| Max Time       | 1.2435178750820342  |
| Mean Time      | 0.8559135251289263  |
| Std Time       | 0.12958702855709325 |
| Last Time      | 0.7995358755579218  |
| Generated Code | 16.84 KB            |
| Argument Size  | 32.00 MB            |
| Output Size    | 32.00 MB            |
| Temporary Size | 32.00 MB            |
---
## Iteration Runs
| Iteration   |     Time |
|-------------|----------|
| Run 0       | 1.24352  |
| Run 1       | 0.817626 |
| Run 2       | 0.810825 |
| Run 3       | 0.808383 |
| Run 4       | 0.808356 |
| Run 5       | 0.812417 |
| Run 6       | 0.818597 |
| Run 7       | 0.802335 |
| Run 8       | 0.837542 |
| Run 9       | 0.799536 |
---
## Compiled Code
```hlo
HloModule jit_do_ifft, is_scheduled=true, entry_computation_layout={(c128[256,32,256]{2,1,0})->c128[256,32,256]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="9b33e371f53f762493e41bc79df542b3"}

%fused_transpose (param_0.1: c128[32,256,256]) -> c128[256,32,8,32] {
  %param_0.1 = c128[32,256,256]{1,0,2} parameter(0)
  %bitcast.37.1 = c128[8,32,32,256]{3,2,1,0} bitcast(c128[32,256,256]{1,0,2} %param_0.1)
  ROOT %transpose.23.1 = c128[256,32,8,32]{3,2,1,0} transpose(c128[8,32,32,256]{3,2,1,0} %bitcast.37.1), dimensions={3,1,0,2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
}

%wrapped_transpose_computation (param_0.3: c128[256,32,256]) -> c128[32,256,256] {
  %param_0.3 = c128[256,32,256]{2,1,0} parameter(0)
  ROOT %transpose.20.1 = c128[32,256,256]{2,1,0} transpose(c128[256,32,256]{2,1,0} %param_0.3), dimensions={1,2,0}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
}

%wrapped_transpose_computation.1 (param_0.4: c128[32,256,256]) -> c128[256,32,256] {
  %param_0.4 = c128[32,256,256]{2,1,0} parameter(0)
  ROOT %transpose.22.1 = c128[256,32,256]{2,1,0} transpose(c128[32,256,256]{2,1,0} %param_0.4), dimensions={2,0,1}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
}

ENTRY %main.34_spmd (param.1: c128[256,32,256]) -> c128[256,32,256] {
  %param.1 = c128[256,32,256]{2,1,0} parameter(0), sharding={devices=[1,8,1]<=[8]}, metadata={op_name="arr"}
  %fft.15.0 = c128[256,32,256]{2,1,0} fft(c128[256,32,256]{2,1,0} %param.1), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
  %wrapped_transpose = c128[32,256,256]{2,1,0} fusion(c128[256,32,256]{2,1,0} %fft.15.0), kind=kInput, calls=%wrapped_transpose_computation, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/transpose[permutation=(1, 2, 0)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=45}
  %fft.16.0 = c128[32,256,256]{2,1,0} fft(c128[32,256,256]{2,1,0} %wrapped_transpose), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %wrapped_transpose.1 = c128[256,32,256]{2,1,0} fusion(c128[32,256,256]{2,1,0} %fft.16.0), kind=kInput, calls=%wrapped_transpose_computation.1, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=83}
  %bitcast.34.0 = c128[32,256,256]{1,0,2} bitcast(c128[256,32,256]{2,1,0} %wrapped_transpose.1)
  %all-to-all-start = ((c128[32,256,256]{1,0,2}), c128[32,256,256]{1,0,2}) all-to-all-start(c128[32,256,256]{1,0,2} %bitcast.34.0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={2}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  %all-to-all-done = c128[32,256,256]{1,0,2} all-to-all-done(((c128[32,256,256]{1,0,2}), c128[32,256,256]{1,0,2}) %all-to-all-start), metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %loop_transpose_fusion = c128[256,32,8,32]{3,2,1,0} fusion(c128[32,256,256]{1,0,2} %all-to-all-done), kind=kLoop, calls=%fused_transpose, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/all_to_all[axis_name=(\'y\',) split_axis=2 concat_axis=0 axis_index_groups=None tiled=True]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=49}
  %bitcast.41.0 = c128[256,32,256]{2,1,0} bitcast(c128[256,32,8,32]{3,2,1,0} %loop_transpose_fusion)
  ROOT %fft.17.0 = c128[256,32,256]{2,1,0} fft(c128[256,32,256]{2,1,0} %bitcast.41.0), fft_type=IFFT, fft_length={256}, metadata={op_name="jit(do_ifft)/jit(main)/jit(do_ifft)/jit(ifft3d)/jit(shmap_body)/jit(fft)/fft[fft_type=jaxlib.xla_extension.FftType.IFFT fft_lengths=(256,)]" source_file="/lustre/fswork/projects/rech/tkc/commun/jaxdecomp-benchmarks/scripts/jaxfft.py" source_line=81}
}


```

---
## Lowered Code
```hlo
module @jit_do_ifft attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "{2,1,0}", mhlo.sharding = "{devices=[1,8,1]<=[8]}"}) -> (tensor<256x256x256xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @do_ifft(%arg0) : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @do_ifft(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = call @ifft3d(%arg0) : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %0 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @ifft3d(%arg0: tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8,1]<=[8]}"} : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x256x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    %2 = call @shmap_body(%1) : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8,1]<=[8]}"} : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x256x256xcomplex<f64>>
    return %4 : tensor<256x256x256xcomplex<f64>>
  }
  func.func private @shmap_body(%arg0: tensor<256x32x256xcomplex<f64>>) -> (tensor<256x32x256xcomplex<f64>> {jax.result_info = "[('z',), ('y',), None]"}) {
    %0 = call @fft(%arg0) : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    %1 = stablehlo.transpose %0, dims = [1, 2, 0] : (tensor<256x32x256xcomplex<f64>>) -> tensor<32x256x256xcomplex<f64>>
    %2 = call @fft_0(%1) : (tensor<32x256x256xcomplex<f64>>) -> tensor<32x256x256xcomplex<f64>>
    %3 = "stablehlo.all_to_all"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, split_count = 8 : i64, split_dimension = 2 : i64}> : (tensor<32x256x256xcomplex<f64>>) -> tensor<256x256x32xcomplex<f64>>
    %4 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<256x256x32xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    %5 = call @fft_1(%4) : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    return %5 : tensor<256x32x256xcomplex<f64>>
  }
  func.func private @fft(%arg0: tensor<256x32x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x32x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    return %0 : tensor<256x32x256xcomplex<f64>>
  }
  func.func private @fft_0(%arg0: tensor<32x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<32x256x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<32x256x256xcomplex<f64>>) -> tensor<32x256x256xcomplex<f64>>
    return %0 : tensor<32x256x256xcomplex<f64>>
  }
  func.func private @fft_1(%arg0: tensor<256x32x256xcomplex<f64>> {mhlo.layout_mode = "default"}) -> (tensor<256x32x256xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %0 = stablehlo.fft %arg0, type =  IFFT, length = [256] : (tensor<256x32x256xcomplex<f64>>) -> tensor<256x32x256xcomplex<f64>>
    return %0 : tensor<256x32x256xcomplex<f64>>
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
                  jaxpr={ lambda ; g:c128[256,32,256]. let
                      h:c128[256,32,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; i:c128[256,32,256]. let
                            j:c128[256,32,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] i
                          in (j,) }
                      ] g
                      k:c128[256,32,256] = all_to_all[
                        axis_index_groups=None
                        axis_name=('z',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] h
                      l:c128[32,256,256] = transpose[permutation=(1, 2, 0)] k
                      m:c128[32,256,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; n:c128[32,256,256]. let
                            o:c128[32,256,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] n
                          in (o,) }
                      ] l
                      p:c128[256,256,32] = all_to_all[
                        axis_index_groups=None
                        axis_name=('y',)
                        concat_axis=0
                        split_axis=2
                        tiled=True
                      ] m
                      q:c128[256,32,256] = transpose[permutation=(1, 2, 0)] p
                      r:c128[256,32,256] = pjit[
                        name=fft
                        jaxpr={ lambda ; s:c128[256,32,256]. let
                            t:c128[256,32,256] = fft[
                              fft_lengths=(256,)
                              fft_type=jaxlib.xla_extension.FftType.IFFT
                            ] s
                          in (t,) }
                      ] q
                    in (r,) }
                  mesh=Mesh('z': 1, 'y': 8)
                  out_names=({0: ('z',), 1: ('y',)},)
                  rewrite=True
                ] e
              in (f,) }
          ] c
        in (d,) }
    ] a
  in (b,) }
```
