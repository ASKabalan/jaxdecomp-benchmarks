import jax
from jax._src.util import fun_name

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import argparse
import os
import time
from functools import partial

import jax.numpy as jnp
import numpy as np
from cupy.cuda.nvtx import RangePop, RangePush
from jax import lax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import Timer


def run_benchmark(pdims, global_shape, nb_nodes, precision, iterations, trace, output_path):

  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[global_shape[0] // pdims[1], global_shape[1] // pdims[0], global_shape[2]],
      key=jax.random.PRNGKey(rank)) + jax.process_index()
  backend = "NCCL"
  print(f"Local array shape: {array.shape}")
  # remap to global array
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices.T, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(array, mesh, P('z', 'y'))

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
  def jax_transposeXtoY(data):
    return lax.all_to_all(data, 'y', 2, 1, tiled=True).transpose([2, 0, 1])

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
  def jax_transposeYtoX(x):

    return lax.all_to_all(x, 'y', 2, 0, tiled=True).transpose([1, 2, 0])

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
  def jax_transposeYtoZ(x):
    return lax.all_to_all(x, 'z', 2, 1, tiled=True).transpose([2, 0, 1])

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
  def jax_transposeZtoY(x):
    return lax.all_to_all(x, 'z', 2, 0, tiled=True).transpose([1, 2, 0])

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
  def jax_transposeXtoZ(x):
    return lax.all_to_all(x, 'z', 2, 0, tiled=True).transpose([1, 2, 0])

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
  def jax_transposeZtoX(x):
    return lax.all_to_all(x, 'z', 2, 1, tiled=True).transpose([2, 0, 1])

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def fft1d(mesh):
    return jnp.fft.fft(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('y', 'z'))
  def fft1d_yz(mesh):
    return jnp.fft.fft(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def fft2d(mesh):
    return jnp.fft.fft2(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('y', 'z'))
  def fft2d_yz(mesh):
    return jnp.fft.fft2(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def ifft1d(mesh):
    return jnp.fft.ifft(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('y', 'z'))
  def ifft1d_yz(mesh):
    return jnp.fft.ifft(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def ifft2d(mesh):
    return jnp.fft.ifft2(mesh, norm=None)

  @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('y', 'z'))
  def ifft2d_yz(mesh):
    return jnp.fft.ifft(mesh, norm=None)

  from enum import Enum

  class PencilType(Enum):
    XYSLAB = 0
    YZSLAB = 1
    PENCIL = 2

  if pdims[0] > 1 and pdims[1] == 1:
    penciltype = PencilType.YZSLAB
    fft_out_spec = P('y', 'z')
    transpose_back = [2, 0, 1]
  elif pdims[0] == 1 and pdims[1] > 1:
    penciltype = PencilType.XYSLAB
    fft_out_spec = P('y', 'z')
    transpose_back = [1, 2, 0]
  else:
    penciltype = PencilType.PENCIL
    fft_out_spec = P('z', 'y')
    transpose_back = [1, 2, 0]

  def _fft_norm(s, func_name, norm):
    if norm == "backward":
      return 1 / jnp.prod(s) if func_name.startswith("i") else jnp.array(1)
    elif norm == "ortho":
      return (1 / jnp.sqrt(jnp.prod(s)) if func_name.startswith("i") else 1 / jnp.sqrt(jnp.prod(s)))
    elif norm == "forward":
      return jnp.prod(s) if func_name.startswith("i") else 1 / jnp.prod(s)**2
    raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                     '"ortho" or "forward".')

  @jax.jit
  def fft3d(mesh):
    """ Performs a 3D complex Fourier transform

      Args:
          mesh: a real 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D FFT of the input, note that the dimensions of the output
          are tranposed.
      """
    if penciltype == PencilType.XYSLAB:
      mesh = fft2d(mesh)  # FFT XY
      mesh = jax_transposeXtoZ(mesh)
      mesh = fft1d_yz(mesh)  # FFT Z
    elif penciltype == PencilType.YZSLAB:
      mesh = fft1d(mesh)  # FFT X
      mesh = jax_transposeXtoY(mesh)
      mesh = fft2d_yz(mesh)  # FFT YZ
    elif penciltype == PencilType.PENCIL:
      mesh = fft1d(mesh)
      mesh = jax_transposeXtoY(mesh)
      mesh = fft1d_yz(mesh)
      mesh = jax_transposeYtoZ(mesh)
      mesh = fft1d(mesh)

    return mesh  # Note the output is transposed # [z, x, y]

  @jax.jit
  def ifft3d(mesh):
    """ Performs a 3D complex inverse Fourier transform

      Args:
          mesh: a complex 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D inverse FFT of the input, note that the dimensions of the output
          are tranposed.
      """
    if penciltype == PencilType.XYSLAB:
      mesh = ifft2d_yz(mesh)  # IFFT XY
      mesh = jax_transposeZtoX(mesh)
      mesh = ifft1d(mesh)  # IFFT Z
    elif penciltype == PencilType.YZSLAB:
      mesh = ifft1d_yz(mesh)  # IFFT X
      mesh = jax_transposeYtoX(mesh)
      mesh = ifft2d(mesh)  # IFFT YZ
    elif penciltype == PencilType.PENCIL:
      mesh = ifft1d(mesh)
      mesh = jax_transposeYtoX(mesh)
      mesh = ifft1d_yz(mesh)
      mesh = jax_transposeZtoY(mesh)
      mesh = ifft1d(mesh)

    print(f"Mesh shape {mesh.shape}")

    # s = jnp.array(mesh.shape , dtype=mesh.dtype)
    # mesh = mesh * _fft_norm(s, func_name='ifft', norm='backward')

    return mesh

  @jax.jit
  def do_fft(arr):
    return fft3d(arr)

  @jax.jit
  def do_ifft(arr):
    return ifft3d(arr)

  fft_chrono = Timer(save_jaxpr=True)
  ifft_chrono = Timer(save_jaxpr=True)
  with mesh:
    if trace:
      jit_fft_output = f"{output_path}/jit_fft_trace"
      first_run_fft_output = f"{output_path}/first_run_fft_trace"
      second_run_fft_output = f"{output_path}/second_run_ifft_trace"
      jit_ifft_output = f"{output_path}/jit_ifft_trace"
      first_run_ifft_output = f"{output_path}/first_run_ifft_trace"
      second_run_ifft_output = f"{output_path}/second_run_ifft_trace"

      with jax.profiler.trace(jit_fft_output, create_perfetto_trace=True):
        global_array = do_fft(global_array).block_until_ready()
      with jax.profiler.trace(jit_ifft_output, create_perfetto_trace=True):
        global_array = do_ifft(global_array).block_until_ready()

      with jax.profiler.trace(first_run_fft_output, create_perfetto_trace=True):
        global_array = do_fft(global_array).block_until_ready()
      with jax.profiler.trace(first_run_ifft_output, create_perfetto_trace=True):
        global_array = do_ifft(global_array).block_until_ready()

      with jax.profiler.trace(second_run_fft_output, create_perfetto_trace=True):
        global_array = do_ifft(global_array).block_until_ready()
      with jax.profiler.trace(second_run_ifft_output, create_perfetto_trace=True):
        global_array = do_ifft(global_array).block_until_ready()

    else:
      # Warm start
      RangePush("warmup")
      global_array = fft_chrono.chrono_jit(do_fft, global_array)
      global_array = ifft_chrono.chrono_jit(do_ifft, global_array)
      RangePop()
      sync_global_devices("warmup")
      for i in range(iterations):
        RangePush(f"fft iter {i}")
        global_array = fft_chrono.chrono_fun(do_fft, global_array)
        RangePop()
        RangePush(f"ifft iter {i}")
        global_array = ifft_chrono.chrono_fun(do_ifft, global_array)
        RangePop()

  if not trace:
    fft_metadata = {
        'function': "FFT",
        'precision': precision,
        'x': str(global_shape[0]),
        'px': str(pdims[0]),
        'py': str(pdims[1]),
        'backend': backend,
        'nodes': str(nb_nodes),
    }
    ifft_metadata = {
        'function': "IFFT",
        'precision': precision,
        'x': str(global_shape[0]),
        'px': str(pdims[0]),
        'py': str(pdims[1]),
        'backend': backend,
        'nodes': str(nb_nodes),
    }

    fft_chrono.report(f"{output_path}/jaxfft.csv", **fft_metadata)
    ifft_chrono.report(f"{output_path}/jaxfft.csv", **ifft_metadata)
  print(f"Done")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='JAX FFT Benchmark')
  parser.add_argument('-g', '--global_shape', type=int, help='Global shape', default=None)
  parser.add_argument('-p', '--pdims', type=str, help='GPU grid', required=True)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  parser.add_argument('-pr', '--precision', type=str, help='Precision', default="float32")
  parser.add_argument('-i', '--iterations', type=int, help='Iterations', default=10)
  parser.add_argument('-t', '--trace', action='store_true', help='Profile using tensorboard')

  args = parser.parse_args()

  if args.local_shape is not None and args.global_shape is not None:
    print("Please provide either local_shape or global_shape")
    parser.print_help()
    exit(0)

  if args.local_shape is not None:
    global_shape = (args.local_shape * jax.device_count(), args.local_shape * jax.device_count(),
                    args.local_shape * jax.device_count())
  elif args.global_shape is not None:
    global_shape = (args.global_shape, args.global_shape, args.global_shape)
  else:
    print("Please provide either local_shape or global_shape")
    parser.print_help()
    exit(0)
  pdims = tuple(map(int, args.pdims.split("x")))
  nb_nodes = args.nb_nodes
  output_path = args.output_path
  os.makedirs(output_path, exist_ok=True)

  if args.precision == "float32":
    jax.config.update("jax_enable_x64", False)
  elif args.precision == "float64":
    jax.config.update("jax_enable_x64", True)
  else:
    print("Precision should be either float32 or float64")
    parser.print_help()
    exit(0)

  run_benchmark(pdims, global_shape, nb_nodes, args.precision, args.iterations, args.trace,
                output_path)

jax.distributed.shutdown()
