# -*- coding: utf-8 -*-
import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from cufftmp_jax import cufftmp
from cupy.cuda.nvtx import RangePop, RangePush
from fft_common import Dir, Dist
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import Timer
from xfft import xfft


def run_benchmark(pdims, global_shape, nb_nodes, precision, iterations, trace, output_path):

  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[global_shape[0] // pdims[0], global_shape[1] // pdims[1], global_shape[2]],
      key=jax.random.PRNGKey(rank))

  if pdims[0] == 1:
    dist = Dist.SLABS_X
  elif pdims[1] == 1:
    dist = Dist.SLABS_Y

  dtype = jnp.complex64 if precision == "float32" else jnp.complex128

  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices.T, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(array, mesh, P('z', 'y'))
  if pdims[0] == 1:
    dist = Dist.create("X")
  elif pdims[1] == 1:
    dist = Dist.create("Y")
  else:
    raise ValueError("Invalid pdims, CUFFTMP only supports 1D decomposition (slabs)")

  @jax.jit
  def do_fft(x):
    return cufftmp(x, dist, Dir.FWD)

  @jax.jit
  def do_ifft(x):
    return cufftmp(x, dist.opposite, Dir.FWD)

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
        'backend': "NVSHMEM",
        'nodes': str(nb_nodes),
    }
    ifft_metadata = {
        'function': "IFFT",
        'precision': precision,
        'x': str(global_shape[0]),
        'px': str(pdims[0]),
        'py': str(pdims[1]),
        'backend': "NVSHMEM",
        'nodes': str(nb_nodes),
    }

    fft_chrono.report(f"{output_path}/jaxfft.csv", **fft_metadata)
    ifft_chrono.report(f"{output_path}/jaxfft.csv", **ifft_metadata)
  print(f"Done")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='CufftMP Benchmark')
  parser.add_argument('-p', '--pdims', type=str, help='GPU grid', required=True)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument(
      '-g', '--global_shape', type=int, help='Global shape of the array', default=None)
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  parser.add_argument('-pr', '--precision', type=str, help='Precision', default="float32")
  parser.add_argument('-i', '--iterations', type=int, help='Number of iterations', default=10)

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

  if args.precision == "float32":
    jax.config.update("jax_enable_x64", False)
  elif args.precision == "float64":
    jax.config.update("jax_enable_x64", True)
  else:
    print("Precision should be either float32 or float64")
    parser.print_help()
    exit(0)

  pdims = tuple(map(int, args.pdims.split("x")))

  nb_nodes = args.nb_nodes
  output_path = args.output_path
  import os
  os.makedirs(output_path, exist_ok=True)

  run_benchmark(pdims, global_shape, nb_nodes, args.precision, args.iterations, output_path)
