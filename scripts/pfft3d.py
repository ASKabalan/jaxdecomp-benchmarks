import jax
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
from functools import partial
import re
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from cupy.cuda.nvtx import RangePush, RangePop

import jaxdecomp
import jax.numpy as jnp

import time
import argparse


def run_benchmark(pdims, global_shape, backend, nb_nodes, output_path):

  if backend == "NCCL":
    jaxdecomp.config.update('transpose_comm_backend',
                            jaxdecomp.TRANSPOSE_COMM_NCCL)
  elif backend == "MPI":
    jaxdecomp.config.update('transpose_comm_backend',
                            jaxdecomp.TRANSPOSE_COMM_MPI_A2A)

  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[
          global_shape[0] // pdims[0], global_shape[1] // pdims[1],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(rank))

  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(
      array, mesh, P('z', 'y'))
  
  if jax.process_index() == 0:
    print(f"Devices {jax.devices()}")
    print(
        f"Global dims {global_shape}, pdims {pdims} , Backend {backend} local array shape {array.shape} global array shape {global_array.shape}"
    )
    print("Sharding :")
    print(global_array.sharding)

  @jax.jit
  def do_fft(x):
    return jaxdecomp.fft.pfft3d(x)

  @jax.jit
  def do_ifft(x):
    return jaxdecomp.fft.pifft3d(x)

  # Wass debugging start
  @jax.jit
  def jax_fft(x):
    arr = jax.numpy.fft.fft(x, axis=2)
    arr = jax.numpy.fft.fft(arr, axis=1)
    arr = jax.numpy.fft.fft(arr, axis=0)
    arr = arr.transpose([1, 2, 0])
    return arr

  @jax.jit
  def jax_ifft(x):
    arr = jax.numpy.fft.ifft(x, axis=0)
    arr = jax.numpy.fft.ifft(arr, axis=1)
    arr = jax.numpy.fft.ifft(arr, axis=2)
    arr = arr.transpose(2, 0, 1)
    return arr

  def print_array(arr):
    print(arr.shape)
    for z in range(arr.shape[0]):
      for y in range(arr.shape[1]):
        for x in range(arr.shape[2]):
          print(f"({z},{y},{x}) {arr[z][y][x]}")

  @jax.jit
  def get_diff(arr1, arr2):
        return jnp.abs(arr1 - arr2).max()


  with mesh:
    # Warm start
    RangePush("Warmup")
    do_fft(global_array).block_until_ready()
    RangePop()

    before = time.perf_counter()
    RangePush("Actual FFT Call")
    karray = do_fft(global_array).block_until_ready()
    RangePop()
    after=time.perf_counter()

  print(rank, 'took', after - before, 's')
  with open(f"{output_path}/jaxdecompfft.csv", 'a') as f:
    f.write(
        f"{jax.process_index()},FFT,{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{after - before}\n"
    )

  # And now, let's do the inverse FFT

  with mesh:
    RangePush("IFFT Warmup")
    rec_array=do_ifft(karray).block_until_ready()
    RangePop()

    before = time.perf_counter()
    RangePush("Actual IFFT Call")
    rec_array = do_ifft(karray).block_until_ready()
    RangePop()
    after = time.perf_counter()

    # make sure get_diff is called inside the mesh context
    # it is done on non fully addressable global arrays in needs the mesh and to be jitted
    diff = get_diff(global_array, rec_array)

  if jax.process_index() == 0:
    print(f"Maximum reconstruction diff {diff}")

  with open(f"{output_path}/jaxdecompfft.csv", 'a') as f:
    f.write(
        f"{jax.process_index()},IFFT,{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{after - before}\n"
    )
    



if __name__ == "__main__":

  parser=argparse.ArgumentParser(description='NBody Benchmark')
  parser.add_argument('-p', '--pdims', type=str,
                      help='GPU grid', required=True)
  parser.add_argument('-l', '--local_shape', type=int,
                      help='Local shape', default=None)
  parser.add_argument('-g', '--global_shape', type=int,
                      help='Global shape of the array', default=None)
  parser.add_argument('-b', '--backend', type=str,
                      help='Backend to use for transpose comm', default="NCCL")
  parser.add_argument('-n', '--nb_nodes', type=int,
                      help='Number of nodes', default=1)
  parser.add_argument('-o', '--output_path', type=str,
                      help='Output path', default=".")

  args=parser.parse_args()

  if args.local_shape is not None and args.global_shape is not None:
        print("Please provide either local_shape or global_shape")
        parser.print_help()
        exit(0)

  if args.local_shape is not None:
      global_shape=(args.local_shape * jax.device_count(), args.local_shape * jax.device_count(), args.local_shape * jax.device_count())
  elif args.global_shape is not None:
      global_shape=(args.global_shape, args.global_shape, args.global_shape)
  else:
      print("Please provide either local_shape or global_shape")
      parser.print_help()
      exit(0)


  pdims=tuple(map(int, args.pdims.split("x")))

  backend=args.backend
  nb_nodes=args.nb_nodes
  output_path=args.output_path
  import os
  os.makedirs(output_path, exist_ok=True)

  for dim in global_shape:
    for pdim in pdims:
      if dim % pdim != 0:
        print(f"Global shape {global_shape} is not divisible by pdims {pdims}")
        exit(0)
        # Do not raise error for slurm jobs
        # raise ValueError(f"Global shape {global_shape} is not divisible by pdims {pdims}")

  run_benchmark(pdims, global_shape, backend, nb_nodes, output_path)

jaxdecomp.finalize()
jax.distributed.shutdown()
