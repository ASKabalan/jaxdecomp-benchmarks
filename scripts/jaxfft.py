import jax
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import jax.numpy as jnp

import time
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils, multihost_utils
#from mpi4py import MPI
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
from cupy.cuda.nvtx import RangePush, RangePop
import argparse
from jax.experimental.shard_map import shard_map
from jax import lax
import os



def run_benchmark(pdims,global_shape,nb_nodes, output_path):

  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[
          global_shape[0] // pdims[1], global_shape[1] // pdims[0],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(0)) + jax.process_index()
  backend = "NCCL"
  print(f"Local array shape: {array.shape}")
  # remap to global array
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('a', 'b'))
  global_array = multihost_utils.host_local_array_to_global_array(array , mesh, P('b', 'a'))


  @jax.jit
  @partial(shard_map,
          mesh=mesh,
          in_specs=P('b' , 'a'),
          out_specs=P('b' , 'a'))
  def fft3d(mesh):
      """ Performs a 3D complex Fourier transform

      Args:
          mesh: a real 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D FFT of the input, note that the dimensions of the output
          are tranposed.
      """
      mesh = jnp.fft.fft(mesh)
      mesh = lax.all_to_all(mesh, 'b', 0, 0 ,  tiled=True)
      mesh = jnp.fft.fft(mesh)
      mesh = lax.all_to_all(mesh, 'a', 0, 0 ,  tiled=True)
      return jnp.fft.fft(mesh)  # Note the output is transposed # [z, x, y]

  @jax.jit
  @partial(shard_map,
          mesh=mesh,
          in_specs=P('b' , 'a'),
          out_specs=P('b' , 'a'))

  def ifft3d(mesh):
      """ Performs a 3D complex inverse Fourier transform

      Args:
          mesh: a complex 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D inverse FFT of the input, note that the dimensions of the output
          are tranposed.
      """
      mesh = jnp.fft.ifft(mesh)
      mesh = lax.all_to_all(mesh, 'b', 0, 0)
      mesh = jnp.fft.ifft(mesh)
      mesh = lax.all_to_all(mesh, 'a', 0, 0)
      return jnp.fft.ifft(mesh)



  if jax.process_index() == 0:
    print(f"Global array shape: {global_array.shape}")

  @jax.jit
  def do_fft(arr):
    return fft3d(arr)

  @jax.jit
  def do_ifft(arr):
    return ifft3d(arr)

  @jax.jit
  def get_diff(arr1, arr2):
        return jnp.abs(arr1 - arr2).max()

  with mesh:
    # warm start and get hlo
    RangePush("Warmup")
    do_fft(global_array).block_until_ready()
    RangePop()

    before = time.perf_counter()
    RangePush("Actual FFT Call")
    karray = do_fft(global_array).block_until_ready()
    RangePop()
    after = time.perf_counter()

  print(jax.process_index(), 'took', after - before, 's')
  with open(f"{output_path}/jaxfft.csv", 'a') as f:
    f.write(
        f"{jax.process_index()},{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{after - before}\n"
    )

  # And now, let's do the inverse FFT
  with mesh :
    rec_array = do_ifft(karray)
    # make sure get_diff is called inside the mesh context
    # it is done on non fully addressable global arrays in needs the mesh and to be jitted
    diff = get_diff(global_array, rec_array)

  if jax.process_index() == 0:
    print(f"Maximum reconstruction diff {diff}")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='JAX FFT Benchmark')
  parser.add_argument('-g', '--global_shape', type=int, help='Global shape', default=None)
  parser.add_argument('-p', '--pdims', type=str,
                      help='GPU grid', required=True)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  
  args = parser.parse_args()
  
  if args.local_shape is not None and args.global_shape is not None:
        print("Please provide either local_shape or global_shape")
        parser.print_help()
        exit(0)

  if args.local_shape is not None:
      global_shape = (args.local_shape * jax.device_count(), args.local_shape * jax.device_count(), args.local_shape * jax.device_count())
  elif args.global_shape is not None:
      global_shape = (args.global_shape, args.global_shape, args.global_shape)
  else:
      print("Please provide either local_shape or global_shape")
      parser.print_help()
      exit(0)
  pdims=tuple(map(int, args.pdims.split("x")))
  nb_nodes = args.nb_nodes
  output_path = args.output_path
  os.makedirs(output_path, exist_ok=True)


  run_benchmark(pdims,global_shape,nb_nodes, output_path)

jax.distributed.shutdown()
