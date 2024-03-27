import jax.numpy as jnp
import jax
import time
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils, multihost_utils
#from mpi4py import MPI
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
from cupy.cuda.nvtx import RangePush, RangePop
import argparse

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

jax.distributed.initialize()


def run_benchmark(global_shape,nb_nodes, output_path):

  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[
          global_shape[0] // jax.device_count(), global_shape[1],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(0)) + jax.process_index()
  pdims = (jax.device_count(), 1)
  backend = "NCCL"
  print(f"Local array shape: {array.shape}")
  # remap to global array
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('a', 'b'))
  global_array = multihost_utils.host_local_array_to_global_array(array , mesh, P('a', 'b'))

  if jax.process_index() == 0:
    print(f"Global array shape: {global_array.shape}")

  @jax.jit
  def do_fft(x):
    return jnp.fft.fftn(x)

  @jax.jit
  def do_ifft(x):
    return jnp.fft.ifftn(x)

  @jax.jit
  def get_diff(arr1, arr2):
        return jnp.abs(arr1 - arr2).max()

  with mesh:
    # warm start and get hlo
    RangePush("Warmup")
    do_fft(global_array).block_until_ready()
    RangePop()

    RangePush("Actual FFT Call")
    before = time.perf_counter()
    karray = do_fft(global_array).block_until_ready()
    after = time.perf_counter()
    RangePop()

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

  nb_nodes = args.nb_nodes
  output_path = args.output_path

  run_benchmark(global_shape,nb_nodes, output_path)

jax.distributed.shutdown()
