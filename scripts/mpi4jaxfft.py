import jax
jax.distributed.initialize()
from mpi4py import MPI
import jax.numpy as jnp
import mpi4jax
import time
from cupy.cuda.nvtx import RangePush, RangePop
import argparse
import os
# Create communicators
world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()



if rank == 0:
  print("Communication setup done!")


def fft3d(arr, comms=None):
  """ Computes forward FFT, note that the output is transposed
    """
  if comms is not None:
    shape = list(arr.shape)
    nx = comms[0].Get_size()
    ny = comms[1].Get_size()

  # First FFT along z
  arr = jnp.fft.fft(arr)  # [x, y, z]
  # Perform single gpu or distributed transpose
  if comms == None:
    arr = arr.transpose([1, 2, 0])
  else:
    arr = arr.reshape(shape[:-1] + [nx, shape[-1] // nx])
    #arr = arr.transpose([2, 1, 3, 0])  # [y, z, x]
    arr = jnp.einsum(
        'ij,xyjz->iyzx', jnp.eye(nx), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[0])
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [y, z, x]

  # Second FFT along x
  arr = jnp.fft.fft(arr)
  # Perform single gpu or distributed transpose
  if comms == None:
    arr = arr.transpose([1, 2, 0])
  else:
    arr = arr.reshape(shape[:-1] + [ny, shape[-1] // ny])
    #arr = arr.transpose([2, 1, 3, 0])  # [z, x, y]
    arr = jnp.einsum(
        'ij,yzjx->izxy', jnp.eye(ny), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[1], token=token)
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z, x, y]

  # Third FFT along y
  return jnp.fft.fft(arr)


def ifft3d(arr, comms=None):
  """ Let's assume that the data is distributed accross x
    """
  if comms is not None:
    shape = list(arr.shape)
    nx = comms[0].Get_size()
    ny = comms[1].Get_size()

  # First FFT along y
  arr = jnp.fft.ifft(arr)  # Now [z, x, y]
  # Perform single gpu or distributed transpose
  if comms == None:
    arr = arr.transpose([0, 2, 1])
  else:
    arr = arr.reshape(shape[:-1] + [ny, shape[-1] // ny])
    # arr = arr.transpose([2, 0, 3, 1])  # Now [z, y, x]
    arr = jnp.einsum(
        'ij,zxjy->izyx', jnp.eye(ny), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[1])
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z,y,x]

  # Second FFT along x
  arr = jnp.fft.ifft(arr)
  # Perform single gpu or distributed transpose
  if comms == None:
    arr = arr.transpose([2, 1, 0])
  else:
    arr = arr.reshape(shape[:-1] + [nx, shape[-1] // nx])
    # arr = arr.transpose([2, 3, 1, 0])  # now [x, y, z]
    arr = jnp.einsum(
        'ij,zyjx->ixyz', jnp.eye(nx), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[0], token=token)
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [x,y,z]

  # Third FFT along z
  return jnp.fft.ifft(arr)


def normal(key, shape, comms=None):
  """ Generates a normal variable for the given
    global shape.
    """
  if comms is None:
    return jax.random.normal(key, shape)

  nx = comms[0].Get_size()
  ny = comms[1].Get_size()

  print(shape)
  return jax.random.normal(key,
                           [shape[0] // nx, shape[1] // ny] + list(shape[2:]))


def run_benchmark(global_shape, nb_nodes, pdims , output_path):

  cart_comm = MPI.COMM_WORLD.Create_cart(dims=[2, 2], periods=[True, True])
  comms = [cart_comm.Sub([True, False]), cart_comm.Sub([False, True])]

  backend = "MPI4JAX"
  # Setup random keys
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]

  # Size of the FFT
  N = 256
  mesh_shape = list(global_shape)

  # Generate a random gaussian variable for the global
  # mesh shape
  original_array = normal(key, mesh_shape, comms=comms)

  if jax.process_index() == 0:
    print(f"Devices {jax.devices()}")
    print(
        f"Global dims {global_shape}, pdims ({comms[0].Get_size()},{comms[1].Get_size()}) , Bachend {backend} original_array shape {original_array.shape}"
    )

  @jax.jit
  def do_fft(x):
    return fft3d(x, comms=comms)

  @jax.jit
  def do_ifft(x):
    return ifft3d(x, comms=comms)

  #Warm start and get the HLO
  RangePush("Warmup")
  do_fft(original_array).block_until_ready()
  RangePop()
  
  before = time.time()
  RangePush("Actual FFT Call")
  karray = do_fft(original_array).block_until_ready()
  RangePop()
  after = time.time()
  

  print(rank, 'took', after - before, 's')

  # And now, let's do the inverse FFT
  rec_array = do_ifft(karray)

  # Let's test if things are like we expect
  diff = rec_array - karray
  print('maximum reconstruction difference', jnp.abs(diff).max())
  with open(f"{output_path}/mpi4jax.csv", 'a') as f:
    f.write(
        f"{rank},{global_shape[0]},{global_shape[1]},{global_shape[2]},{comms[0].Get_size()},{comms[1].Get_size()},{backend},{nb_nodes},{after - before}\n"
    )

  # Testing that the fft is indeed invertible
  print("I'm ", rank, abs(rec_array.real - original_array).mean())


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='NBody MPI4JAX Benchmark')
  parser.add_argument('-g', '--global_shape', type=int, help='Global shape', default=None)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-p', '--pdims', type=str, help='GPU grid', required=True)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  
  args = parser.parse_args()

  if args.local_shape is not None:
      global_shape = (args.global_shape * jax.device_count(), args.global_shape * jax.device_count(), args.global_shape * jax.devices())
  elif args.global_shape is not None:
      global_shape = (args.global_shape, args.global_shape, args.global_shape)
  else:
      print("Please provide either local_shape or global_shape")
      parser.print_help()
      exit(0)
  print(f"shape {global_shape}")
  nb_nodes = args.nb_nodes
  output_path = args.output_path
  os.makedirs(output_path, exist_ok=True)
  pdims = [int(x) for x in args.pdims.split("x")]

  run_benchmark(global_shape, nb_nodes, pdims , output_path)
