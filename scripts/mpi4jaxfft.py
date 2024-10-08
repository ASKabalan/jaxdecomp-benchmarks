import jax

jax.distributed.initialize()
import argparse
import os
import time

import jax.numpy as jnp
import mpi4jax
import numpy as np
from cupy.cuda.nvtx import RangePop, RangePush
from mpi4py import MPI

# Create communicators
world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

if rank == 0:
  print("Communication setup done!")


def chrono_fun(fun, *args):
  start = time.perf_counter()
  out = fun(*args).block_until_ready()
  end = time.perf_counter()
  return out, end - start


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
        'ij,xyjz->iyzx', jnp.eye(nx),
        arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
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
        'ij,yzjx->izxy', jnp.eye(ny),
        arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
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
        'ij,zxjy->izyx', jnp.eye(ny),
        arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
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
        'ij,zyjx->ixyz', jnp.eye(nx),
        arr)  # TODO: remove this hack when we understand why transpose before alltoall doenst work
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
  return jax.random.normal(key, [shape[0] // nx, shape[1] // ny] + list(shape[2:]))


def run_benchmark(global_shape, nb_nodes, pdims, precision, iterations, output_path):
  """ Run the benchmark
    """

  cart_comm = MPI.COMM_WORLD.Create_cart(dims=list(pdims), periods=[True, True])
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
  global_array = normal(key, mesh_shape, comms=comms)

  if jax.process_index() == 0:
    print(f"Devices {jax.devices()}")
    print(
        f"Global dims {global_shape}, pdims ({comms[0].Get_size()},{comms[1].Get_size()}) , Bachend {backend} original_array shape {global_array.shape}"
    )

  @jax.jit
  def do_fft(x):
    return fft3d(x, comms=comms)

  @jax.jit
  def do_ifft(x):
    return ifft3d(x, comms=comms)

  jit_fft_time = 0
  jit_ifft_time = 0
  ffts_times = []
  iffts_times = []
  # Warm start
  RangePush("warmup")
  global_array, jit_fft_time = chrono_fun(do_fft, global_array)
  global_array, jit_ifft_time = chrono_fun(do_ifft, global_array)
  RangePop()
  MPI.COMM_WORLD.Barrier()
  for i in range(iterations):
    RangePush(f"fft iter {i}")
    global_array, fft_time = chrono_fun(do_fft, global_array)
    RangePop()
    ffts_times.append(fft_time)
    RangePush(f"ifft iter {i}")
    global_array, ifft_time = chrono_fun(do_ifft, global_array)
    RangePop()
    iffts_times.append(ifft_time)

  ffts_times = np.array(ffts_times) * 1e3
  iffts_times = np.array(iffts_times) * 1e3
  # FFT
  jit_fft_time *= 1e3
  fft_min_time = np.min(ffts_times)
  fft_max_time = np.max(ffts_times)
  fft_mean_time = jnp.mean(ffts_times)
  fft_std_time = jnp.std(ffts_times)
  last_fft_time = ffts_times[-1]
  # IFFT
  jit_ifft_time *= 1e3
  ifft_min_time = np.min(iffts_times)
  ifft_max_time = np.max(iffts_times)
  ifft_mean_time = jnp.mean(iffts_times)
  ifft_std_time = jnp.std(iffts_times)
  last_ifft_time = iffts_times[-1]

  # RANK TYPE PRECISION SIZE PDIMS BACKEND NB_NODES JIT_TIME MIN MAX MEAN STD
  with open(f"{output_path}/mpi4jaxfft.csv", 'a') as f:
    f.write(
        f"{jax.process_index()},FFT,{precision},{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{jit_fft_time:.4f},{fft_min_time:.4f},{fft_max_time:.4f},{fft_mean_time:.4f},{fft_std_time:.4f},{last_fft_time:.4f}\n"
    )
    f.write(
        f"{jax.process_index()},IFFT,{precision},{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{jit_ifft_time:.4f},{ifft_min_time:.4f},{ifft_max_time:.4f},{ifft_mean_time:.4f},{ifft_std_time:.4f},{last_ifft_time:.4f}\n"
    )

  print(f"Done")
  print(f"FFT times")
  print(f"JIT time {jit_fft_time:.4f} ms")
  for i in range(iterations):
    print(f"FFT {i} time {ffts_times[i]:.4f} ms")
  print(f"IFFT times ")
  print(f"JIT time {jit_ifft_time:.4f} ms")
  for i in range(iterations):
    print(f"IFFT {i} time {iffts_times[i]:.4f} ms")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='NBody MPI4JAX Benchmark')
  parser.add_argument('-g', '--global_shape', type=int, help='Global shape', default=None)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-p', '--pdims', type=str, help='GPU grid', required=True)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  parser.add_argument('-pr', '--precision', type=str, help='Precision', default="float32")
  parser.add_argument('-i', '--iterations', type=int, help='Number of iterations', default=10)

  args = parser.parse_args()

  if args.local_shape is not None:
    global_shape = (args.global_shape * jax.device_count(), args.global_shape * jax.device_count(),
                    args.global_shape * jax.devices())
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

  nb_nodes = args.nb_nodes
  output_path = args.output_path
  os.makedirs(output_path, exist_ok=True)
  pdims = [int(x) for x in args.pdims.split("x")]

  run_benchmark(global_shape, nb_nodes, pdims, args.precision, args.iterations, output_path)
