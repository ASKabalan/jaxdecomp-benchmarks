import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import argparse
import os
import time
from functools import partial

import jax.numpy as jnp
from cupy.cuda.nvtx import RangePop, RangePush
from jax import lax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.pjit import pjit
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import jax.profiler


def run_benchmark(pdims, global_shape, nb_nodes, precision, iterations , output_path):

    # Initialize the local slice with the local slice shape
    array = jax.random.normal(shape=[
        global_shape[0] // pdims[1], global_shape[1] // pdims[0],
        global_shape[2]
    ],
                              key=jax.random.PRNGKey(0)) + jax.process_index()
    backend = "NCCL"
    print(f"Local array shape: {array.shape}")
    # remap to global array
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    global_array = multihost_utils.host_local_array_to_global_array(
        array, mesh, P('z', 'y'))

    # @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
    def jax_transposeXtoY(data):
        return lax.all_to_all(data, 'y', 2, 1, tiled=True).transpose([2, 0, 1])

    # @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoZ(x):
        return lax.all_to_all(x, 'z', 2, 1, tiled=True).transpose([2, 0, 1])

    # @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
    def jax_transposeZtoY(x):
        return lax.all_to_all(x, 'z', 2, 0, tiled=True).transpose([1, 2, 0])

    # @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoX(x):
        return lax.all_to_all(x, 'y', 2, 0, tiled=True).transpose([1, 2, 0])

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
    def fft3d(mesh):
        """ Performs a 3D complex Fourier transform

      Args:
          mesh: a real 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D FFT of the input, note that the dimensions of the output
          are tranposed.
      """
        mesh = jnp.fft.fft(mesh)
        mesh = jax_transposeXtoY(mesh)
        mesh = jnp.fft.fft(mesh)
        mesh = jax_transposeYtoZ(mesh)
        return jnp.fft.fft(mesh)  # Note the output is transposed # [z, x, y]

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
    def ifft3d(mesh):
        """ Performs a 3D complex inverse Fourier transform

      Args:
          mesh: a complex 3D tensor of shape [Nx, Ny, Nz]

      Returns:
          3D inverse FFT of the input, note that the dimensions of the output
          are tranposed.
      """
        mesh = jnp.fft.ifft(mesh)
        mesh = jax_transposeZtoY(mesh)
        mesh = jnp.fft.ifft(mesh)
        mesh = jax_transposeYtoX(mesh)
        return jnp.fft.ifft(mesh)

    if jax.process_index() == 0:
        print(f"Global array shape: {global_array.shape}")

    @jax.jit
    def do_fft(arr):
        return fft3d(arr)

    @jax.jit
    def do_ifft(arr):
        return ifft3d(arr)



    with mesh:
        # Warm start
        do_fft(global_array).block_until_ready()
        for _ in range(iterations):
            global_array = do_fft(global_array).block_until_ready()

    out_path_params = f"{output_path}/{pdims[0]}x{pdims[1]}_{global_shape[0]}_{backend}_{nb_nodes}_{precision}"
    os.makedirs(out_path_params, exist_ok=True)
    jax.profiler.save_device_memory_profile(f"{out_path_params}/jaxfft{rank}.prof")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='JAX FFT Benchmark')
    parser.add_argument('-g',
                        '--global_shape',
                        type=int,
                        help='Global shape',
                        default=None)
    parser.add_argument('-p',
                        '--pdims',
                        type=str,
                        help='GPU grid',
                        required=True)
    parser.add_argument('-l',
                        '--local_shape',
                        type=int,
                        help='Local shape',
                        default=None)
    parser.add_argument('-n',
                        '--nb_nodes',
                        type=int,
                        help='Number of nodes',
                        default=1)
    parser.add_argument('-o',
                        '--output_path',
                        type=str,
                        help='Output path',
                        default=".")
    parser.add_argument('-pr',
                        '--precision',
                        type=str,
                        help='Precision',
                        default="float32")
    parser.add_argument('-i',
                        '--iterations',
                        type=int,
                        help='Iterations',
                        default=10)

    args = parser.parse_args()

    if args.local_shape is not None and args.global_shape is not None:
        print("Please provide either local_shape or global_shape")
        parser.print_help()
        exit(0)

    if args.local_shape is not None:
        global_shape = (args.local_shape * jax.device_count(),
                        args.local_shape * jax.device_count(),
                        args.local_shape * jax.device_count())
    elif args.global_shape is not None:
        global_shape = (args.global_shape, args.global_shape,
                        args.global_shape)
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

    run_benchmark(pdims, global_shape, nb_nodes, args.precision, args.iterations,output_path)

jax.distributed.shutdown()
