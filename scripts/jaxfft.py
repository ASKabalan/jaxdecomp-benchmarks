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


def run_benchmark(pdims, global_shape, nb_nodes, precision, output_path):

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
        return  lax.all_to_all(data, 'y', 2, 1, tiled=True).transpose([2 , 0 , 1])

    # @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoZ(x):
        return lax.all_to_all(x, 'z', 2, 1, tiled=True).transpose([2 , 0 , 1])

    # @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
    def jax_transposeZtoY(x):
        return lax.all_to_all(x, 'z', 2, 0, tiled=True).transpose([1, 2 ,0])

    # @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoX(x):
        return lax.all_to_all(x, 'y', 2, 0, tiled=True).transpose([1, 2 ,0])

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

    print(jax.process_index(), 'fft took', after - before, 's')
    with open(f"{output_path}/jaxfft.csv", 'a') as f:
        f.write(
            f"{jax.process_index()},FFT,{precision},{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{after - before}\n"
        )

    # And now, let's do the inverse FFT
    with mesh:
        RangePush("IFFT Warmup")
        rec_array = do_ifft(karray).block_until_ready()
        RangePop()

        before = time.perf_counter()
        RangePush("Actual IFFT Call")
        rec_array = do_ifft(karray).block_until_ready()
        RangePop()
        after = time.perf_counter()
        # make sure get_diff is called inside the mesh context
        # it is done on non fully addressable global arrays in needs the mesh and to be jitted
        diff = get_diff(global_array, rec_array)

    # gathered_array = multihost_utils.process_allgather(global_array,tiled=True)
    # k_gathered = jnp.fft.fftn(gathered_array).transpose([1 , 2 , 0])
    # rec_array_gathered =jnp.fft.ifftn(k_gathered).transpose([2 , 0 , 1])

    # jd_k_gathered = multihost_utils.process_allgather(karray,tiled=True)
    # jd_rec_array_gathered = multihost_utils.process_allgather(rec_array,tiled=True)

    # diff_gathered = jnp.abs(gathered_array - rec_array_gathered).max()
    # diff_k_gathered = jnp.abs(jd_k_gathered.real- k_gathered.real).max()

    # from itertools import permutations
    # for perm in permutations([0, 1, 2]):
    #     if jnp.all(jd_k_gathered == k_gathered.transpose(perm)):
    #         print(f"gathered_jd_xy Permutation {perm} is good 🎉")
    #     else:
    #         print(f"gathered_jd_xy Permutation {perm} is bad")

    # print(f"Diff between gathered and iffted array: {diff_gathered}")
    # print(f"Diff between diff_k_gathered and jd_k_gathered array : {diff_k_gathered}")

    print(jax.process_index(), 'ifft took', after - before, 's')

    with open(f"{output_path}/jaxfft.csv", 'a') as f:
        f.write(
            f"{jax.process_index()},IFFT,{precision},{global_shape[0]},{global_shape[1]},{global_shape[2]},{pdims[0]},{pdims[1]},{backend},{nb_nodes},{after - before}\n"
        )

    if jax.process_index() == 0:
        print(f"Maximum reconstruction diff {diff}")


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

    run_benchmark(pdims, global_shape, nb_nodes, args.precision, output_path)

jax.distributed.shutdown()
