import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import jax.numpy as jnp
from mpi4py import MPI

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from jax.experimental.shard_map import shard_map
from jax import lax

import pytest
from math import prod

from jaxdecomp import (transposeXtoY, transposeYtoX, transposeYtoZ,
                       transposeZtoY)
import mpi4jax
from numpy.testing import assert_array_equal

# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

    assert (len(global_shape) == 3)
    assert (len(pdims) == 2)
    assert (
        prod(pdims) == size
    ), "The product of pdims must be equal to the number of MPI processes"
    local_array = jnp.arange((global_shape[0] // pdims[1]) *
                             (global_shape[1] // pdims[0]) * global_shape[2])

    local_array = local_array.reshape(global_shape[0] // pdims[1],
                                      global_shape[1] // pdims[0],
                                      global_shape[2])
    local_array = local_array + (100**rank)
    local_array = jnp.array(local_array, dtype=jnp.float32)

    # Remap to the global array from the local slice
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    global_array = multihost_utils.host_local_array_to_global_array(
        local_array, mesh, P('z', 'y'))

    return global_array, mesh


pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [(4, 8, 16), (4, 4, 4), (29 * size, 19 * size, 17 * size)
                 ]  # Cubes, non-cubes and primes


def mpi4jax_transposeXtoY(arr , comms):
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    arr = arr.reshape(shape[:-1] + [nx, shape[-1] // nx])
    #arr = arr.transpose([2, 1, 3, 0])  # [y, z, x]
    arr = jnp.einsum(
        'ij,xyjz->iyzx', jnp.eye(nx), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[0])
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [y, z, x]
    return arr  

def mpi4jax_transposeYtoX(arr, comms):
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    arr = arr.reshape(shape[:-1] + [ny, shape[-1] // ny])
    #arr = arr.transpose([2, 1, 3, 0])  # [z, x, y]
    arr = jnp.einsum(
        'ij,yzjx->izxy', jnp.eye(ny), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[1], token=token)
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z, x, y]




def mpi4jax_transposeYtoZ(arr, comms):
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    arr = arr.reshape(shape[:-1] + [ny, shape[-1] // ny])
    # arr = arr.transpose([2, 0, 3, 1])  # Now [z, y, x]
    arr = jnp.einsum(
        'ij,zxjy->izyx', jnp.eye(ny), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[1])
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [z,y,x]


def mpi4jax_transposeZtoY(arr, comms):
    if comms is not None:
        shape = list(arr.shape)
        nx = comms[0].Get_size()
        ny = comms[1].Get_size()

    arr = arr.reshape(shape[:-1] + [nx, shape[-1] // nx])
    # arr = arr.transpose([2, 3, 1, 0])  # now [x, y, z]
    arr = jnp.einsum(
        'ij,zyjx->ixyz', jnp.eye(nx), arr
    )  # TODO: remove this hack when we understand why transpose before alltoall doenst work
    arr, token = mpi4jax.alltoall(arr, comm=comms[0], token=token)
    arr = arr.transpose([1, 2, 0, 3]).reshape(shape)  # Now [x,y,z]


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
def test_tranpose(pdims, global_shape):
    """ Goes from an array of shape [z,y,x] # What we call an x pencil
    to [x,z,y] # what we call a y pencil
    """
    print("*" * 80)
    print(f"Testing with pdims {pdims} and global shape {global_shape}")

    global_array, mesh = create_spmd_array(global_shape, pdims)

    @jax.jit
    @partial(shard_map, mesh=mesh, in_parts=P('z', 'y'), out_parts=P('y', 'z'))
    def jax_transposeXtoY(x):
        return lax.all_to_all(x, 'z', 0, 0, tiled=True)
    
    @jax.jit
    @partial(shard_map, mesh=mesh, in_parts=P('y', 'z'), out_parts=P('z', 'y'))
    def jax_transposeYtoX(x):
        return lax.all_to_all(x, 'y', 0, 0, tiled=True)
    
    @jax.jit
    @partial(shard_map, mesh=mesh, in_parts=P('z', 'y'), out_parts=P('y', 'z'))
    def jax_transposeYtoZ(x):
        return lax.all_to_all(x, 'z', 0, 0, tiled=True)
    
    @jax.jit
    @partial(shard_map, mesh=mesh, in_parts=P('y', 'z'), out_parts=P('z', 'y'))
    def jax_transposeZtoY(x):
        return lax.all_to_all(x, 'y', 0, 0, tiled=True)
    


    cart_comm = MPI.COMM_WORLD.Create_cart(dims=list(pdims), periods=[True, True])
    comms = [cart_comm.Sub([True, False]), cart_comm.Sub([False, True])]
    


    with mesh:
        # JD transposes
        jd_tranposed_xy = transposeXtoY(global_array)
        jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
        jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
        jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)
        # JAX transposes
        jax_transposed_xy = jax_transposeXtoY(global_array)
        jax_tranposed_yz = jax_transposeYtoZ(jax_transposed_xy)
        jax_tranposed_zy = jax_transposeZtoY(jax_tranposed_yz)
        jax_tranposed_yx = jax_transposeYtoX(jax_tranposed_zy)
        # MPI4JAX transposes
        mpi4jax_transposed_xy = mpi4jax_transposeXtoY(global_array, comms)
        mpi4jax_tranposed_yz = mpi4jax_transposeYtoZ(mpi4jax_transposed_xy, comms)
        mpi4jax_tranposed_zy = mpi4jax_transposeZtoY(mpi4jax_tranposed_yz, comms)
        mpi4jax_tranposed_yx = mpi4jax_transposeYtoX(mpi4jax_tranposed_zy, comms)





    gathered_array = multihost_utils.process_allgather(global_array,
                                                       tiled=True)

    gathered_jd_xy = multihost_utils.process_allgather(jd_tranposed_xy,
                                                       tiled=True)
    gathered_jd_yz = multihost_utils.process_allgather(jd_tranposed_yz,
                                                       tiled=True)
    gathered_jd_zy = multihost_utils.process_allgather(jd_tranposed_zy,
                                                       tiled=True)
    gathered_jd_yx = multihost_utils.process_allgather(jd_tranposed_yx,
                                                       tiled=True)
    
    gathered_jax_xy = multihost_utils.process_allgather(jax_transposed_xy,
                                                        tiled=True)
    gathered_jax_yz = multihost_utils.process_allgather(jax_tranposed_yz,
                                                        tiled=True)
    gathered_jax_zy = multihost_utils.process_allgather(jax_tranposed_zy,
                                                        tiled=True)
    gathered_jax_yx = multihost_utils.process_allgather(jax_tranposed_yx,
                                                        tiled=True)
    

    # Explanation :
    # Tranposing forward is a shift axis to the right so ZYX to XZY to YXZ (2 0 1)
    # Tranposing backward is a shift axis to the left so YXZ to XZY to ZYX (1 2 0)
    # Double Tranposing from ZYX to YXZ is double (2 0 1) so  (1 2 0)

    forward_tranpose = [2, 0, 1]
    backward_tranpose = [1, 2, 0]
    double_forward = [1, 2, 0]

    #
    # Test X to Y transpose
    # It tranposes ZYX to XZY so from 0 1 2 to 2 0 1
    assert_array_equal(gathered_array.transpose(forward_tranpose),
                       gathered_jd_xy)
    assert_array_equal(gathered_array.transpose(forward_tranpose),
                        gathered_jax_xy)
    assert_array_equal(gathered_array.transpose(forward_tranpose),
                        mpi4jax_transposed_xy)
    # *********************************************
    # Test Y to Z transpose
    # It tranposes XZY to YXZ so from 0 1 2 to 2 0 1 again
    assert_array_equal(gathered_jd_xy.transpose(forward_tranpose),
                       gathered_jd_yz)
    assert_array_equal(gathered_jax_xy.transpose(forward_tranpose),
                        gathered_jax_yz)
    assert_array_equal(mpi4jax_transposed_xy.transpose(forward_tranpose),
                        mpi4jax_tranposed_yz)
    # and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_array.transpose(double_forward),
                       gathered_jd_yz)
    assert_array_equal(gathered_array.transpose(double_forward),
                        gathered_jax_yz)
    assert_array_equal(gathered_array.transpose(double_forward),
                        mpi4jax_tranposed_yz)
    # *********************************************
    # Test Z to Y transpose
    # It tranposes YXZ to XZY so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_jd_yz.transpose(backward_tranpose),
                       gathered_jd_zy)
    assert_array_equal(gathered_jax_yz.transpose(backward_tranpose),
                        gathered_jax_zy)
    assert_array_equal(mpi4jax_tranposed_yz.transpose(backward_tranpose),
                        mpi4jax_tranposed_zy)
    # The Y pencils should match in forward and backward transposes (despite the inverted grid)
    # assert_array_equal(gathered_jd_zy, gathered_jd_xy)
    # *********************************************
    # Test Y to X transpose
    # It tranposes XZY to ZYX so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_jd_zy.transpose(backward_tranpose),
                       gathered_jd_yx)
    assert_array_equal(gathered_jax_zy.transpose(backward_tranpose),
                        gathered_jax_yx)
    assert_array_equal(mpi4jax_tranposed_zy.transpose(backward_tranpose),
                        mpi4jax_tranposed_yx)
    # The X pencils should match in forward and backward transposes (original array)
    assert_array_equal(gathered_jd_yx, gathered_array)
    assert_array_equal(gathered_jax_yx, gathered_array)
    assert_array_equal(mpi4jax_tranposed_yx, gathered_array)

    print(f"Pdims {pdims} are ok!")
