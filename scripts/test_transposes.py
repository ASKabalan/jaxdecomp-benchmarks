import jax

jax.distributed.initialize()
from mpi4py import MPI

rank = jax.process_index()
size = jax.process_count()
import jax.numpy as jnp

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from jax.experimental.shard_map import shard_map
from jax import lax

import pytest
from math import prod

from jaxdecomp import (transposeXtoY, transposeYtoX, transposeYtoZ,
                       transposeZtoY)
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

decomp = [pencil_1]
global_shapes = [(4, 4, 4)]



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
    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
    def jax_transposeXtoY(data):
        return lax.all_to_all(data, 'y', 2, 1, tiled=True).transpose([2 , 0 , 1])

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoZ(x):
        return lax.all_to_all(x, 'z', 2, 1, tiled=True).transpose([2 , 0 , 1])

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('y', 'z'))
    def jax_transposeZtoY(x):
        return lax.all_to_all(x, 'z', 2, 0, tiled=True).transpose([1, 2 ,0])

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=P('y', 'z'), out_specs=P('z', 'y'))
    def jax_transposeYtoX(x):
        return lax.all_to_all(x, 'y', 2, 0, tiled=True).transpose([1, 2 ,0])



    with mesh:
        print(f"Running JD")
        # JD transposes
        jd_tranposed_xy = transposeXtoY(global_array)
        jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
        jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
        jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)
        # JAX transposes
        print(f"Running JAX")
        jax_transposed_xy = jax_transposeXtoY(global_array)
        jax_tranposed_yz = jax_transposeYtoZ(jax_transposed_xy)
        jax_tranposed_zy = jax_transposeZtoY(jax_tranposed_yz)
        jax_tranposed_yx = jax_transposeYtoX(jax_tranposed_zy)


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

    # is gathered_jax_xy == gathered_array with a transpose


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
    # *********************************************
    # Test Y to Z transpose
    # It tranposes XZY to YXZ so from 0 1 2 to 2 0 1 again
    assert_array_equal(gathered_jd_xy.transpose(forward_tranpose),
                       gathered_jd_yz)
    assert_array_equal(gathered_jax_xy.transpose(forward_tranpose),
                       gathered_jax_yz)
    # and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_array.transpose(double_forward),
                       gathered_jd_yz)
    assert_array_equal(gathered_array.transpose(double_forward),
                       gathered_jax_yz)
    # *********************************************
    # Test Z to Y transpose
    # It tranposes YXZ to XZY so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_jd_yz.transpose(backward_tranpose),
                       gathered_jd_zy)
    assert_array_equal(gathered_jax_yz.transpose(backward_tranpose),
                       gathered_jax_zy)
    # The Y pencils should match in forward and backward transposes (despite the inverted grid)
    # assert_array_equal(gathered_jd_zy, gathered_jd_xy)
    # *********************************************
    # Test Y to X transpose
    # It tranposes XZY to ZYX so from 0 1 2 to 1 2 0
    assert_array_equal(gathered_jd_zy.transpose(backward_tranpose),
                       gathered_jd_yx)
    assert_array_equal(gathered_jax_zy.transpose(backward_tranpose),
                       gathered_jax_yx)
    # The X pencils should match in forward and backward transposes (original array)
    assert_array_equal(gathered_jd_yx, gathered_array)
    assert_array_equal(gathered_jax_yx, gathered_array)

    print(f"Pdims {pdims} are ok!")
