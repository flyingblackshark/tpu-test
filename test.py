import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils,multihost_utils
import numpy as np
from jax.debug import visualize_array_sharding
jax.distributed.initialize()

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()), ['host', 'dev'])
pspecs = jax.sharding.PartitionSpec('host')
test = None
if jax.process_index() == 0:
    test = np.ones((4))
else:
    test = np.zeros((4))
arr = multihost_utils.host_local_array_to_global_array(test,mesh,pspecs)
#arr = multihost_utils.broadcast_one_to_all(test)
arr = jnp.asarray(arr)
visualize_array_sharding(arr)
print(jax.process_index())

arr = multihost_utils.process_allgather(arr)
print(arr is None)
print(arr)