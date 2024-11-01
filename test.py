import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils,multihost_utils
import numpy as np
from jax.debug import visualize_array_sharding
jax.distributed.initialize()


# 定义 sharding 策略（例如在 2 个设备上分配）
device_mesh = mesh_utils.create_device_mesh((16,1))
mesh = jax.sharding.Mesh(device_mesh, ['host', 'dev'])
pspecs = jax.sharding.PartitionSpec('host')
test = None
if jax.process_index() == 0:
    test = np.arange(4)

arr = multihost_utils.host_local_array_to_global_array(test, mesh, pspecs)  

visualize_array_sharding(arr)