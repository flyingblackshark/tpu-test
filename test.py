import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils,multihost_utils
import numpy as np
jax.distributed.initialize()
# 定义目标数组的形状
shape = (8,4)

# 定义 sharding 策略（例如在 2 个设备上分配）
device_mesh = mesh_utils.create_device_mesh((8,))
mesh = Mesh(device_mesh, axis_names=('data'))
x_sharding = NamedSharding(mesh,PartitionSpec('data'))
pspecs = PartitionSpec('data')
test = None
if jax.process_index() == 0:
    test = np.arange(4)
arr = multihost_utils.host_local_array_to_global_array(test, mesh, pspecs)  

print(arr)