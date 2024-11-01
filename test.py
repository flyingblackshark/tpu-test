import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
# 定义目标数组的形状
shape = (16,4)

# 定义 sharding 策略（例如在 2 个设备上分配）
device_mesh = mesh_utils.create_device_mesh((16,))
#mesh = Mesh(devices=jax.devices(), axis_names=('data'))
x_sharding = NamedSharding(device_mesh,PartitionSpec('data'))

# 假设两个设备上各自持有 4x2 的数据
arrays = [jnp.ones((4,4)),jnp.ones((4,4)),jnp.ones((4,4)),jnp.ones((4,4))]

# 使用 make_array_from_single_device_arrays 来构造跨设备数组
distributed_array = jax.make_array_from_single_device_arrays(shape, x_sharding, arrays)

print(distributed_array)