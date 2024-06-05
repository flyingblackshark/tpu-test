from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

# Create a mesh and annotate each axis with a name.
device_mesh = mesh_utils.create_device_mesh((4, 4))
print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
print(mesh)

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)