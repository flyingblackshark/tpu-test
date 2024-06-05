from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial
# Create a mesh and annotate each axis with a name.
device_mesh = mesh_utils.create_device_mesh((4, 4))
#print(device_mesh)

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
#print(mesh)

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
    return self.linear2(x)
  
@partial(nnx.vmap, axis_size=5)
def create_model(rngs: nnx.Rngs):
  return MLP(10, 32, 10, rngs=rngs)

model = create_model(nnx.Rngs(0))

@nnx.scan
def forward(x, model: MLP):
  x = model(x)
  return x, None

x = jnp.ones((3, 10))
x_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
x = jax.device_put(x, x_sharding)
jax.debug.visualize_array_sharding(x)
y, _ = forward(x, model)

print(f'{y.shape = }')
nnx.display(model)