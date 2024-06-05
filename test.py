from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.lax import with_sharding_constraint
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
jax.distributed.initialize()
# Create a mesh and annotate each axis with a name.
mesh_rows = 2
mesh_cols =  jax.device_count() // 2
mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))


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
  return MLP(64, 256, 64, rngs=rngs)

model = create_model(nnx.Rngs(0))

@nnx.scan
def forward(x, model: MLP):
  x = model(x)
  return x, None

x = jnp.ones((16, 64))
global_shape = (16,64)
arrays = [jax.device_put(x[index], d)
       for d, index in sharding.addressable_devices_indices_map(global_shape).items()]

arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
y, _ = forward(arr, model)

print(f'{y.shape = }')
nnx.display(model)