# %%
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

# %%
def concordant_pairs(hazards, current_hazard):
        all_pairs = jnp.sum(hazards != 0)
        concordant = jnp.sum(jnp.where(hazards != 0, hazards > current_hazard, 0))
        tied = jnp.sum(jnp.where(hazards != 0, hazards == current_hazard, 0))

        return concordant + tied / 2, all_pairs


# %%
@jit
def c_index(hazards: jnp.ndarray, times: jnp.ndarray, events: jnp.ndarray) -> float:
    def order_inputs(hazards: jnp.ndarray, times: jnp.ndarray, events: jnp.ndarray): 
        time_order = times.argsort()
        return hazards[time_order], events[time_order]
    
    hazards, events = order_inputs(hazards, times, events)
    hazards = jnp.outer(events, hazards)
    hazards_matrix = jnp.triu(hazards, 1)
    concordant, all = vmap(concordant_pairs)(hazards_matrix, hazards.diagonal())

    return jnp.sum(concordant) / (jnp.sum(all) + 1e-9)

# %%
# # TODO: move to tests
# times = np.random.rand(50000)
# events = np.random.rand(50000) > 0.2
# print(c_index(times, times, events), c_index(-times, times, events), c_index(jnp.ones_like(times), times, events))

# # %%
# assert jnp.allclose(c_index(times, times, events), 1)
# assert jnp.allclose(c_index(-times, times, events), 0)
# assert c_index(jnp.ones_like(times), times, events) == 0.5
# %%
