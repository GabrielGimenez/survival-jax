import jax
import jax.numpy as jnp
import jax.lax as lax

@jax.jit
def concordant_pairs(hazards, current_hazard):
        all_pairs = len(hazards)
        concordant = jnp.sum(hazards < current_hazard)
        tied = jnp.sum(hazards == current_hazard)

        return concordant + tied / 2, all_pairs

def concordance_index(hazards, times, events):

    assert len(hazards) == len(times) == len(events)
    all_samples = len(hazards)
    time_order = times.argsort()

    times = times[time_order]
    hazards = hazards[time_order]
    events = events[time_order]

    all_pairs = 0
    concordant = 0

    for i in range(all_samples):
        if events[i]:
            concord, pairs = concordant_pairs(hazards[i + 1:], hazards[i])

            concordant += concord
            all_pairs += pairs

    return concordant / all_pairs


def concordance_generator(hazards):
    @jax.jit
    def concordance(current, hazard):
        
        current_concordance = current["concordant"]
        new_concordants = current_concordance + jnp.sum(hazards < hazard) + (jnp.sum(hazards == hazard) *  0.5)
        
        new_pairs = current["all"] + len(hazards)
        new = {"concordant": new_concordants, "all": new_pairs}
        return new, current
    return concordance