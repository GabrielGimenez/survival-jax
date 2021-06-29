import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit

def kaplan_meier(times, events):
    death_t = death_times(times, events)
    hz = hazard_generator(times, events)
    km = km_generator(hz)
    _, survivals = lax.scan(km, init=1.0, xs=death_t)
    return survivals, death_t

def km_generator(hazards):
    @jit
    def _km(prev_survival, death_time):
        new_survival = prev_survival * (1 - hazards(death_time))
        return new_survival, prev_survival
    return _km

def hazard_generator(times, events):
    @jit
    def hazards(e_times):
        in_risk = jnp.sum(jnp.where(times > e_times, 1, 0))  # em risco
        e = jnp.nansum(jnp.where(times == e_times, events, jnp.nan))  # ocorreu evento
        return e / in_risk
    return hazards


def death_times(times, events):
    # Nonzero is equivalent to np where
    return jnp.unique(jnp.take(times, events.nonzero()))
