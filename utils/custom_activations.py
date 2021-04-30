import jax.numpy as jnp
from jax.scipy import special


def scaled_elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs / jnp.sqrt(2), **fun_kwargs)
    return init_fun, apply_fun


ScaledErf = scaled_elementwise(special.erf)
