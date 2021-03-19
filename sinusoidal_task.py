import base_task
import jax
import jax.numpy as jnp


class SinusoidTask(base_task.BaseTask):
    def __init__(self, x_range, amplitude, phase):
        self._x_range = x_range
        self._amplitude = amplitude
        self._phase = phase
        super().__init__()

    def sample_data(self, key, num_datapoints):
        x = jax.random.uniform(
            key=key,
            shape=(num_datapoints, 1),
            minval=self._x_range[0],
            maxval=self._x_range[1],
        )
        y = self._amplitude * jnp.sin(x + self._phase)
        return x, y

    def __call__(self, x):
        return self._amplitude * jnp.sin(x + self._phase)
