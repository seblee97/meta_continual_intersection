from typing import Tuple, Union

import base_task_distribution
import jax
import jax.numpy as jnp
import sinusoidal_task


class SinusoidalTaskDistribution(base_task_distribution.BaseTaskDistribution):
    def __init__(
        self,
        x_range: Tuple[Union[float, int], Union[float, int]],
        amplitude_range: Tuple[Union[float, int], Union[float, int]],
        phase_range: Tuple[Union[float, int], Union[float, int]],
    ):
        self._x_range = x_range
        self._amplitude_range = amplitude_range
        self._phase_range = phase_range
        super().__init__()

    def sample(self, key, num_tasks: int):
        amplitude_samples = jax.random.uniform(
            key=key,
            shape=(num_tasks,),
            minval=self._amplitude_range[0],
            maxval=self._amplitude_range[1],
        )
        phase_samples = jax.random.uniform(
            key=key,
            shape=(num_tasks,),
            minval=self._phase_range[0],
            maxval=self._phase_range[1],
        )
        tasks = [
            sinusoidal_task.SinusoidTask(x_range=self._x_range, amplitude=a, phase=p)
            for a, p in zip(amplitude_samples, phase_samples)
        ]
        return tasks
