from typing import Tuple, Union

import base_task_distribution
import jax
import jax.numpy as jnp
import sinusoidal_task


class TwoLayerTeacherTaskDistribution(base_task_distribution.BaseTaskDistribution):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        max_rotation: float,
    ):
        self._max_rotation = max_rotation
        super().__init__()

    def sample(self, key, num_tasks: int):
        rotation_samples = jax.random.uniform(
            key=key,
            shape=(num_tasks,),
            minval=0.0,
            maxval=self._max_rotation,
        )
        tasks = [
            sinusoidal_task.SinusoidTask(x_range=self._x_range, amplitude=a, phase=p)
            for a, p in zip(amplitude_samples, phase_samples)
        ]
        return tasks
