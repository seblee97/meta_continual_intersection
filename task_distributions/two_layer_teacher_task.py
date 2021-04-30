from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from task_distributions import base_task


class TwoLayerTeacherTask(base_task.BaseTask):
    def __init__(
        self,
        input_dimension: int,
        forward_call: Callable,
        parameters: List[Tuple[jnp.ndarray]],
    ):

        self._input_dimension = input_dimension
        self._forward = forward_call
        self._parameters = parameters

        super().__init__()

    def sample_data(self, key, num_datapoints):
        x = jax.random.normal(key=key, shape=(num_datapoints, self._input_dimension))
        y = self._forward(self._parameters, x)
        return x, y

    def __call__(self, x):
        return self._forward(x)
