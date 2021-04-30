from typing import Tuple, Union

import constants
import jax
import jax.numpy as jnp
from jax.experimental import stax
from task_distributions import base_task_distribution, two_layer_teacher_task
from utils import custom_activations


class TwoLayerTeacherTaskDistribution(base_task_distribution.BaseTaskDistribution):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: str,
        max_rotation: float,
        key,
    ):
        self._input_dimension = input_dimension
        self._hidden_dimension = hidden_dimension
        self._output_dimension = output_dimension
        self._activation = activation
        self._max_rotation = max_rotation
        (
            self._network_forward,
            self._base_network_parameters,
        ) = self._initialise_base_network(key=key)
        super().__init__()

    def _initialise_base_network(self, key):

        input_to_hidden_layer = stax.Dense(self._hidden_dimension)
        hidden_to_output_layer = stax.Dense(self._output_dimension)

        if self._activation == constants.RELU:
            activation_fn = stax.Relu
        elif self._activation == constants.SIGMOIDAL:
            activation_fn = custom_activations.ScaledErf
        else:
            raise ValueError(f"Activation type {self._activation} not recognised.")

        layers = [input_to_hidden_layer, activation_fn, hidden_to_output_layer]

        init, forward = stax.serial(*layers)
        _, params = init(key, (-1, self._input_dimension))

        return forward, params

    def _rotate_parameters(self, rotation: float):
        # import pdb

        # pdb.set_trace()

        return self._base_network_parameters

    def sample(self, key, num_tasks: int):
        rotation_samples = jax.random.uniform(
            key=key,
            shape=(num_tasks,),
            minval=0.0,
            maxval=self._max_rotation,
        )
        tasks = [
            two_layer_teacher_task.TwoLayerTeacherTask(
                input_dimension=self._input_dimension,
                forward_call=self._network_forward,
                parameters=self._rotate_parameters(rotation),
            )
            for rotation in rotation_samples
        ]
        return tasks
