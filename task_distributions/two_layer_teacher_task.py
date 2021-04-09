import base_task
import jax
import jax.numpy as jnp


class TwoLayerTeacherTask(base_task.BaseTask):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: str,
    ):

        self._network_forward, self._network_parameters = self._setup_network(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            output_dimension=output_dimension,
            activation=activation,
        )

        super().__init__()

    def _setup_network(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        activation: str,
    ):
        layers = []

        layer_1 = stax.Dense(hidden_dimension)
        layer_2 = stax.Dense(output_dimension)

        if activation == constants.RELU:
            layer_2_pre_activation = stax.Relu
        else:
            raise ValueError("ReLU activation")

        layers.append(layer_1)
        layers.append(layer_2_pre_activation)
        layers.append(layer_2)

        init, forward = stax.serial(*layers)
        _, params = init(self._key, (-1, input_dimension))

        return forward, params

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
        return self._network_forward(x)
