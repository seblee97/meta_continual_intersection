import functools

import constants
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.experimental import optimizers, stax
from task_distributions import sinusoidal_task_distribution


class MAML:
    def __init__(
        self,
        key,
        inner_optimiser_type: str,
        outer_optimiser_type: str,
        inner_lr: float,
        outer_lr: float,
        task_distribution,
        input_dimension: int,
        network_specification,
    ):

        self._key = key
        self._task_distribution = task_distribution

        def setup_outer_optimiser(optimiser_type: str, lr: float):
            if optimiser_type == constants.ADAM:
                init, update, get_params = optimizers.adam(step_size=lr)

            return init, update, get_params

        (
            outer_optimiser_initialiser,
            outer_optimiser_update,
            outer_get_parameters,
        ) = setup_outer_optimiser(optimiser_type=outer_optimiser_type, lr=outer_lr)

        network_forward, network_parameters = self._setup_network(
            input_dimension=input_dimension, network_specification=network_specification
        )

        def loss_function(parameters, inputs, labels):
            predictions = network_forward(parameters, inputs)
            return jnp.mean((predictions - labels) ** 2)

        def inner_loop(parameters, x, y):
            gradients = jax.grad(loss_function)(parameters, x, y)
            if False:
                # get inner loop optimiser
                (
                    inner_optimiser_initialiser,
                    inner_optimiser_update,
                    inner_get_parameters,
                ) = setup_optimiser(optimiser_type="adam", lr=0.01)

                inner_optimiser_state = inner_optimiser_initialiser(parameters)

                updated_optimiser_state = inner_optimiser_update(
                    0, gradients, inner_optimiser_state
                )
                updated_parameters = inner_get_parameters(updated_optimiser_state)

            else:
                # return updated_parameters
                inner_sgd_fn = lambda g, state: (state - 0.01 * g)
                updated_parameters = jax.tree_multimap(
                    inner_sgd_fn, gradients, parameters
                )

            return updated_parameters

        def compute_meta_loss(parameters, x_inner, y_inner, x_outer, y_outer):
            updated_parameters = inner_loop(parameters=parameters, x=x_inner, y=y_inner)
            loss = loss_function(updated_parameters, x_outer, y_outer)
            return loss

        def compute_batch_meta_loss(
            parameters, batch_x_inner, batch_y_inner, batch_x_outer, batch_y_outer
        ):
            task_losses = jax.vmap(functools.partial(compute_meta_loss, parameters))(
                batch_x_inner, batch_y_inner, batch_x_outer, batch_y_outer
            )
            batch_meta_loss = jnp.mean(task_losses)
            return batch_meta_loss

        def step(epoch: int, outer_optimiser_state, inner_x, inner_y, outer_x, outer_y):
            parameters = outer_get_parameters(outer_optimiser_state)
            batch_meta_loss, gradients = jax.value_and_grad(compute_batch_meta_loss)(
                parameters, inner_x, inner_y, outer_x, outer_y
            )
            return (
                outer_optimiser_update(epoch, gradients, outer_optimiser_state),
                batch_meta_loss,
            )

        self._outer_optimiser_state = outer_optimiser_initialiser(network_parameters)
        self._inner_loop = inner_loop
        self.network_forward = network_forward
        self.outer_get_parameters = outer_get_parameters
        self.step = jax.jit(step)

    @property
    def outer_optimiser_state(self):
        return self._outer_optimiser_state

    @outer_optimiser_state.setter
    def outer_optimiser_state(self, state):
        self._outer_optimiser_state = state

    def _setup_network(self, input_dimension, network_specification):

        layers = []
        for layer_specification in network_specification:

            layer_type = list(layer_specification.keys())[0]
            layer_info = list(layer_specification.values())[0]

            if layer_type == constants.FC:
                layer = stax.Dense(layer_info[constants.OUTPUT_DIM])

            layers.append(layer)

            activation_type = layer_info.get(constants.ACTIVATION)
            if activation_type is not None:
                if activation_type == constants.RELU:
                    activation = stax.Relu

                layers.append(activation)

        init, forward = stax.serial(*layers)
        _, params = init(self._key, (-1, input_dimension))

        return forward, params

    def fine_tune(self, parameters, x, y, adaptation_steps):
        fine_tuned_parameters = []

        updated_parameters = parameters
        fine_tuned_parameters.append(updated_parameters)

        for i in range(adaptation_steps):
            updated_parameters = self._inner_loop(updated_parameters, x, y)
            fine_tuned_parameters.append(updated_parameters)
        return fine_tuned_parameters

    # def _get_data_batch_from_tasks(self, tasks, batch_size: int):
    #     batch_x_inner = []
    #     batch_y_inner = []
    #     batch_x_outer = []
    #     batch_y_outer = []

    #     for task in tasks:

    #         x_inner, y_inner = task.sample_data(
    #             key=self._key, num_datapoints=batch_size
    #         )
    #         x_outer, y_outer = task.sample_data(
    #             key=self._key, num_datapoints=batch_size
    #         )

    #         batch_x_inner.append(x_inner)
    #         batch_y_inner.append(y_inner)
    #         batch_x_outer.append(x_outer)
    #         batch_y_outer.append(y_outer)

    #     return (
    #         jnp.stack(batch_x_inner),
    #         jnp.stack(batch_y_inner),
    #         jnp.stack(batch_x_outer),
    #         jnp.stack(batch_y_outer),
    #     )


#     def train(self, epochs: int, num_tasks: int, batch_size: int):

#         meta_losses = []

#         for i in range(epochs):

#             task_sample = self._task_distribution.sample(
#                 key=self._key, num_tasks=num_tasks
#             )

#             (
#                 batch_x_inner,
#                 batch_y_inner,
#                 batch_x_outer,
#                 batch_y_outer,
#             ) = self._get_data_batch_from_tasks(
#                 tasks=task_sample, batch_size=batch_size
#             )

#             self._optimiser_state, meta_loss = self._step(
#                 epoch=i,
#                 optimiser_state=self._optimiser_state,
#                 inner_x=batch_x_inner,
#                 inner_y=batch_y_inner,
#                 outer_x=batch_x_outer,
#                 outer_y=batch_y_outer,
#             )

#             meta_losses.append(meta_loss)

#             if i % 100 == 0:
#                 print(f"{i}: {meta_loss}")
#         return meta_losses

#     def test(
#         self,
#         num_evaluations: int,
#         num_examples: int,
#         num_adaptation_steps: int,
#         plot: bool,
#     ):
#         evaluation_tasks = self._task_distribution.sample(
#             key=self._key, num_tasks=num_evaluations
#         )

#         trained_parameters = self._get_parameters(self._optimiser_state)

#         for i, task in enumerate(evaluation_tasks):
#             x, y = task.sample_data(key=self._key, num_datapoints=num_examples)
#             adapted_parameters = self._fine_tune(
#                 trained_parameters, x, y, num_adaptation_steps
#             )

#             if plot:
#                 self._plot_evaluation(x, y, task, adapted_parameters, f"{i}_test.pdf")

#     def _plot_evaluation(self, x, y, task, adapted_parameters, save_name):
#         fig = plt.figure()
#         plt.scatter(x, y)

#         x_range = jnp.linspace(-5, 5, 100).reshape(-1, 1)

#         plt.plot(x_range, task(x_range), label="ground truth")
#         for i, parameters in enumerate(adapted_parameters):
#             regression = self._network_forward(parameters, x_range)
#             plt.plot(x_range, regression, label=f"{i} tuning")

#         plt.legend()
#         fig.savefig(save_name)


# if __name__ == "__main__":
#     task_distribution = sinusoidal_task_distribution.SinusoidalTaskDistribution(
#         x_range=(-5, 5), amplitude_range=(0.1, 5), phase_range=(0, jnp.pi)
#     )
#     network_specification = {
#         "input_dim": 1,
#         "layer_specifications": [
#             {"linear": {"output_dim": 40, "activation": "relu"}},
#             {"linear": {"output_dim": 40, "activation": "relu"}},
#             {"linear": {"output_dim": 1}},
#         ],
#     }
#     rng = jax.random.PRNGKey(0)

#     maml = MAML(
#         key=rng,
#         task_distribution=task_distribution,
#         optimiser_type="adam",
#         lr=0.001,
#         network_specification=network_specification,
#     )

#     meta_losses = maml.train(10000, 5, 5)

#     fig = plt.figure()
#     plt.plot(range(len(meta_losses)), meta_losses)
#     plt.xlabel("epochs")
#     plt.ylabel("meta loss")
#     fig.savefig("losses.pdf")

#     maml.test(5, 10, 5, True)
