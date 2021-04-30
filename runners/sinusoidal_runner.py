import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from experiments import maml_config
from runners import base_runner
from task_distributions import sinusoidal_task_distribution


class SinusoidalRunner(base_runner.BaseRunner):
    def __init__(self, configuration: maml_config.MAMLConfig):

        self._x_min = configuration.x_min
        self._x_max = configuration.x_max

        self._amplitude_min = configuration.amplitude_min
        self._amplitude_max = configuration.amplitude_max

        self._phase_min = configuration.phase_min
        self._phase_max = configuration.phase_max

        super().__init__(configuration=configuration)

    def _setup_task_distribution(self, configuration: maml_config.MAMLConfig):

        task_distribution = sinusoidal_task_distribution.SinusoidalTaskDistribution(
            x_range=(self._x_min, self._x_max),
            amplitude_range=(self._amplitude_min, self._amplitude_max),
            phase_range=(self._phase_min, self._phase_max),
        )

        return task_distribution

    def _test(self, step: int):
        evaluation_tasks = self._task_distribution.sample(
            key=self._get_key(), num_tasks=self._num_evaluations
        )

        trained_parameters = self._model.outer_get_parameters(
            self._model.outer_optimiser_state
        )

        for i, task in enumerate(evaluation_tasks):
            x, y = task.sample_data(
                key=self._get_key(), num_datapoints=self._num_examples
            )
            adapted_parameters = self._model.fine_tune(
                trained_parameters, x, y, self._num_adaptation_steps
            )

            if self._plot_evaluations:
                plot_path = os.path.join(self._experiment_path, f"{i}_test.pdf")
                self._plot_evaluation(x, y, task, adapted_parameters, plot_path)

    def _plot_evaluation(self, x, y, task, adapted_parameters, save_name):
        fig = plt.figure()
        plt.scatter(x, y)

        x_range = jnp.linspace(self._x_min, self._x_max, 100).reshape(-1, 1)

        plt.plot(x_range, task(x_range), label="ground truth")
        for i, parameters in enumerate(adapted_parameters):
            regression = self._model.network_forward(parameters, x_range)
            if i == 0 or i > 98:
                plt.plot(x_range, regression, label=f"{i} tuning")

        plt.legend()
        fig.savefig(save_name)

    def _post_process(self):
        pass
