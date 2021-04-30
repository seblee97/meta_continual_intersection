import os

import constants
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from experiments import maml_config
from runners import base_runner
from task_distributions import two_layer_teacher_task_distribution


class TwoLayerTeacherRunner(base_runner.BaseRunner):
    def __init__(self, configuration: maml_config.MAMLConfig):

        self._input_dimension = configuration.two_layer_teachers_input_dim
        self._hidden_dimension = configuration.hidden_dimension
        self._output_dimension = configuration.output_dim
        self._activation = configuration.two_layer_teachers_activation
        self._max_rotation = configuration.max_rotation

        super().__init__(configuration=configuration)

        self._fixed_evaluation_tasks = self._get_evaluation_tasks()

    def _setup_task_distribution(self, configuration: maml_config.MAMLConfig):

        task_distribution = (
            two_layer_teacher_task_distribution.TwoLayerTeacherTaskDistribution(
                input_dimension=self._input_dimension,
                hidden_dimension=self._hidden_dimension,
                output_dimension=self._output_dimension,
                activation=self._activation,
                max_rotation=self._max_rotation,
                key=self._get_key(),
            )
        )

        return task_distribution

    def _get_evaluation_tasks(self):
        evaluation_tasks = self._task_distribution.sample(
            key=self._get_key(), num_tasks=self._num_evaluations
        )
        return evaluation_tasks

    def _test(self, step: int):

        trained_parameters = self._model.outer_get_parameters(
            self._model.outer_optimiser_state
        )

        generalisation_errors = []

        for i, task in enumerate(self._fixed_evaluation_tasks):
            x, y = task.sample_data(
                key=self._get_key(), num_datapoints=self._num_examples
            )
            adapted_parameters = self._model.fine_tune(
                trained_parameters, x, y, self._num_adaptation_steps
            )
            test_x, test_y = task.sample_data(key=self._get_key(), num_datapoints=10000)

            test_y_predictions = self._model.network_forward(
                adapted_parameters[-1], test_x
            )
            generalisation_error = jnp.mean((test_y_predictions - test_y) ** 2)

            self._data_logger.write_scalar(
                f"{constants.GENERALISATION_ERROR}_{i}",
                step,
                float(generalisation_error),
            )

            generalisation_errors.append(float(generalisation_error))

        self._data_logger.write_scalar(
            constants.MEAN_GENERALISATION_ERROR,
            step,
            onp.mean(generalisation_errors),
        )

    def _post_process(self):

        df = pd.read_csv(self._data_logger.df_path)

        fig = plt.figure()
        for i in range(self._num_evaluations):
            error_i = df[f"{constants.GENERALISATION_ERROR}_{i}"].dropna().to_numpy()
            plt.plot(range(len(error_i)), error_i)
        fig.savefig(
            os.path.join(self._experiment_path, constants.GENERALISATION_ERROR_PLOT),
            dpi=100,
        )
        plt.close()

        fig = plt.figure()
        mean_error = df[constants.MEAN_GENERALISATION_ERROR].dropna().to_numpy()
        plt.plot(range(len(mean_error)), mean_error)
        fig.savefig(
            os.path.join(
                self._experiment_path, constants.MEAN_GENERALISATION_ERROR_PLOT
            ),
            dpi=100,
        )
