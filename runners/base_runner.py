import abc

import constants
import jax
import jax.numpy as jnp
import pandas as pd
from experiments import maml_config
from model import maml
from utils import data_logger, experiment_logger, plotters


class BaseRunner(abc.ABC):
    def __init__(self, configuration: maml_config.MAMLConfig):

        self.__key = jax.random.PRNGKey(configuration.seed)

        self._task_distribution = self._setup_task_distribution(
            configuration=configuration
        )
        self._model = self._setup_model(configuration=configuration)

        if configuration.log_to_df:
            self._data_logger = data_logger.DataLogger(configuration=configuration)
        self._logger = experiment_logger.get_logger(
            experiment_path=configuration.experiment_path, name=__name__
        )
        self._num_epochs = configuration.num_epochs
        self._num_tasks = configuration.num_tasks
        self._batch_size = configuration.batch_size

        self._test_frequency = configuration.test_frequency
        self._num_evaluations = configuration.num_evaluations
        self._num_examples = configuration.num_examples
        self._num_adaptation_steps = configuration.num_adaptation_steps
        self._plot_evaluations = configuration.plot_evaluations

        self._print_frequency = configuration.print_frequency
        self._log_to_df = configuration.log_to_df
        self._checkpoint_frequency = configuration.checkpoint_frequency
        self._experiment_path = configuration.experiment_path
        self._plot_losses = configuration.plot_losses

    @abc.abstractmethod
    def _setup_task_distribution(self, configuration: maml_config.MAMLConfig):
        pass

    def _get_key(self):
        self.__key, new_key = jax.random.split(self.__key)
        return new_key

    def _setup_model(self, configuration: maml_config.MAMLConfig):
        return maml.MAML(
            key=self._get_key(),
            inner_optimiser_type=configuration.inner_optimiser_type,
            outer_optimiser_type=configuration.outer_optimiser_type,
            inner_lr=configuration.inner_lr,
            outer_lr=configuration.outer_lr,
            task_distribution=self._task_distribution,
            input_dimension=configuration.input_dim,
            network_specification=configuration.network_specification,
        )

    def _get_data_batch_from_tasks(self, tasks, batch_size: int):
        batch_x_inner = []
        batch_y_inner = []
        batch_x_outer = []
        batch_y_outer = []

        for task in tasks:

            x_inner, y_inner = task.sample_data(
                key=self._get_key(), num_datapoints=batch_size
            )
            x_outer, y_outer = task.sample_data(
                key=self._get_key(), num_datapoints=batch_size
            )

            batch_x_inner.append(x_inner)
            batch_y_inner.append(y_inner)
            batch_x_outer.append(x_outer)
            batch_y_outer.append(y_outer)

        return (
            jnp.stack(batch_x_inner),
            jnp.stack(batch_y_inner),
            jnp.stack(batch_x_outer),
            jnp.stack(batch_y_outer),
        )

    def train(self):

        for i in range(self._num_epochs):

            if i % self._test_frequency == 0 and i != 0:
                self._test(i)

            task_sample = self._task_distribution.sample(
                key=self._get_key(), num_tasks=self._num_tasks
            )

            (
                batch_x_inner,
                batch_y_inner,
                batch_x_outer,
                batch_y_outer,
            ) = self._get_data_batch_from_tasks(
                tasks=task_sample, batch_size=self._batch_size
            )

            outer_optimiser_state, meta_loss = self._model.step(
                epoch=i,
                outer_optimiser_state=self._model.outer_optimiser_state,
                inner_x=batch_x_inner,
                inner_y=batch_y_inner,
                outer_x=batch_x_outer,
                outer_y=batch_y_outer,
            )

            self._model.outer_optimiser_state = outer_optimiser_state

            if self._log_to_df:
                self._data_logger.write_scalar(
                    tag=constants.META_LOSS, step=i, scalar=float(meta_loss)
                )

            if i % self._print_frequency == 0:
                self._logger.info(f"{i}: {meta_loss}")

            if i % self._checkpoint_frequency == 0 and i != 0:
                self._data_logger.checkpoint()

    @abc.abstractmethod
    def _test(self, step: int):
        pass

    @abc.abstractmethod
    def _post_process(self):
        pass

    def post_process(self):
        if self._plot_losses:
            df = pd.read_csv(self._data_logger.df_path)
            plotters.plot_losses(
                losses=df[constants.META_LOSS].dropna().to_numpy(),
                save_folder=self._experiment_path,
            )

        self._post_process()
