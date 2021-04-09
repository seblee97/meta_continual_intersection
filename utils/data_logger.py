import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiments import maml_config


class DataLogger:
    """Class for logging experimental data.

    Data can be stored in a csv. # TODO: add TensorBoard event file.
    Experiment monitoring can also be sent to a log file.
    """

    def __init__(self, configuration: maml_config.MAMLConfig):
        self._experiment_path = configuration.experiment_path
        self._logfile_path = os.path.join(self._experiment_path, "data_logger.csv")
        self._df_columns = []
        self._logger_data = {}

    def write_scalar(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dictionary.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.
        """
        if tag not in self._logger_data:
            self._logger_data[tag] = {}
            self._df_columns.append(tag)

        self._logger_data[tag][step] = scalar

    def checkpoint(self) -> None:
        """Construct dataframe from data and merge with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        assert (
            list(self._logger_data.keys()) == self._df_columns
        ), "Incorrect dataframe columns for merging"

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        pd.DataFrame(self._logger_data).to_csv(
            self._logfile_path, mode="a", header=header, index=False
        )

        # reset logger in memory to empty.
        self._df_columns = []
        self._logger_data = {}
