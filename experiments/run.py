import argparse
import os
import shutil

import constants
from experiments import maml_config
from runners import sinusoidal_runner, two_layer_teacher_runner
from utils import experiment_logger, experiment_utils

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--config_path",
    default="config.yaml",
    type=str,
    help="path to configuration file of experiment.",
)


def run(args):
    configuration = maml_config.MAMLConfig(config=args.config_path)

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = configuration.results_folder
    experiment_name = configuration.experiment_name or ""
    experiment_path = experiment_utils.get_experiment_path(
        folder=results_folder, timestamp=timestamp, experiment_name=experiment_name
    )

    os.makedirs(name=experiment_path, exist_ok=True)
    config_copy_path = os.path.join(experiment_path, "config.yaml")
    shutil.copyfile(args.config_path, config_copy_path)

    logger = experiment_logger.get_logger(
        experiment_path=experiment_path, name=__name__
    )

    configuration.add_property(constants.EXPERIMENT_PATH, experiment_path)

    if configuration.task_distribution == constants.SINUSOIDAL:
        runner = sinusoidal_runner.SinusoidalRunner(configuration=configuration)
    elif configuration.task_distribution == constants.TWO_LAYER_TEACHERS:
        runner = two_layer_teacher_runner.TwoLayerTeacherRunner(
            configuration=configuration
        )

    runner.train()
    runner.post_process()


if __name__ == "__main__":
    args = arg_parser.parse_args()
    run(args)
