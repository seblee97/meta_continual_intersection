import constants
from config_manager import config_field, config_template


class MAMLConfigTemplate:

    _sinusoidal_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.X_MIN, types=[int, float]),
            config_field.Field(name=constants.X_MAX, types=[int, float]),
            config_field.Field(name=constants.AMPLITUDE_MIN, types=[int, float]),
            config_field.Field(name=constants.AMPLITUDE_MAX, types=[int, float]),
            config_field.Field(name=constants.PHASE_MIN, types=[int, float]),
            config_field.Field(name=constants.PHASE_MAX, types=[int, float]),
        ],
        dependent_variables=[constants.TASK_DISTRIBUTION],
        dependent_variables_required_values=[[constants.SINUSOIDAL]],
        level=[constants.SINUSOIDAL],
    )

    _training_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_EPOCHS, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.NUM_TASKS, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.BATCH_SIZE, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.INNER_LR,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.OUTER_LR,
                types=[int, float],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.INNER_OPTIMISER_TYPE,
                types=[str],
                requirements=[lambda x: x in [constants.SGD, constants.ADAM]],
            ),
            config_field.Field(
                name=constants.OUTER_OPTIMISER_TYPE,
                types=[str],
                requirements=[lambda x: x in [constants.SGD, constants.ADAM]],
            ),
        ],
        level=[constants.TRAINING],
    )

    _testing_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.NUM_EVALUATIONS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NUM_EXAMPLES,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.NUM_ADAPTATION_STEPS,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.PLOT_EVALUATIONS,
                types=[bool],
            ),
        ],
        level=[constants.TESTING],
    )

    _logging_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.RESULTS_FOLDER, types=[str]),
            config_field.Field(name=constants.PRINT_FREQUENCY, types=[int]),
            config_field.Field(name=constants.LOG_TO_DF, types=[bool]),
            config_field.Field(
                name=constants.CHECKPOINT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        level=[constants.LOGGING],
    )

    _model_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.INPUT_DIM, types=[int]),
            config_field.Field(name=constants.NETWORK_SPECIFICATION, types=[list]),
        ],
        level=[constants.MODEL],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.EXPERIMENT_NAME, types=[str, type(None)]),
            config_field.Field(name=constants.SEED, types=[int]),
            config_field.Field(
                name=constants.TASK_DISTRIBUTION,
                types=[str],
                requirements=[
                    lambda x: x in [constants.SINUSOIDAL, constants.TWO_LAYER_TEACHERS]
                ],
            ),
        ],
        nested_templates=[
            _sinusoidal_template,
            _training_template,
            _testing_template,
            _logging_template,
            _model_template,
        ],
    )
