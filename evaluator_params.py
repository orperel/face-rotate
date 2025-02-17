import os

evaluating_params = {

    # General params
    'use_cuda': True,

    # Limit how many GPUs are used, even if more are available.
    # If 0 is specified the trainer will use all available GPUs.
    # --NOTE--: Multiple GPUs processing does not improve performance at the moment.
    'force-gpu-count': 1,

    # Training, Validation, Test..
    'data_type': 'validation',

    # Choose the group of data used:
    # - 'enlarged' - faces smaller than 256, scaled up
    # - 'decimated - faces bigger than 256, scaled down
    # - 'all' - use entire dataset regardless of size
    # - 'debug' - small set for debug purposes
    'data_group': 'debug',

    # Location of saved dataset
    'dataset_path': "dataset" + os.path.sep,

    # Location of saved plots
    'plot_path': "plots" + os.path.sep,

    # Location of output saved models
    'models_path': "models" + os.path.sep,

    # If true, the cross-entropy component is added to the AutoEndoer & Discriminator losses.
    # Rotation degrees are treated as discreet one hot vectors with log loss.
    'ypr_quant': True,

    # Determines how many dimensions the one-hot-vector representation of yaw-pitch-roll should contain.
    # The value should divide 180 without residuals.
    # --Used only if ypr_quant is True.
    'deg_dim': 45,

    # If true, the regression component is added to the AutoEncoder & Discriminator losses -
    # Repeat 3 times for yaw, pitch, roll:
    # For the discriminator: loss = loss - log(1 - || deg - deg_predict || **2)
    # For the auto-encoder: loss = loss - log(|| deg - deg_predict || **2)
    # Where deg, deg_predict are one dimensional vectors with [batch_size] dimensions
    'ypr_regress': True,

    # When true, a random chance of 0.5 is applied to flip images horizontally.
    # Yaw, Roll labels are updated accordingly.
    'h_flip_augment': True
}