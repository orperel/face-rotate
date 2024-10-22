import os

training_params = {

    # General params
    'use_cuda': True,

    # Limit how many GPUs are used, even if more are available.
    # If 0 is specified the trainer will use all available GPUs.
    # --NOTE--: Multiple GPUs processing does not improve performance at the moment.
    'force-gpu-count': 1,

    # When only a single gpu is specified, this will be the default gpu index used.
    'default-gpu': 0,

    # Choose the group of data used:
    # - 'enlarged' - faces smaller than 256, scaled up
    # - 'decimated - faces bigger than 256, scaled down
    # - 'all' - use entire dataset regardless of size
    # - 'debug' - small set for debug purposes
    'data_group': 'all',

    # Location of saved dataset
    'dataset_path': "/mnt/data/orperel/dataset" + os.path.sep,

    # Location of saved plots
    'plot_path': "plots" + os.path.sep,

    # Location of output saved models
    'models_path': "/mnt/data/orperel/models" + os.path.sep,

    # Use Spatial Transformer Networks, choose from:
    # 'attention' - For scale, crop, translate
    # 'affine' - For affine transformations
    # None - If no STN should be employed
    'stn': None,

    # SGD parameters
    'batch_size': 32,
    'epochs': 1000,

    # Number of samples per epoch - set 0 for no maximum
    'training_samples_per_epoch': 50000,
    'validation_samples_per_epoch': 50000,

    # Adam optimizer params
    'learning_rate': 0.0002,  # Paper: 0.002,
    'beta1': 0.5,   # Originally: 0.5
    'beta2': 0.999,

    # Maximum gradient norm, clipping occurs for norms larger than this value. 0 performs no gradient clipping.
    'gradient_max_norm': 5,

    # This is lambda_e in the paper, when this parameter is higher, more weight is given to the
    # adversarial component of the loss and less to the L2 loss
    'autoenc_loss_reg': 0.0001,

    # The initial value for lambda_e
    'autoenc_loss_reg_init': 0,

    # Number of SGD iterations required for lambda_e to reach it's final value
    'autoenc_loss_reg_adaption_steps': 500000,  # Originally: 500000,

    # Number of convolution layers in encoder / decoder
    'autoenc_layer_count': 6,

    # If true, the cross-entropy component is added to the AutoEndoer & Discriminator losses.
    # Rotation degrees are treated as discreet one hot vectors with log loss.
    'ypr_quant': False,

    # Determines how many dimensions the one-hot-vector representation of yaw-pitch-roll should contain.
    # The value should divide 180 without residuals.
    # --Used only if ypr_quant is True.
    'deg_dim': 180,

    # If true, the regression component is added to the AutoEncoder & Discriminator losses -
    # Repeat 3 times for yaw, pitch, roll:
    # For the discriminator: loss = loss - log(1 - || deg - deg_predict || **2)
    # For the auto-encoder: loss = loss - log(|| deg - deg_predict || **2)
    # Where deg, deg_predict are one dimensional vectors with [batch_size] dimensions
    'ypr_regress': True,

    # A multiplier for the ypr_regress loss component, to make sure it has the same order of magnitude as
    # the cross entropy component
    'ypr_regress_weight': 1,

    # When true, a random chance of 0.5 is applied to flip images horizontally.
    # Yaw, Roll labels are updated accordingly.
    'h_flip_augment': True
}