import os
from trainer import FaderNetTrainer

training_params = {

    # General params
    'use_cuda': True,

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
    'models_path': "models" + os.path.sep,

    # SGD parameters
    'batch_size': 32,
    'epochs': 1000,

    # Adam optimizer params
    'learning_rate': 0.002,
    'beta1': 0.5,
    'beta2': 0.999,

    # Maximum gradient norm, clipping occurs for norms larger than this value. 0 performs no gradient clipping.
    'gradient_max_norm': 5,

    # This is lambda_e in the paper, when this parameter is higher, more weight is given to the
    # adversarial component of the loss and less to the L2 loss
    'autoenc_loss_reg': 0.0001,

    # The initial value for lambda_e
    'autoenc_loss_reg_init': 0,

    # Number of SGD iterations required for lambda_e to reach it's final value
    'autoenc_loss_reg_adaption_steps': 500000,

    # Number of convolution layers in encoder / decoder
    'autoenc_layer_count': 7,

    # Change how yaw-pitch-roll should be handled:
    # - True means 'discreet' - Rotation degrees are treated as discreet one hot vectors with loss loss.
    # - False means 'continuous' - Rotation degrees are handled as a regression problem
    'ypr_quant': True,

    # Determines how many dimensions the one-hot-vector representation of yaw-pitch-roll should contain.
    # The value should divide 180 without residuals.
    # --Used only if ypr_quant is True.
    'deg_dim': 45,

    # When true, a random chance of 0.5 is applied to flip images horizontally.
    # Yaw, Roll labels are updated accordingly.
    'h_flip_augment': False
}

trainer = FaderNetTrainer(training_params)
trainer.train()
