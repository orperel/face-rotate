from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_initialization(modules):
    for m in modules:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal(m.weight.data, gain=0.02)
            nn.init.constant(m.bias.data, 0.0)


class SpatialTransformerNetwork(nn.Module):

    class Transform(Enum):
        ATTENTION = {'params_count': 4, 'dim': (2, 3), 'initial': [1, 0, 0, 0, 1, 0]},
        AFFINE = {'params_count': 6, 'dim': (2, 3), 'initial': [1, 0, 0, 0, 1, 0]}

    def __init__(self, in_channels, transform=Transform.AFFINE):
        super(SpatialTransformerNetwork, self).__init__()

        self.transform_params = transform

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Regression for the transformation matrix
        self.localization_regress = nn.Sequential(
            nn.Linear(in_features=32 * 4 * 4, out_features=32),
            nn.ReLU(True),
            nn.Linear(32, transform.value[0]['params_count'])
        )

        xavier_initialization(self.localization)
        xavier_initialization(self.localization_regress)

        # Initialize the weights/bias with identity transformation
        self.localization_regress[2].weight.data.fill_(0)
        self.localization_regress[2].bias.data = torch.FloatTensor(transform.value[0]['initial'])

    def forward(self, x):

        # Feed to localization network
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        theta = self.localization_regress(xs)
        theta = theta.view(-1, *self.transform_params.value[0]['dims'])

        # Generate parametrized sampling grid
        grid = F.affine_grid(theta=theta, size=x.size())

        # Sample interpolated transformed features
        x = F.grid_sample(input=x, grid=grid)

        return x
