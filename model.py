import torch.nn as nn
import torch
from spatial_transformer_network import SpatialTransformerNetwork

MAX_NUM_FEATURES = 512  # Maximum depth (k) in Encoder / Decoder / Discriminator convolution block Ck
IMAGE_DIMENSIONS = 3    # RGB


def xavier_initialization(modules):
    for m in modules:
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal(m.weight.data, gain=0.02)
            nn.init.constant(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal(m.weight.data, 1.0, 0.02)
            nn.init.constant(m.bias.data, 0.0)


class FaderNetAutoencoder(nn.Module):

    @staticmethod
    def create_encoder_blocks(num_of_layers):

        in_channels = IMAGE_DIMENSIONS  # Normally RGB
        out_channels = 32

        neural_net = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        in_channels = out_channels
        out_channels = min(out_channels * 2, MAX_NUM_FEATURES)

        for i in range(num_of_layers-1):
            neural_net.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)])

            in_channels = out_channels
            out_channels = min(out_channels * 2, MAX_NUM_FEATURES)

        return nn.Sequential(*neural_net)

    @staticmethod
    def create_decoder_blocks(num_of_layers, attr_dim):

        in_channels = min(2**(4+num_of_layers), MAX_NUM_FEATURES)
        out_channels = min(2**(3+num_of_layers), MAX_NUM_FEATURES)
        neural_net = []

        for i in range(1, num_of_layers):
            neural_net.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels+attr_dim, out_channels=out_channels,
                                   kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            ))

            in_channels = out_channels
            out_channels = IMAGE_DIMENSIONS if (i == num_of_layers-1) else min(2**(3+num_of_layers-i), MAX_NUM_FEATURES)

        neural_net.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels + attr_dim, out_channels=out_channels, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh()
        ))

        return nn.ModuleList(neural_net)

    def __init__(self, num_of_layers, attr_dim, stn_transform=None, gpus_count=1):
        super(FaderNetAutoencoder, self).__init__()

        self.gpus_count = gpus_count
        self.encoder_layers = self.create_encoder_blocks(num_of_layers)
        self.decoder_layers = self.create_decoder_blocks(num_of_layers, attr_dim)
        # xavier_initialization(self.encoder_layers)
        # xavier_initialization(self.decoder_layers)

        self.stn = None
        if stn_transform is not None:
            transform = SpatialTransformerNetwork.Transform.ATTENTION if stn_transform == 'attention' else \
                        SpatialTransformerNetwork.Transform.AFFINE
            self.stn = SpatialTransformerNetwork(in_channels=IMAGE_DIMENSIONS, transform=transform)

    def encode(self, x):
        if self.gpus_count > 1:
            x = nn.parallel.data_parallel(self.stn, x, range(self.gpus_count))
            return nn.parallel.data_parallel(self.encoder_layers, x, range(self.gpus_count))
        else:
            if self.stn is not None:
                x = self.stn(x)
            return self.encoder_layers(x)

    def decode(self, z, y):
        x_decoded = z
        y_reshape = y.unsqueeze(2).unsqueeze(3)
        for layer in self.decoder_layers:
            y_expanded = y_reshape.expand(-1, -1, x_decoded.shape[2], x_decoded.shape[3])
            x_decoded = torch.cat([x_decoded, y_expanded], dim=1)
            x_decoded = layer(x_decoded)

        return x_decoded

    def forward(self, x, y):
        z = self.encode(x)
        x_reconstruct = self.decode(z, y)
        return z, x_reconstruct


class FaderNetDiscriminator(nn.Module):

    def __init__(self, num_of_layers, attr_dim, gpus_count=1):
        super(FaderNetDiscriminator, self).__init__()

        self.gpus_count = gpus_count
        block_size = min(2 ** (3 + num_of_layers), MAX_NUM_FEATURES)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=block_size, out_channels=block_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=block_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),

            nn.Conv2d(in_channels=block_size, out_channels=block_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=block_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3)
        )

        self.proj = nn.Sequential(
            nn.Linear(in_features=block_size, out_features=512, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=attr_dim, bias=True),
            nn.Tanh()
        )

        # xavier_initialization(self.modules())

    def forward(self, z):

        batch_size = z.size()[0]
        y_prediction = self.conv(z)
        y_flat = y_prediction.view(batch_size, -1)

        y_prediction = self.proj(y_flat)

        return y_prediction
