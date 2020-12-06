import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=None, upsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if activation is not None:
            self.activation = activation
        else:
            self.res_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, bias=False
            )
            self.res_bn = nn.BatchNorm2d(num_features=out_channels)

        self.upsample = upsample

    def forward(self, t):

        res = t
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)

        if self.upsample is not None:
            res = self.res_conv(res)
            res = self.res_bn(res)
            t += res
            t = self.relu(t)
            t = self.upsample(t)
        else:
            t += res
            t = self.activation(t)

        return t


class ColorizeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # make pretrained=True before starting the training process
        resnet18 = models.resnet18(pretrained=False)
        # change first conv layer to accept single channel (grayscale)
        resnet18.conv1.weight = nn.Parameter(
            resnet18.conv1.weight.mean(dim=1).unsqueeze(dim=1))

        # use first 3 layers of ResNet-18 as encoder
        self.encoder = nn.Sequential(
            *list(resnet18.children())[:6]
        )
        self.decoder = nn.Sequential(
            self._make_layer(BasicBlock, 128, 64, nn.ReLU(inplace=True)),
            self._make_layer(BasicBlock, 64, 32, nn.ReLU(inplace=True)),
            self._make_layer(BasicBlock, 32, 2, nn.Sigmoid())
        )

    def _make_layer(self, block, in_channels, out_channels, activation):
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        layers = []
        layers.append(block(in_channels, out_channels, upsample=upsample))
        layers.append(block(out_channels, out_channels, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, t):
        t = self.encoder(t)
        t = self.decoder(t)

        return t
