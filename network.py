import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.upsample = upsample

    def forward(self, t):

        residual = t
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)

        if self.upsample is not None:
            t = self.upsample(t)
        else:
            t += residual
            t = self.relu(t)

        return t


class ColorizeNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet18 = models.resnet18(pretrained=False)
        # change first conv layer to accept single channel (grayscale)
        resnet18.conv1.weight = nn.Parameter(
            resnet18.conv1.weight.mean(dim=1).unsqueeze(dim=1))

        # use first 3 layers of ResNet-18 as encoder
        self.encoder = nn.Sequential(
            *list(resnet18.children())[:6]
        )
        self.decoder = nn.Sequential(
            self._make_layer(BasicBlock, 128, 64),
            self._make_layer(BasicBlock, 64, 32),
            self._make_layer(BasicBlock, 32, 2)
        )

    def _make_layer(self, block, in_channels, out_channels):
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        layers = []
        layers.append(block(in_channels, out_channels, upsample))
        layers.append(block(out_channels, out_channels, None))

        return nn.Sequential(*layers)

    def forward(self, t):
        t = self.encoder(t)
        t = self.decoder(t)

        return t
