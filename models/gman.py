import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + res)

class GMAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)

        self.out_conv = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        inp = x

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))

        x = self.res_blocks(x)

        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))

        x = self.out_conv(x)

        return torch.clamp(x + inp, 0, 1)