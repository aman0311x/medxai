import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    """
    Standard PyTorch implementation of 3D U-Net for volumetric medical
    image segmentation (e.g. CT/MRI scans).

    Input/output tensors are expected in (B, C, D, H, W) format.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(256, 512))

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(512, 256)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(256, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(128, 64)

        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits
