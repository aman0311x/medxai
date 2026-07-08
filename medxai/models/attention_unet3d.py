import torch
import torch.nn as nn

from .layers import AttentionGate3d
from .unet3d import DoubleConv3D


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for volumetric medical image segmentation.
    Mirrors UNet3D's encoder-decoder backbone, with skip connections
    gated by AttentionGate3d — parity counterpart to AttentionUNet (2D).

    Input/output tensors are expected in (B, C, D, H, W) format.
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(256, 512))

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.att1 = AttentionGate3d(F_g=256, F_l=256, F_int=128)
        self.conv_up1 = DoubleConv3D(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate3d(F_g=128, F_l=128, F_int=64)
        self.conv_up2 = DoubleConv3D(256, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionGate3d(F_g=64, F_l=64, F_int=32)
        self.conv_up3 = DoubleConv3D(128, 64)

        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        g1 = self.up1(x4)
        x3_att = self.att1(g=g1, x=x3)
        x = self.conv_up1(torch.cat([g1, x3_att], dim=1))

        g2 = self.up2(x)
        x2_att = self.att2(g=g2, x=x2)
        x = self.conv_up2(torch.cat([g2, x2_att], dim=1))

        g3 = self.up3(x)
        x1_att = self.att3(g=g3, x=x1)
        x = self.conv_up3(torch.cat([g3, x1_att], dim=1))

        logits = self.outc(x)
        return logits
