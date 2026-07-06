import torch
import torch.nn as nn

from .layers import AttentionGate2d
from .unet import DoubleConv


class AttentionUNet(nn.Module):

    def __init__(self, n_channels: int = 1, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = AttentionGate2d(F_g=256, F_l=256, F_int=128)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate2d(F_g=128, F_l=128, F_int=64)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionGate2d(F_g=64, F_l=64, F_int=32)
        self.conv_up3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

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
