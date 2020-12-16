""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, freeze=True)
        self.down1 = Down(64, 128, freeze=True)
        self.down2 = Down(128, 256, freeze=True)
        self.down3 = Down(256, 512, freeze=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, freeze=True)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc1 = OutConv(512 // factor, n_classes)
        self.outc2 = OutConv(256 // factor, n_classes)
        self.outc3 = OutConv(128 // factor, n_classes)
        self.outc4 = OutConv(64, n_classes) 

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        z1 = self.outc1(x)
        x = self.up2(x, x3)
        z2 = self.outc2(x)
        x = self.up3(x, x2)
        z3 = self.outc3(x)
        x = self.up4(x, x1)
        z4 = self.outc4(x)
        # logits = self.outc(x)
        return [z1, z2, z3, z4]
