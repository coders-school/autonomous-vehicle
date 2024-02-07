# Define deeplabv3+ model using resnet50, Atrous convolutions, ASPP modules, Decoder

# Import PyTorch
# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class aspp_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(aspp_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, dilation=dilation_rate, padding=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)