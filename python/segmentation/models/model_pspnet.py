# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class auxiliary_branch(nn.Module):
    """
    Auxiliary branch helps in setting initial values for the residual blocks.
    Auxiliary branch is only used during training and not during inference!
    Auxiliary branch uses similar classifier and loss function of the main branch.
    TotalLoss=α∗(auxloss)+(1−α)∗(mainloss)
    Alpha is a hyperparameter. Authors used `alpha - 0.4*
    """
    def __init__(self, in_channels, num_classes):
        super(auxiliary_branch, self).__init__()
        self.aux = nn.Sequential(
                    nn.Conv2d(in_channels , 256, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(256, num_classes, kernel_size=1)
                )
    def forward(self, x, img_size):
        return F.interpolate(self.aux(x), img_size, mode='bilinear', align_corners=False)
        
        
class pyramid_pooling_module(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sizes):
        super(pyramid_pooling_module, self).__init__()
        
        # create pyramid pooling layers for each level
        self.pyramid_pool_layers = []
        for bin_sz in bin_sizes:
            self.pyramid_pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sz),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.pyramid_pool_layers = nn.ModuleList(self.pyramid_pool_layers)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for layer in self.pyramid_pool_layers:
            out.append(F.interpolate(layer(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)