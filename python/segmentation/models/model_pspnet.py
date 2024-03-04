# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


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
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
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


class PSPNet(nn.Module):
    def __init__(self, in_channels, num_classes, use_aux=False):
        super(PSPNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # backbone layers
        backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
        self.initial = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Pyramid pooling module components
        ppm_in_channels = int(backbone.fc.in_features)
        self.ppm = pyramid_pooling_module(in_channels=ppm_in_channels,
                                          out_channels=512, bin_sizes=[1, 2, 3, 6])

        # classifier head
        self.cls = nn.Sequential(
            nn.Conv2d(ppm_in_channels * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

        # main branch is composed of PPM + Classifier
        self.main_branch = nn.Sequential(self.ppm, self.cls)

        # Define Auxilary branch if specified
        self.use_aux = False
        if (self.training and use_aux):
            self.use_aux = True
            self.aux_branch = nn.Sequential(
                nn.Conv2d(int(ppm_in_channels / 2), 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, self.num_classes, kernel_size=1)
            )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Pass input through Backbone layers
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        # Get Main branch output
        main_output = self.main_branch(x)
        main_output = F.interpolate(main_output, size=input_size, mode='bilinear')

        # If needed, get auxiliary branch output
        if (self.training and self.use_aux):
            aux_output = F.interpolate(self.aux_branch(x_aux), size=input_size, mode='bilinear')
            output = {}
            output['aux'] = aux_output
            output['main'] = main_output
            return output

        return main_output
