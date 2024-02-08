# Define deeplabv3+ model using resnet50, Atrous convolutions, ASPP modules, Decoder

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import ResNet50 as backbone
from torchvision.models import resnet50


class aspp_conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(aspp_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, dilation=dilation_rate, padding=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class aspp_pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(aspp_pool, self).__init__()
        self.pooling_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.pooling_module(x)
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        

class atrous_spatial_pyramid_pooling(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(atrous_spatial_pyramid_pooling, self).__init__()

        layers = nn.ModuleList([])

        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        for rate in dilation_rates:
            layers.append(aspp_conv(in_channels, out_channels, rate))

        layers.append(aspp_pool(in_channels, out_channels))

        self.layers = nn.ModuleList(layers)

        self.project = nn.Sequential(
            nn.Conv2d(len(layers) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        conv_outputs = []
        for mod in self.layers:
            mod_output = mod(x)
            conv_outputs.append(mod_output)

        output = self.project(torch.cat(conv_outputs, dim=1))
        return output
        
        
class deeplabv3_decoder(nn.Module):
    def __init__(self, num_classes):
        super(deeplabv3_decoder, self).__init__()
        self.num_classes = num_classes

        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU())

        self.cls = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, self.num_classes, kernel_size=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_project(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, low_level_feat), dim=1)

        x = self.cls(x)
        return x
        
        
class deeplabv3_plus(nn.Module):
    def __init__(self, in_channels, output_stride, num_classes):
        super(deeplabv3_plus, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_stride = output_stride

        if (output_stride == 16):
            dilation_rates = [6, 12, 18]
            replace_stride_with_dilation = [False, False, True]

        elif (output_stride == 8):
            dilation_rates = [12, 24, 36]
            replace_stride_with_dilation = [False, True, True]

        backbone = resnet50(pretrained=True, replace_stride_with_dilation=replace_stride_with_dilation)
        self.initial = nn.Sequential(*list(backbone.children())[:4])
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        aspp_out_channels = 256
        aspp_in_channels = int(backbone.fc.in_features)
        self.aspp_module = atrous_spatial_pyramid_pooling(aspp_in_channels,
                                                          out_channels=aspp_out_channels, dilation_rates=dilation_rates)

        self.decoder = deeplabv3_decoder(self.num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.initial(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        aspp_output = self.aspp_module(x)
        decoder_output = self.decoder(aspp_output, low_level_feat)
        return F.interpolate(decoder_output, size=input_size, mode='bilinear', align_corners=False)
        