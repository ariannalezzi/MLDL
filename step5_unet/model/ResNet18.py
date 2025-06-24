import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)

        # Encoder
        self.conv1  = backbone.conv1
        self.bn1    = backbone.bn1
        self.relu   = backbone.relu
        self.maxpool= backbone.maxpool
        self.layer1 = backbone.layer1   # 1/4
        self.layer2 = backbone.layer2   # 1/8
        self.layer3 = backbone.layer3   # 1/16
        self.layer4 = backbone.layer4   # 1/32
        
        # 1×1 conv to align channels in skip connections
        self.score32 = nn.Conv2d(512, num_classes, 1)   # layer4
        self.score16 = nn.Conv2d(256, num_classes, 1)   # layer3
        self.score8  = nn.Conv2d(128, num_classes, 1)   # layer2

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f4 = self.layer1(x)      # 1/4
        f8 = self.layer2(f4)     # 1/8
        f16 = self.layer3(f8)    # 1/16
        f32 = self.layer4(f16)   # 1/32

        # Decoder with bilinear upsample
        s32 = self.score32(f32)                             # 1/32
        s32_up = F.interpolate(s32, size=f16.shape[2:], mode='bilinear', align_corners=False)
        s16 = s32_up + self.score16(f16)                    # 1/16

        s16_up = F.interpolate(s16, size=f8.shape[2:], mode='bilinear', align_corners=False)
        s8 = s16_up + self.score8(f8)                       # 1/8

        out = F.interpolate(s8, size=(h, w), mode='bilinear', align_corners=False)  # final: 1/1
        return out                  # (N, num_classes, H, W)
