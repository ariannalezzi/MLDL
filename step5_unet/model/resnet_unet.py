import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SResUnet(nn.Module):
    """Shallow U-Net with ResNet18 or ResNet34 as encoder, we use F.interpolate and 1x1 conv for the upsample"""

    def __init__(self, encoder, *, pretrained=False, out_channels=2):
        super().__init__()
        self.encoder = encoder(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())

        # ResNet in blocks
        self.block1 = nn.Sequential(*self.encoder_layers[:3])   # [conv1, bn1, relu] → H/2
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])  # [maxpool, layer1] → H/4
        self.block3 = self.encoder_layers[5]                     # layer2 → H/8
        self.block4 = self.encoder_layers[6]                     # layer3 → H/16
        self.block5 = self.encoder_layers[7]                     # layer4 → H/32

        # Decoder
        self.conv6 = double_conv(512 + 256, 512)
        self.reduce7 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv7 = double_conv(256 + 128, 256)
        self.reduce8 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv8 = double_conv(128 + 64, 128)
        self.reduce9 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv9 = double_conv(64 + 64, 64)
        self.reduce10 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1)

        if not pretrained:
            self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        orig_size = x.size()[2:]  # original dimension (H, W)

        # Encoder
        block1 = self.block1(x)       # [N, 64, H/2,  W/2]
        block2 = self.block2(block1)  # [N, 64, H/4,  W/4]
        block3 = self.block3(block2)  # [N, 128, H/8, W/8]
        block4 = self.block4(block3)  # [N, 256, H/16, W/16]
        block5 = self.block5(block4)  # [N, 512, H/32, W/32]

        # Decoder
        # 1) upsampling block4 dimension
        x = F.interpolate(block5, size=block4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, block4], dim=1)          # [N, 768, H/16, W/16]
        x = self.conv6(x)                          # [N, 512, H/16, W/16]

        # 2) channels reduce: 512->256 and upsampling - block3 dimension
        x = self.reduce7(x)                        # [N, 256, H/16, W/16]
        x = F.interpolate(x, size=block3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, block3], dim=1)          # [N, 384, H/8, W/8]
        x = self.conv7(x)                          # [N, 256, H/8, W/8]

        # 3) channels reduce: 256->128 and upsampling - block2 dimension
        x = self.reduce8(x)                        # [N, 128, H/8, W/8]
        x = F.interpolate(x, size=block2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, block2], dim=1)          # [N, 192, H/4, W/4]
        x = self.conv8(x)                          # [N, 128, H/4, W/4]

        # 4) channels reduce: 128->64 and upsampling - block1 dimension
        x = self.reduce9(x)                        # [N, 64, H/4, W/4]
        x = F.interpolate(x, size=block1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, block1], dim=1)          # [N, 128, H/2, W/2]
        x = self.conv9(x)                          # [N, 64, H/2, W/2]

        # 5) channels reduce: 64->32 and upsampling to original dimension
        x = self.reduce10(x)                       # [N, 32, H/2, W/2]
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=True)  # [N, 32, H, W]
        x = self.conv10(x)                         # [N, out_channels, H, W]

        return x