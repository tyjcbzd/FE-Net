from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FAM(nn.Module):
    def __init__(self, in_c, out_c):
        super(FAB, self).__init__()
        # double conv
        self.conv0_1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(out_c)
        self.relu0_1 = nn.ReLU(inplace=True)

        self.conv0_2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(out_c)
        self.relu0_2 = nn.ReLU(inplace=True)

        # shortcut
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): # [2, 3, 256, 256]
        x0_1 = self.conv0_1(x)
        x0_1 = self.bn0_1(x0_1)
        x0_1 = self.relu0_1(x0_1) # [2, 64, 256, 256]

        x0_2 = self.conv0_2(x0_1)
        x0_2 = self.bn0_2(x0_2)
        x0_2 = self.relu0_2(x0_2) # [2, 64, 256, 256]

        x1 = self.conv1(x0_2)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.se(x2)

        x3 = self.conv3(x0_2)
        x3 = self.bn3(x3)
        # x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def skip_con(x1: torch.Tensor, x2: torch.Tensor):

    # x1 = self.up(x1)
    # [N, C, H, W]
    diff_y = x2.size()[2] - x1.size()[2]
    diff_x = x2.size()[3] - x1.size()[3]

    # padding_left, padding_right, padding_top, padding_bottom
    x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2])

    x = torch.cat([x2, x1], dim=1)
    # x = self.conv(x)
    return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class FENet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(FENet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # encoder
        self.encoder1 = FAM(in_channels, base_c) # double conv + se (3,32)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.encoder2 = FAM(base_c, base_c*2)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.encoder3 = FAM(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.encoder4 = FAM(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # transfer double conv
        self.bridge_conv = DoubleConv(base_c * 8, base_c * 8)

        #decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up1 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.decoder1 = FAM(base_c * 16, base_c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up2 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.decoder2 = FAM(base_c * 8, base_c * 2)

        # self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3 = FAM(base_c * 4, base_c)

        # self.up4 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder4 = FAM(base_c * 2, base_c)

        self.out_conv = OutConv(base_c, num_classes)

# x: [2, 3, 256, 256]  mask: [2, 1, 256, 256]
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # encoder
        x1 = self.encoder1(x)
        x1_down = self.pool1(x1)

        x2 = self.encoder2(x1_down)
        x2_down = self.pool2(x2)

        x3 = self.encoder3(x2_down)
        x3_down = self.pool3(x3)

        x4 = self.encoder4(x3_down)
        x4_down = self.pool4(x4)

        x5 = self.bridge_conv(x4_down)

        # decoder
        x6 = self.up1(x5)
        x6 = skip_con(x4, x6)
        x6 = self.decoder1(x6)

        x7 = self.up2(x6)
        x7 = skip_con(x3, x7)
        x7 = self.decoder2(x7)

        x8 = self.up3(x7)
        x8 = skip_con(x2, x8)
        x8 = self.decoder3(x8)

        x9 = self.up4(x8)
        x9 = skip_con(x1, x9)
        x9 = self.decoder4(x9)

        # 一个卷积层用于预测
        logits = self.out_conv(x9)

        # return x
        return logits

if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)).cuda()
    # m = torch.randn((2, 1, 256, 256)).cuda()
    model = FENet(3, 1).cuda()
    y = model(x)
    print(y.shape)
    print(model)
    # print(model)