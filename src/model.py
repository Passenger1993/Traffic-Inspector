"""
Модуль модели: U-Net с энкодером ResNet34 и классификационной головой.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Апсэмплинг с последующим DoubleConv.

    Args:
        bottom_channels (int): число каналов нижнего уровня (x1)
        skip_channels (int): число каналов skip-соединения (x2)
        out_channels (int): число выходных каналов после DoubleConv
        bilinear (bool): использовать билинейную интерполяцию вместо транспонированной свёртки
    """
    def __init__(self, bottom_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # После апсэмплинга число каналов не меняется
            conv_in = bottom_channels + skip_channels
            self.conv = DoubleConv(conv_in, out_channels, conv_in // 2)
        else:
            self.up = nn.ConvTranspose2d(bottom_channels, skip_channels, kernel_size=2, stride=2)
            # После конкатенации будет 2 * skip_channels каналов
            self.conv = DoubleConv(2 * skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Размеры могут не совпадать из-за округления
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net с предобученным энкодером ResNet34 и классификационной головой.
    Вход: изображение (B, 3, H, W)
    Выход: (seg_logits, cls_logits)
        seg_logits: (B, n_classes, H, W)
        cls_logits: (B, n_classes) - логиты для классификации доминирующего класса.
    """
    def __init__(self, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Энкодер: ResNet34 (pretrained)
        resnet = models.resnet34(pretrained=True)
        # Первые слои
        self.inc = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.down1 = resnet.layer1   # 64 -> 64
        self.down2 = resnet.layer2   # 64 -> 128
        self.down3 = resnet.layer3   # 128 -> 256
        self.down4 = resnet.layer4   # 256 -> 512

        # Декодер
        factor = 2 if bilinear else 1
        # up1: bottom 512, skip 256, out = 256 // factor
        self.up1 = Up(512, 256, 256 // factor, bilinear)
        # up2: bottom (256//factor), skip 128, out = 128 // factor
        self.up2 = Up(256 // factor, 128, 128 // factor, bilinear)
        # up3: bottom (128//factor), skip 64, out = 64 // factor
        self.up3 = Up(128 // factor, 64, 64 // factor, bilinear)
        # up4: bottom (64//factor), skip 64, out = 32
        self.up4 = Up(64 // factor, 64, 32, bilinear)

        # Дополнительный апсэмплинг для полного восстановления размера (H/2 -> H)
        if bilinear:
            self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        # Финальная свёртка после апсэмплинга
        self.outc = OutConv(32, n_classes)

        # Классификационная голова: глобальный пулинг из последнего энкодерного слоя
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        # Энкодер
        x1 = self.inc(x)          # shape: (B, 64, H/2, W/2)
        x2 = self.maxpool(x1)     # shape: (B, 64, H/4, W/4)
        x2 = self.down1(x2)       # shape: (B, 64, H/4, W/4)
        x3 = self.down2(x2)       # shape: (B, 128, H/8, W/8)
        x4 = self.down3(x3)       # shape: (B, 256, H/16, W/16)
        x5 = self.down4(x4)       # shape: (B, 512, H/32, W/32)

        # Декодер
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)       # shape: (B, 32, H/2, W/2)
        x = self.up5(x)           # shape: (B, 32, H, W)
        seg_logits = self.outc(x) # shape: (B, n_classes, H, W)

        # Классификация
        cls_logits = self.classifier(x5)

        return seg_logits, cls_logits