import torch
from torch import nn


from .res_conv import ResidualConv


class ResUNet(nn.Module):
    def __init__(self, n_classes, in_channels: int = 1):
        super().__init__()

        self.n_classes = n_classes

        # Encoder
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        )
        # self.enc1 = ResidualConv(1, 64, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ResidualConv(64, 128, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ResidualConv(128, 256, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ResidualConv(256, 512, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5 = ResidualConv(512, 1024, stride=1, padding=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = ResidualConv(1024, 512, stride=1, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ResidualConv(512, 256, stride=1, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualConv(256, 128, stride=1, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = ResidualConv(128, 64, stride=1, padding=1)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.input_layer(x) + self.input_skip(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool4(enc4)

        enc5 = self.enc5(pool4)

        # Decoder
        up1 = self.up1(enc5)
        dec1 = self.dec1(torch.cat([up1, enc4], dim=1))

        up2 = self.up2(dec1)
        dec2 = self.dec2(torch.cat([up2, enc3], dim=1))

        up3 = self.up3(dec2)
        dec3 = self.dec3(torch.cat([up3, enc2], dim=1))

        up4 = self.up4(dec3)
        dec4 = self.dec4(torch.cat([up4, enc1], dim=1))

        out = self.out(dec4)

        return out
