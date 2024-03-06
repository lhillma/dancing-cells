import torch
from torch import nn


class PaddedUNet(nn.Module):
    """Padded version of the UNet model that does not require post processing"""

    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        # Encoder

        # input: 576 x 576 x 1
        self.enc1_1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            padding=1,
        )  # output 576 x 576 x 64
        self.enc1_2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1,
        )  # output 576 x 576 x 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 288 x 288 x 64

        # input: 288 x 288 x 64
        self.enc2_1 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
        )  # output 288 x 288 x 128
        self.enc2_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
        )  # output 288 x 288 x 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 144 x 144 x 128

        # input: 144 x 144 x 128
        self.enc3_1 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
            padding=1,
        )  # output 144 x 144 x 256
        self.enc3_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )  # output 144 x 144 x 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 72 x 72 x 256

        # input: 72 x 72 x 256
        self.enc4_1 = nn.Conv2d(
            256,
            512,
            kernel_size=3,
            padding=1,
        )  # output 72 x 72 x 512
        self.enc4_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            padding=1,
        )  # output 72 x 72 x 512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 36 x 36 x 512

        # input: 36 x 36 x 512
        self.enc5_1 = nn.Conv2d(
            512,
            1024,
            kernel_size=3,
            padding=1,
        )  # output 36 x 36 x 1024
        self.enc5_2 = nn.Conv2d(
            1024,
            1024,
            kernel_size=3,
            padding=1,
        )  # output 36 x 36 x 1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )  # output 72 x 72 x 512
        self.dec1_1 = nn.Conv2d(
            1024,
            512,
            kernel_size=3,
            padding=1,
        )  # output 72 x 72 x 512
        self.dec1_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            padding=1,
        )  # output 72 x 72 x 512

        # input: 72 x 72 x 512
        self.upconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )  # output 144 x 144 x 256
        self.dec2_1 = nn.Conv2d(
            512,
            256,
            kernel_size=3,
            padding=1,
        )  # output 144 x 144 x 256
        self.dec2_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )  # output 144 x 144 x 256

        # input: 144 x 144 x 256
        self.upconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )  # output 288 x 288 x 128
        self.dec3_1 = nn.Conv2d(
            256,
            128,
            kernel_size=3,
            padding=1,
        )  # output 288 x 288 x 128
        self.dec3_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1,
        )  # output 288 x 288 x 128

        # input: 288 x 288 x 128
        self.upconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )  # output 576 x 576 x 64
        self.dec4_1 = nn.Conv2d(
            128,
            64,
            kernel_size=3,
            padding=1,
        )  # output 576 x 576 x 64
        self.dec4_2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1,
        )  # output 576 x 576 x 64

        # Output layer
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe1_1 = torch.relu(self.enc1_1(x))
        xe1_2 = torch.relu(self.enc1_2(xe1_1))
        xp1 = self.pool1(xe1_2)

        xe2_1 = torch.relu(self.enc2_1(xp1))
        xe2_2 = torch.relu(self.enc2_2(xe2_1))
        xp2 = self.pool2(xe2_2)

        xe3_1 = torch.relu(self.enc3_1(xp2))
        xe3_2 = torch.relu(self.enc3_2(xe3_1))
        xp3 = self.pool3(xe3_2)

        xe4_1 = torch.relu(self.enc4_1(xp3))
        xe4_2 = torch.relu(self.enc4_2(xe4_1))
        xp4 = self.pool4(xe4_2)

        xe5_1 = torch.relu(self.enc5_1(xp4))
        xe5_2 = torch.relu(self.enc5_2(xe5_1))

        # Decoder
        xd1 = self.upconv1(xe5_2)
        xd1 = torch.cat((xd1, xe4_2), dim=1)
        xd1_1 = torch.relu(self.dec1_1(xd1))
        xd1_2 = torch.relu(self.dec1_2(xd1_1))

        xd2 = self.upconv2(xd1_2)
        xd2 = torch.cat((xd2, xe3_2), dim=1)
        xd2_1 = torch.relu(self.dec2_1(xd2))
        xd2_2 = torch.relu(self.dec2_2(xd2_1))

        xd3 = self.upconv3(xd2_2)
        xd3 = torch.cat((xd3, xe2_2), dim=1)
        xd3_1 = torch.relu(self.dec3_1(xd3))
        xd3_2 = torch.relu(self.dec3_2(xd3_1))

        xd4 = self.upconv4(xd3_2)
        xd4 = torch.cat((xd4, xe1_2), dim=1)
        xd4_1 = torch.relu(self.dec4_1(xd4))
        xd4_2 = torch.relu(self.dec4_2(xd4_1))

        # Output layer
        out = self.output(xd4_2)

        return out
