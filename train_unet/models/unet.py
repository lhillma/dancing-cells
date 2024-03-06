import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.n_classes = n_classes

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to
        # extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a
        # max-pooling layer, with the exception of the last block which does not
        # include a max-pooling layer.

        # input: 572 x 572 x 1
        self.enc1_1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
        )  # output 570 x 570 x 64
        self.enc1_2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
        )  # output 568 x 568 x 64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 284 x 284 x 64

        # input: 284 x 284 x 64
        self.enc2_1 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
        )  # output 282 x 282 x 128
        self.enc2_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
        )  # output 280 x 280 x 128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 140 x 140 x 128

        # input: 140 x 140 x 128
        self.enc3_1 = nn.Conv2d(
            128,
            256,
            kernel_size=3,
        )  # output 138 x 138 x 256
        self.enc3_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
        )  # output 136 x 136 x 256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 68 x 68 x 256

        # input: 68 x 68 x 256
        self.enc4_1 = nn.Conv2d(256, 512, kernel_size=3)  # output 66 x 66 x 512
        self.enc4_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
        )  # output 64 x 64 x 512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output 32 x 32 x 512

        # input: 32 x 32 x 512
        self.enc5_1 = nn.Conv2d(
            512,
            1024,
            kernel_size=3,
        )  # output 30 x 30 x 1024
        self.enc5_2 = nn.Conv2d(
            1024,
            1024,
            kernel_size=3,
        )  # output 28 x 28 x 1024

        # Decoder
        # The decoder upsamples the feature maps from the encoder to generate a
        # segmentation map. Each block in the decoder consists of an upsampling
        # layer followed by two convolutional layers.

        # input: 28 x 28 x 1024
        self.upconv1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )  # output 56 x 56 x 512
        self.dec1_1 = nn.Conv2d(
            1024,
            512,
            kernel_size=3,
        )  # output 54 x 54 x 512
        self.dec1_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
        )  # output 52 x 52 x 512

        # input: 52 x 52 x 512
        self.upconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )  # output 104 x 104 x 256
        self.dec2_1 = nn.Conv2d(
            512,
            256,
            kernel_size=3,
        )  # output 102 x 102 x 256
        self.dec2_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
        )  # output 100 x 100 x 256

        # input: 100 x 100 x 256
        self.upconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )  # output 200 x 200 x 128
        self.dec3_1 = nn.Conv2d(
            256,
            128,
            kernel_size=3,
        )  # output 198 x 198 x 128
        self.dec3_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
        )  # output 196 x 196 x 128

        # input: 196 x 196 x 128
        self.upconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )  # output 392 x 392 x 64
        self.dec4_1 = nn.Conv2d(
            128,
            64,
            kernel_size=3,
        )  # output 390 x 390 x 64
        self.dec4_2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
        )  # output 388 x 388 x 64

        # Output layer
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder

        # input: 572 x 572 x 1
        xe1_1 = torch.relu(self.enc1_1(x))
        xe1_2 = torch.relu(self.enc1_2(xe1_1))
        xp1 = self.pool1(xe1_2)
        print(xp1.shape)

        # input: 284 x 284 x 64
        xe2_1 = torch.relu(self.enc2_1(xp1))
        xe2_2 = torch.relu(self.enc2_2(xe2_1))
        xp2 = self.pool2(xe2_2)
        print(xp2.shape)

        # input: 140 x 140 x 128
        xe3_1 = torch.relu(self.enc3_1(xp2))
        xe3_2 = torch.relu(self.enc3_2(xe3_1))
        xp3 = self.pool3(xe3_2)
        print(xp3.shape)

        # input: 68 x 68 x 256
        xe4_1 = torch.relu(self.enc4_1(xp3))
        xe4_2 = torch.relu(self.enc4_2(xe4_1))
        xp4 = self.pool4(xe4_2)
        print(xp4.shape)

        # input: 32 x 32 x 512
        xe5_1 = torch.relu(self.enc5_1(xp4))
        xe5_2 = torch.relu(self.enc5_2(xe5_1))
        print(xe5_2.shape)

        # Decoder
        # input: 28 x 28 x 1024
        xd1 = self.upconv1(xe5_2)
        xd1 = torch.cat((xd1, xe4_2[:, :, 4:-4, 4:-4]), dim=1)
        xd1_1 = torch.relu(self.dec1_1(xd1))
        xd1_2 = torch.relu(self.dec1_2(xd1_1))
        print(xd1_2.shape)

        # input: 52 x 52 x 512
        xd2 = self.upconv2(xd1_2)
        xd2 = torch.cat((xd2, xe3_2[:, :, 16:-16, 16:-16]), dim=1)
        xd2_1 = torch.relu(self.dec2_1(xd2))
        xd2_2 = torch.relu(self.dec2_2(xd2_1))
        print(xd2_2.shape)

        # input: 100 x 100 x 256
        xd3 = self.upconv3(xd2_2)
        xd3 = torch.cat((xd3, xe2_2[:, :, 40:-40, 40:-40]), dim=1)
        xd3_1 = torch.relu(self.dec3_1(xd3))
        xd3_2 = torch.relu(self.dec3_2(xd3_1))
        print(xd3_2.shape)

        # input: 196 x 196 x 128
        xd4 = self.upconv4(xd3_2)
        xd4 = torch.cat((xd4, xe1_2[:, :, 88:-88, 88:-88]), dim=1)
        xd4_1 = torch.relu(self.dec4_1(xd4))
        xd4_2 = torch.relu(self.dec4_2(xd4_1))
        print(xd4_2.shape)

        # Output layer
        # input: 388 x 388 x 64
        out = self.output(xd4_2)

        return out
