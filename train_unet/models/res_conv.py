import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.skip(x)
