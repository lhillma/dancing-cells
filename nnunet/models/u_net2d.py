"""A module containing a 2D U-Net implementation for image segmentation.

The U-Net architecture consists of an encoder and a decoder, with skip connections
between corresponding encoder and decoder layers. The encoder consists of a series of
convolutional blocks with max pooling, while the decoder consists of a series of
transposed convolutional blocks with concatenation of the corresponding encoder
features and upsampling.

Classes:
- UNet: A representation for a 2D U-Net.

"""

from typing import List

from .building_blocks import Encoder, Decoder

import torch.nn as nn
import torch


class UNet2d(nn.Module):
    """A representation for a 2D U-Net.

    Args:
    - nr_input_channels (int): The number of channels of the input image.
    - channels_list (List[int]): A list holding the number of channels for each block,
        output encoder/input decoder.
    - nr_output_classes (int): The number of output classes of the segmentation.
    - nr_output_scales (int): The number of output scales to use. Defaults to 1.
    - dropout (float): The dropout probability to use before the output convolutions. Defaults to 0.0.

    Methods:
    - forward(x): Performs the forward pass of the U-Net.
    """

    def __init__(
        self,
        nr_input_channels: int,
        channels_list: List[int],
        nr_output_classes: int,
        nr_output_scales: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(nr_input_channels, channels_list)
        self.decoder = Decoder(
            channels_list[::-1], nr_output_classes, nr_output_scales, dropout
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Performs the forward pass of the U-Net.

        Args:
            x (Tensor): The input to the U-Net (image).

        Returns:
            List[Tensor]: The output of the U-Net, the logits of the predicted mask.

        """
        # reverse order to match upsampling order
        encoder_features = self.encoder(x)[::-1]
        decoder_outputs = self.decoder(encoder_features[0], encoder_features[1:])
        return decoder_outputs
