from models import UNet2d
from monai.losses import DeepSupervisionLoss
import torch
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet2d(
    nr_input_channels=4,
    channels_list=[32, 64, 128, 256, 512, 512, 512],
    nr_output_classes=1,
    nr_output_scales=-1,  # deep supervision, -1 = all but last, 1 = only last
).to(device)

summary(model, (4, 256, 256))
outputs = model(torch.rand(8, 4, 256, 256).to(device))
for output in outputs:
    print(output.shape)

# deep supervision loss wrapper
loss_fn = DeepSupervisionLoss(loss_fn, weight_mode="exp")
