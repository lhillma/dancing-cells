from monai.data import DataLoader
from monai_dataset import get_dataset
from pathlib import Path
import torch

dataset = get_dataset(
    Path(r"G:\Hackathon\hires"),
    keys=["image", "label"],
)

# basically the same as torch DataLoader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=2,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for data_dict in dataloader:
    image = data_dict["image"].to(device)
    label = data_dict["label"].to(device)
    break
