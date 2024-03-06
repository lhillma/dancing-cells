from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as Fn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as tfn

from tqdm import tqdm

from dataset import CPFrameDataset, MFCPFrameDataset
import dataset as ds
from models import PaddedUNet, ResUNet

from matplotlib import pyplot as plt


BATCH_SIZE = 4
BATCH_AVG_SIZE = 8


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5) -> torch.Tensor:
    bce = Fn.binary_cross_entropy_with_logits(pred, target)

    pred = Fn.softmax(pred, dim=1)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
    metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

    return loss


def eval_model(
    model: nn.Module, data_loader: DataLoader, device, bce_weight=0.5
) -> dict[str, float]:
    model.eval()

    metrics = defaultdict(float)

    with torch.no_grad():
        for images, cell_types in tqdm(
            data_loader, total=len(data_loader), desc="Validation"
        ):
            images = images.to(device)
            cell_types = cell_types.to(device)

            output = model(images)
            calc_loss(output, cell_types, metrics, bce_weight=bce_weight)

        for k in metrics.keys():
            metrics[k] /= len(data_loader.dataset)

    return metrics


def print_metrics(metrics: dict[str, float]):
    for k, v in metrics.items():
        print(f"{k}: {v:4f}")


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def plot_predictions(
    model: nn.Module, images: torch.Tensor, cell_types: torch.Tensor, device
) -> None:
    model.eval()
    outputs = model(images)
    for image, output, cell_type in zip(images, outputs, cell_types):
        output = Fn.softmax(output, dim=0)
        output = torch.cat(
            (output, torch.zeros(1, *output.shape[1:]).to(device)), dim=0
        )
        output = output * image[0] if image.shape[0] == 3 else output * image
        output = output.permute(1, 2, 0)
        plt.subplot(1, 2, 1)
        plt.title("Prediction")
        plt.imshow(output.detach().cpu().numpy())

        plt.subplot(1, 2, 2)
        cell_type = torch.cat(
            (cell_type, torch.zeros(1, *cell_type.shape[1:]).to(device)), dim=0
        )
        cell_type = cell_type * image[0] if image.shape[0] == 3 else cell_type * image
        cell_type = cell_type.permute(1, 2, 0)
        plt.title("Ground truth")
        plt.imshow(cell_type.detach().cpu().numpy())
        plt.show()


def transpose_metrics(metrics: list[dict[str, float]]) -> dict[str, np.ndarray]:
    transposed_metrics = {}
    for metric in metrics:
        for k, v in metric.items():
            if k in transposed_metrics:
                transposed_metrics[k] = np.append(transposed_metrics[k], v)
            else:
                transposed_metrics[k] = np.array([v])
    return transposed_metrics


def plot_metrics(
    train_metrics: dict[str, np.ndarray], val_metrics: dict[str, np.ndarray]
):
    n_metrics = len(train_metrics)
    _, axs = plt.subplots(n_metrics, 1, sharex=True)
    for i, (k, v) in enumerate(train_metrics.items()):
        axs[i].plot(v, label="train")
        axs[i].plot(val_metrics[k], label="val")
        axs[i].set_title(k)
        axs[i].legend()
    plt.show()


def save_metrics(
    train_metrics: dict[str, np.ndarray], val_metrics: dict[str, np.ndarray]
):
    np.savez("train_metrics.npz", **train_metrics)
    np.savez("val_metrics.npz", **val_metrics)


def save_dataset(data: MFCPFrameDataset):
    torch.save(data, "dataset.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = CPFrameDataset(
    [
        Path("../../data/hires_hiprop/"),
    ],
    glob="*_[0-7]/**/*.vtk",
    skip=3 * 8,
    step=1,
)
val_data = ds.CPFrameDataset(
    [
        Path("/home/leon/vicsek_datasets/dataset_6_full_images_576x576/"),
    ],
    glob="*_[8-9]/**/*.vtk",
    skip=3 * 2,
    step=1,
)

# train_data, val_data = random_split(data, [train_size, val_size])
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

model = ResUNet(n_classes=2, in_channels=1).to(device)

images, cell_types = next(iter(val_dataloader))
# model = PaddedUNet(n_classes=2).to(device)

images = images.to(device)
cell_types = cell_types.to(device)
images = images.swapaxes(2, 3)
cell_types = cell_types.swapaxes(2, 3)
plot_predictions(model, images, cell_types, device)


metrics = defaultdict(float)
metrics = eval_model(model, val_dataloader, device, bce_weight=0.5)
print("Initial metrics:")
print_metrics(metrics)
print()


# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=10, verbose=True
)

train_metrics: list[dict[str, float]] = []
val_metrics: list[dict[str, float]] = []

for epoch in range(60):
    model.train()
    metrics = defaultdict(float)
    optimizer.zero_grad()
    for batch_idx, (images, cell_types) in enumerate(
        tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    ):
        # Augment data
        h_transform = random.choice([identity, tfn.hflip])
        v_transform = random.choice([identity, tfn.vflip])
        images = images.to(device)
        cell_types = cell_types.to(device)

        images = h_transform(images)
        cell_types = h_transform(cell_types)

        images = v_transform(images)
        cell_types = v_transform(cell_types)

        output = model(images)
        loss = calc_loss(output, cell_types, metrics, bce_weight=0.5)
        loss /= BATCH_AVG_SIZE

        loss.backward()

        if ((batch_idx + 1) % BATCH_AVG_SIZE == 0) or (
            batch_idx + 1 == len(train_dataloader)
        ):
            optimizer.step()
            optimizer.zero_grad()

    for k in metrics.keys():
        metrics[k] /= len(train_dataloader.dataset)

    print(f"Epoch {epoch}")
    print_metrics(metrics)
    print()
    train_metrics.append(metrics)

    metrics = eval_model(model, val_dataloader, device, bce_weight=0.5)
    print("Validation metrics:")
    print_metrics(metrics)
    print()
    val_metrics.append(metrics)

    scheduler.step(metrics["loss"])

    torch.save(model.state_dict(), f"model_{epoch}.pth")

train_metrics_all = transpose_metrics(train_metrics)
val_metrics_all = transpose_metrics(val_metrics)


torch.save(train_dataloader, "train_dataloader.pth")
torch.save(val_dataloader, "val_dataloader.pth")

model.load_state_dict(torch.load("model_19.pth"))

for images, cell_types in val_dataloader:
    images = images.to(device)
    cell_types = cell_types.to(device)
    images = images.swapaxes(2, 3)
    cell_types = cell_types.swapaxes(2, 3)
    plot_predictions(model, images, cell_types, device)
