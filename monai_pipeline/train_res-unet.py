from collections import defaultdict

import click
import numpy as np

from monai.losses.dice import DiceLoss
from torch.nn import BCEWithLogitsLoss

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from models import ResUNet
from monai_dataset import get_dataset


def eval_model(
    model: nn.Module, data_loader: DataLoader, device, bce_weight=0.5, no_nucleus=False
) -> dict[str, float]:
    model.eval()

    metrics = defaultdict(float)

    dice = DiceLoss(sigmoid=True)
    bce = BCEWithLogitsLoss()

    total = 0

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), desc="Validation"):
            image = data["image"].astype(torch.float32)
            nucleus = data["nucleus"].astype(torch.float32)
            labels = data["label"].astype(torch.float32).to(device)
            input = (
                torch.cat((image, nucleus), dim=1).to(device)
                if not no_nucleus
                else image.to(device)
            )

            output = model(input)

            bce_loss = bce(output, labels)
            dice_loss = dice(output, labels)
            loss = (1 - bce_weight) * dice_loss + bce_weight * bce_loss
            metrics["bce"] += bce_loss.detach().cpu().numpy() * labels.size(0)
            metrics["dice"] += dice_loss.detach().cpu().numpy() * labels.size(0)
            metrics["loss"] += loss.detach().cpu().numpy() * labels.size(0)
            total += labels.size(0)

        for k in metrics.keys():
            metrics[k] /= total

    return metrics


def train_model(
    model: nn.Module,
    optimizer,
    data_loader: DataLoader,
    device,
    bce_weight=0.5,
    scheduler=None,
    no_nucleus=False,
) -> dict[str, float]:
    model.train()

    metrics = defaultdict(float)

    dice = DiceLoss(sigmoid=True)
    bce = BCEWithLogitsLoss()

    total = 0

    for data in tqdm(data_loader, total=len(data_loader), desc="Training"):
        image = data["image"].astype(torch.float32)
        nucleus = data["nucleus"].astype(torch.float32)
        labels = data["label"].astype(torch.float32).to(device)
        input = (
            torch.cat((image, nucleus), dim=1).to(device)
            if not no_nucleus
            else image.to(device)
        )

        output = model(input)

        bce_loss = bce(output, labels)
        dice_loss = dice(output, labels)
        loss = (1 - bce_weight) * dice_loss + bce_weight * bce_loss
        metrics["bce"] += bce_loss.detach().cpu().numpy() * labels.size(0)
        metrics["dice"] += dice_loss.detach().cpu().numpy() * labels.size(0)
        metrics["loss"] += loss.detach().cpu().numpy() * labels.size(0)
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    for k in metrics.keys():
        metrics[k] /= total

    return metrics


def print_metrics(metrics: dict[str, float]):
    for k, v in metrics.items():
        print(f"{k}: {v:4f}")


def transpose_metrics(metrics: list[dict[str, float]]) -> dict[str, np.ndarray]:
    transposed_metrics = {}
    for metric in metrics:
        for k, v in metric.items():
            if k in transposed_metrics:
                transposed_metrics[k] = np.append(transposed_metrics[k], v)
            else:
                transposed_metrics[k] = np.array([v])
    return transposed_metrics


def save_metrics(path: Path, metrics: list[dict[str, float]]):
    np.savez(path, **transpose_metrics(metrics))


@click.command()
@click.option("--train-batch-size", default=4)
@click.option("--val-batch-size", default=8)
@click.option("--data-path", default="../../../../data/hires/")
@click.option("--distance-transform", is_flag=True, default=False)
@click.option("--n-workers", default=8)
@click.option("--cache", is_flag=True, default=False)
@click.option("--out-path", default=".")
@click.option("--no-nucleus", is_flag=True, default=False)
def main(
    train_batch_size: int,
    val_batch_size: int,
    data_path: str,
    distance_transform: bool,
    n_workers: int,
    cache: bool,
    out_path: str,
    no_nucleus: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUNet(n_classes=1, in_channels=2 if not no_nucleus else 1).to(device)

    dataset_train = get_dataset(
        Path(data_path),
        keys=["image", "nucleus", "label"],
        min_step=3000,
        distance_transform=distance_transform,
        cache=cache,
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
    )

    dataset_valid = get_dataset(
        Path(data_path),
        keys=["image", "nucleus", "label"],
        min_step=3000,
        dataset_type="valid",
        distance_transform=distance_transform,
        cache=cache,
    )
    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
    )

    # initial_metrics = eval_model(model, data_loader_valid, device)

    # print_metrics(initial_metrics)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, verbose=True
    )

    all_train_metrics = []
    all_valid_metrics = []

    out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(50):
        print(f"Epoch {epoch}")
        train_metrics = train_model(
            model, optimizer, data_loader_train, device, no_nucleus=no_nucleus
        )
        print("Train metrics:")
        print_metrics(train_metrics)
        all_train_metrics.append(train_metrics)

        valid_metrics = eval_model(
            model, data_loader_valid, device, no_nucleus=no_nucleus
        )
        print("Validation metrics:")
        print_metrics(valid_metrics)
        all_valid_metrics.append(valid_metrics)

        scheduler.step(valid_metrics["loss"])

        if epoch % 5 == 0:
            torch.save(model.state_dict(), out_dir / f"res-unet-epoch-{epoch}.pth")

        save_metrics(out_dir / "train_metrics.npz", all_train_metrics)
        save_metrics(out_dir / "valid_metrics.npz", all_valid_metrics)


if __name__ == "__main__":
    main()
