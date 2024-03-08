from pathlib import Path

import click
import matplotlib.pyplot as plt
import monai
import numpy as np
import seaborn as sns
import torch
from monai import transforms as mt
from monai.networks.layers import Act, Norm
from monai.networks.nets import Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from singlenet import get_dataset
from singlenet.model import SingleNet
from singlenet.transforms.calculate_features import CalculateFeatures
from singlenet.transforms.cell_loader import CellIndexLoaderd, CellLabeld, VTKLoader
from singlenet.transforms.cell_type import CellTyped
from singlenet.transforms.rand_cell_crop import RandCellCropd


@click.command()
@click.option(
    "-d",
    "--data",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("-e", "--epochs", type=int, default=100)
@click.option("-p", "--patch-size", type=int, default=96)
@click.option("-b", "--batch-size", type=int, default=32)
def main(
    data: Path,
    epochs: int = 100,
    patch_size: int = 96,
    batch_size: int = 32
):
    transforms = mt.Compose(
        [
            VTKLoader(keys=["image"]),
            CellIndexLoaderd(keys=["label"]),
            CellLabeld(cluster_key="image_cluster", indices_key="label", output_key="label"),
            RandCellCropd(keys=["image", "label"], crop_size=patch_size),
            CellTyped(keys=["label"]),
            CalculateFeatures(keys=["image"]),
            mt.DeleteItemsd(keys=["image_frame", "image_cluster"]),
            mt.EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            mt.RandRotated(
                keys=["image"],
                range_x=np.pi / 4,
                range_y=np.pi / 4,
                padding_mode="reflect",
                prob=0.5,
            ),
            mt.RandFlipd(keys=["image"], spatial_axis=[0, 1], prob=0.5),
            mt.ToTensord(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    dataset = get_dataset(
        dataset_path=data,
    )[:500]

    dataset = monai.data.CacheDataset(
        data=dataset,
        transform=transforms,
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_set = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_set = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
    )

    loss = torch.nn.BCELoss()

    image_head_outputs = 64
    image_head = Classifier(
        in_shape=(1, patch_size, patch_size),
        classes=image_head_outputs,
        channels=[16, 32, 64, 128],
        strides=[2, 2, 2, 2],
        kernel_size=3,
        num_res_units=2,
        act=Act.RELU,
        norm=Norm.INSTANCE,
        dropout=0.0,
        bias=True,
        last_act=None,
    )

    features = ["cell_volume", "n_first_neighbours", "n_second_neighbours"]
    feature_head_outputs = 64
    feature_head = torch.nn.Sequential(
        torch.nn.Linear(len(features), 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, feature_head_outputs),
        torch.nn.ReLU(),
    )

    model = SingleNet(
        image_head=image_head,
        features_head=feature_head,
        backbone=torch.nn.Sequential(
            torch.nn.Linear(image_head_outputs + feature_head_outputs, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ),
        final_activation=torch.nn.Sigmoid(),
    )

    # model = image_head

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_val_losses = []
    epoch_val_accs = []

    model.to(device)
    model.train()
    for _ in (epoch_bar := tqdm(range(epochs), desc="Epochs", total=epochs, unit="epoch", position=0, leave=True)):
        model.train()

        for data in (batch_bar := tqdm(train_set, desc="Batches", position=1, leave=False, unit="batch")):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            in_feats = torch.stack([data[feat].type(torch.float32) for feat in features], dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs, in_feats)

            train_loss = loss(outputs, labels[..., None])
            train_loss.backward()
            optimizer.step()

            batch_bar.set_postfix(loss=train_loss.item())

        scheduler.step()

        val_losses = []
        val_accs = []

        model.eval()
        for data in (batch_bar := tqdm(val_set, desc="Validation", position=1, leave=False, unit="batch")):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            in_feats = torch.stack([data[feat].type(torch.float32) for feat in features], dim=1).to(device)

            outputs = model(inputs, in_feats)
            val_loss = loss(outputs, labels[..., None])
            val_acc = torch.where(outputs > 0.5, 1, 0).eq(labels).sum() / labels.numel()

            val_losses.append(val_loss.item())
            val_accs.append(val_acc.item())

            batch_bar.set_postfix(val_loss=val_loss.item())

        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        epoch_bar.set_postfix(val_loss=val_loss, val_acc=val_acc, lr=optimizer.param_groups[0]["lr"])

        epoch_val_losses.append(val_loss)
        epoch_val_accs.append(val_acc)

    sns.set_style("whitegrid")
    sns.set_context("notebook")

    sns.lineplot(x=range(epochs), y=epoch_val_losses, label="Validation Loss")
    sns.lineplot(x=range(epochs), y=epoch_val_accs, label="Validation Accuracy")

    sns.despine()
    plt.show()


if __name__ == "__main__":
    main()
