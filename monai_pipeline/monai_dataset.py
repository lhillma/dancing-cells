import torch
from transforms import LoadVTKDatad
from pathlib import Path
from monai.transforms import (
    Compose,
    ToTensord,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    CenterSpatialCropd,
    DistanceTransformEDTd,
)
from monai.data import Dataset
from typing import Iterable
from natsort import natsorted

# import matplotlib.pyplot as plt
# import torch


def get_dataset(
    dataset_path: Path, keys: Iterable, dataset_type: str = "train", min_step: int = 0
):
    data_dicts = []
    run_folders = natsorted(dataset_path.glob("*"))
    if dataset_type == "train":
        run_folders = run_folders[:30]
    else:
        run_folders = run_folders[30:]
    for run_folder in run_folders:
        vtk_files = sorted(run_folder.glob("**/*.vtk"))
        data_dicts.extend(
            [
                {
                    "idx_txt_path": run_folder / "indices.txt",
                    "frame_vtk_path": vtk_file,
                }
                for vtk_file in vtk_files
                if int(vtk_file.stem.split("_")[-1]) >= min_step
            ]
        )
    transforms = [
        LoadVTKDatad(keys=keys),
        ToTensord(keys=keys, dtype=torch.float32),
    ]

    if dataset_type == "train":
        transforms.extend(
            [
                RandSpatialCropd(keys=keys, roi_size=[256, 256], random_size=False),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.75, max_k=3),
            ]
        )
    else:
        transforms.append(CenterSpatialCropd(keys=keys, roi_size=[256, 256]))

    transforms.append(DistanceTransformEDTd(keys=["image", "nucleus"]))

    return Dataset(data_dicts, Compose(transforms))


def main():
    # dataset_path = Path(r"../../data/hires_hiprop/")
    dataset_path = Path(r"G:\Hackathon\hires")
    keys = ["image", "label", "nucleus"]
    min_step = 3000

    train_dataset = get_dataset(dataset_path, keys, "train", min_step)
    valid_dataset = get_dataset(dataset_path, keys, "valid", min_step)
    for dataset in [train_dataset, valid_dataset]:
        for i, data in enumerate(dataset):
            print(data["image"].shape)
            if i > 10:
                break
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(torch.squeeze(data["image"]), cmap="gray")
            # ax[1].imshow(torch.squeeze(data["nucleus"]), cmap="gray")
            # ax[2].imshow(torch.squeeze(data["label"]), cmap="gray")
            # plt.show()


if __name__ == "__main__":
    main()
