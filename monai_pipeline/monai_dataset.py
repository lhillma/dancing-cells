from transforms import LoadVTKDatad
from pathlib import Path
from monai.transforms import Compose, ToTensord, RandSpatialCropd, DistanceTransformEDTd
from monai.data import Dataset
from typing import Iterable
import matplotlib.pyplot as plt
import torch


def get_dataset(dataset_path: Path, keys: Iterable):
    data_dicts = []
    run_folders = sorted(dataset_path.glob("*"))
    for run_folder in run_folders:
        vtk_files = sorted(run_folder.glob("**/*.vtk"))
        data_dicts.extend(
            [
                {
                    "idx_txt_path": run_folder / "indices.txt",
                    "frame_vtk_path": vtk_file,
                }
                for vtk_file in vtk_files
            ]
        )
    transform = Compose(
        [
            LoadVTKDatad(keys=keys),
            ToTensord(keys=keys),
            RandSpatialCropd(keys=keys, roi_size=[256, 256], random_size=False),
            # RandRotate90d(keys=keys, prob=0.1, max_k=3),
            DistanceTransformEDTd(keys=keys),
        ]
    )
    return Dataset(data_dicts, transform)


def main():
    dataset = get_dataset(
        Path(r"G:\Hackathon\hires"),
        keys=["image", "label"],
    )
    for data in dataset:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(torch.squeeze(data["image"]), cmap="gray")
        ax[1].imshow(torch.squeeze(data["label"]), cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
