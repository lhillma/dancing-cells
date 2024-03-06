from typing import Iterable
from functools import partial
import itertools
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import functional as fn
import numpy as np

from numba import njit

# from torch.utils.data.dataloader import multiprocessing

from tqdm import tqdm

from cell2image import image as cimg

from analysis import (
    fit_ellipse,
    cart_to_pol,
    get_cell_boundary,
    get_ellipse_fit_params,
    hsv_to_rgb,
    get_ellipse_image,
    EllipseParams,
)
from cell2image.types import SimulationFrame


class CPFrameDataset(Dataset):
    def __init__(
        self,
        root_dirs: list[Path],
        transform=None,
        device: torch.device | None = None,
        glob="*.vtk",
        skip=3,
        step=1,
    ):
        self.transform = transform.to(device) if transform is not None else None

        if device is None:
            device = torch.device("cpu")
        self.images: list[tuple[Tensor, Tensor]] = []
        vtk_paths = list(
            itertools.chain(
                *map(
                    partial(get_vtk_files_for_folder, glob=glob, skip=skip, step=step),
                    root_dirs,
                )
            )
        )

        self.images = list(self.extract_frames(vtk_paths, device=device))

    def extract_frames(
        self, paths: list[Path], device: torch.device
    ) -> Iterable[tuple[Tensor, Tensor]]:
        for path in tqdm(paths):
            frame = cimg.read_vtk_frame(path)
            np_image = cimg.get_cell_outlines(frame.cluster_id, frame.cell_id.shape)
            np_image += np.where(frame.cell_type == 2, 1, 0).astype(np.uint64)
            image = Tensor(np_image.copy()).to(device)
            cell_type = Tensor((frame.cell_type - 1)).to(device)

            if self.transform is not None:
                image = self.transform(image[None, :, :])[0]
                cell_type = self.transform(cell_type[None, :, :])[0]

            yield (
                image[None, :, :],
                fn.one_hot(cell_type.long()).permute(2, 0, 1).float(),
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.images[idx][:2]


@njit(parallel=True)
def convert_vtk_ellipse_fit(
    image: np.ndarray, cell_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    img_phi = np.zeros(image.shape, dtype=float)
    img_r = np.zeros(image.shape, dtype=float)
    # ellipses: list[EllipseParams] = []

    for id in range(1, cell_id.max() + 1):
        cell_outline = get_cell_boundary(image, cell_id, id)
        coords = np.argwhere(cell_outline)
        x = coords[:, 0]
        y = coords[:, 1]
        width, height = image.shape
        dx = x.max() - x.min()
        dy = y.max() - y.min()

        x[x > width / 2] -= width if dx > width / 2 else 0
        y[y > height / 2] -= height if dy > height / 2 else 0

        ellipse = get_ellipse_fit_params(x, y)

        img_phi += ellipse.phi / np.pi * (cell_id == id).astype(np.float64)
        # img_r += (ellipse.b / ellipse.a) * (cell_id == id).astype(float)
        img_r += ellipse.e * (cell_id == id).astype(np.float64)

        # ellipses.append(ellipse)

    return img_phi, img_r


class MFCPFrameDataset(CPFrameDataset):
    def __init__(
        self,
        root_dirs: list[Path],
        transform=None,
        device: torch.device | None = None,
        glob="*.vtk",
        skip=3,
        step=1,
    ):
        super().__init__(root_dirs, transform, device, glob, skip, step)

    def _extract_frame(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        frame = cimg.read_vtk_frame(path)
        img_phi, img_r = convert_vtk_ellipse_fit(frame.image, frame.cell_id)

        image = np.concatenate(
            (frame.image.copy()[None, :, :], img_phi[None, :, :], img_r[None, :, :])
        )
        cell_type = frame.cell_type - 1

        return image, cell_type

    def extract_frames(
        self, paths: list[Path], device: torch.device
    ) -> Iterable[tuple[Tensor, Tensor]]:
        # with multiprocessing.Pool() as p:
        #     images = list(tqdm(p.imap(self._extract_frame, paths), total=len(paths)))

        images = list(tqdm(map(self._extract_frame, paths), total=len(paths)))

        for image, cell_type in images:
            image_t = Tensor(image.copy()).to(device)
            cell_type_t = Tensor(cell_type.copy()).to(device)

            if self.transform is not None:
                image_t = self.transform(image_t[None, :, :])[0]
                cell_type_t = self.transform(cell_type_t[None, :, :])[0]

            yield (
                image_t,
                fn.one_hot(cell_type_t.long()).permute(2, 0, 1).float(),
            )

        # return (
        #     (torch.Tensor(img[0]).to(device), torch.Tensor(img[1]).to(device))
        #     for img in images
        # )
        # img_ellipse = get_ellipse_image(ellipses, *frame.image.shape)

        # print(img_phi.max(), img_phi.min())
        # print(img_r.max(), img_r.min())

        # rgb_image = torch.cat(
        #     hsv_to_rgb(
        #         Tensor(img_phi).to(device)[None, :, :],
        #         Tensor(img_r).to(device)[None, :, :],
        #         image[None, :, :],
        #     ),
        #     dim=0,
        # )
        # rgb_image = torch.cat(
        #     (
        #         (1 - image[None, :, :]),
        #         Tensor(img_phi).to(device)[None, :, :],
        #         Tensor(img_r).to(device)[None, :, :],
        #     ),
        #     dim=0,
        # )

        # plt.hist(img_phi.flatten(), bins=100)
        # plt.show()
        # plt.hist(img_r.flatten(), bins=100)
        # plt.show()
        # image = Tensor(img_r).to(device)
        # plt.subplot(121)
        # rgb_image *= torch.Tensor(img_ellipse).to(device)
        # plt.imshow(rgb_image.permute(1, 2, 0).detach().cpu().numpy())
        # plt.imshow(image.detach().cpu().numpy())
        # plt.colorbar()
        # cell_type_p = torch.cat(
        #     (
        #         fn.one_hot(cell_type.long()).permute(2, 0, 1).float(),
        #         torch.zeros(1, *cell_type.shape).to(device),
        #     ),
        #     dim=0,
        # )
        # cell_type_p *= image
        # cell_type_p *= image * torch.Tensor(img_ellipse).to(device)
        # plt.subplot(122)
        # plt.imshow(cell_type_p.permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
        # plt.show()

        # if self.transform is not None:
        #     image = self.transform(rgb_image[None, :, :])[0]
        #     cell_type = self.transform(cell_type[None, :, :])[0]

        # return (
        #     image[None, :, :],
        #     fn.one_hot(cell_type.long()).permute(2, 0, 1).float(),
        # )


def get_vtk_files_for_folder(path: Path, glob="*.vtk", skip=3, step=1) -> list[Path]:
    vtk_paths = path.glob(glob)
    vtk_paths = sorted(vtk_paths, key=lambda x: x.stem)[skip::step]
    return vtk_paths
