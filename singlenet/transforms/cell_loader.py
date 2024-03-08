from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from cell2image import image as c2i
from monai.transforms import MapTransform


class VTKLoader(MapTransform):
    def __init__(
        self,
        keys: Sequence[str],
        frame_suffix: str = "_frame",
        cluster_suffix: str = "_cluster",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.frame_suffix = frame_suffix
        self.cluster_suffix = cluster_suffix

    def __call__(self, data: dict[str, Path]) -> dict:
        d = dict(data)

        for key in self.key_iterator(data):
            frame = c2i.read_vtk_frame(d[key])
            np_image = c2i.get_cell_outlines(frame.cluster_id, frame.cell_id.shape)
            np_image += np.where(frame.cell_type == 2, 1, 0).astype(np.uint64)

            d[key] = np_image
            d[f"{key}{self.frame_suffix}"] = frame
            d[f"{key}{self.cluster_suffix}"] = frame.cluster_id

        return d


class CellIndexLoaderd(MapTransform):
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        for key in self.key_iterator(data):
            indices = np.loadtxt(d[key], dtype=np.int64)
            d[key] = indices

        return d


class CellLabeld(MapTransform):
    def __init__(
        self, cluster_key: str, indices_key: str, output_key: str, allow_missing_keys: bool = False
    ) -> None:
        super().__init__([cluster_key, indices_key, output_key], allow_missing_keys)

        self.cluster_key = cluster_key
        self.indices_key = indices_key
        self.output_key = output_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        d[self.output_key] = c2i.get_motility_labels(d[self.cluster_key], d[self.indices_key])

        return d
