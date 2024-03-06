from monai.transforms import MapTransform
from cell2image import image as cimg
from monai.config import KeysCollection
import numpy as np


class LoadVTKDatad(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        frame_vtk_path_key: str = "frame_vtk_path",
        idx_txt_path_key: str = "idx_txt_path",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.frame_vtk_path_key = frame_vtk_path_key
        self.idx_txt_path_key = idx_txt_path_key

    def __call__(self, data):
        frame = cimg.read_vtk_frame(data[self.frame_vtk_path_key])
        indices = np.loadtxt(data[self.idx_txt_path_key]).astype(np.int32)
        label = cimg.get_motility_labels(frame.cluster_id, indices)

        image = cimg.get_cell_outlines(frame.cluster_id, frame.cluster_id.shape).astype(
            np.float32
        )
        image += np.where(frame.cell_type == 2, 0, 1).astype(np.float32)
        data_dict = {
            "image": image[np.newaxis, ...],
            "label": label[np.newaxis, ...],
            # "step": frame.step,
            # "cell_id": frame.cell_id,
            # "cluster_id": frame.cluster_id,
        }
        return data_dict
