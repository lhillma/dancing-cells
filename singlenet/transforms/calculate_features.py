import numpy as np
from cell2image.image import crop_cells_by_id, get_cell_neighbour_ids
from monai.transforms import MapTransform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class CalculateFeatures(MapTransform):
    def __init__(
        self,
        keys: list[str],
        cell_id_key: str = "cell_id",
        frame_key: str = "image_frame",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.cell_id_key = cell_id_key
        self.frame_key = frame_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        cell_id = d[self.cell_id_key]
        frame = d[self.frame_key]

        for key in self.key_iterator(data):
            im = d[key]

            # Calculate features
            mask = np.where(frame.cluster_id == cell_id, 1, 0)

            d["cell_volume"] = np.sum(mask)
            d["cell_surface_area"] = np.sum(mask)

            # PCA
            # x = StandardScaler().fit_transform(np.argwhere(mask).astype(float))
            # pca = PCA(n_components=2)
            # x = np.argwhere(x)
            # pca_result = pca.fit_transform(x)

            # d["cell_pca_1"] = pca_result[0]
            # d["cell_pca_2"] = pca_result[1]

            d["n_first_neighbours"] = len(
                get_cell_neighbour_ids(
                    cell_id=cell_id, cell_ids=frame.cluster_id, neighbour_order=1
                )
            )
            d["n_second_neighbours"] = len(
                get_cell_neighbour_ids(
                    cell_id=cell_id, cell_ids=frame.cluster_id, neighbour_order=2
                )
            )

        return d
