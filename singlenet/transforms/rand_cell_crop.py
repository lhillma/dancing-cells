# from cell2image.image import crop_cell_neighbourhood
from cell2image import image as cimg
from monai.transforms import MapTransform, RandomizableTransform


class RandCellCropd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: list[str],
        crop_size: int = 64,
        frame_key: str = "image_frame",
        cell_id_key: str = "cell_id",
        n_cells: int = 100,
        prob: float = 1.0,
        allow_missing_keys: bool = False,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.crop_size = crop_size
        self.n_cells = n_cells
        self.frame_key = frame_key
        self.cell_id_key = cell_id_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        selected_cell = self.R.randint(0, self.n_cells)
        # Get odd values for selected_cell
        if selected_cell % 2 == 0:
            selected_cell += 1

        d[self.cell_id_key] = selected_cell

        for key in self.key_iterator(data):
            d[key] = cimg.crop_cell_neighbourhood(
                image_in=d[key],  # either image or label
                cell_ids=d[self.frame_key].cluster_id,
                cell_id=selected_cell,
                size=self.crop_size,
            )

        return d
