import torch
from monai.transforms import MapTransform


class CellTyped(MapTransform):
    def __init__(
        self,
        keys: list[str],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        for key in self.key_iterator(data):
            im = d[key]
            shape = im.shape
            center = (s // 2 for s in shape)
            d[key] = torch.tensor(im[*center])

        return d
