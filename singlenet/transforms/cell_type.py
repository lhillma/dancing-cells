from cell2image.image import get_motility_labels
from monai.transforms import MapTransform


class CellTyped(MapTransform):
    def __init__(
        self,
        frame_key: str = "image_frame",
        cluster_id_key: str = "image_cluster",
        output_key: str = "label",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([frame_key, cluster_id_key, output_key], allow_missing_keys)

        self.frame_key = frame_key
        self.cluster_id_key = cluster_id_key
        self.output_key = output_key

    def __call__(self, data: dict) -> dict:
        d = dict(data)

        frame = d[self.frame_key]
        cluster_id = d[self.cluster_id_key]

        d[self.output_key] = get_motility_labels(frame, cluster_id)

        return d
