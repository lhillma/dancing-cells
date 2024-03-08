from pathlib import Path
import torch
from train_unet.models import ResUNet
from torch.utils.data import DataLoader
from monai_pipeline.monai_dataset import get_dataset
from monai.data.utils import no_collation


def get_model(weights_path: Path, device: torch.device):
    model = ResUNet(n_classes=1, in_channels=2).to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def main():
    weights_path = Path(r"G:\snapshots\lo_frac\res-unet-epoch-95.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(weights_path, device)

    dataset_path = Path(r"G:\Hackathon\hires_lofrac_short")
    keys = ["image", "label", "nucleus", "cluster_id"]
    min_step = 3000
    n_workers = 4

    dataset_valid = get_dataset(
        dataset_path,
        keys,
        "valid",
        min_step,
        distance_transform=False,
        cache=False,
    )

    data_loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=n_workers,
        collate_fn=no_collation,  # just give list of dicts
    )

    for data in data_loader_valid:
        data_dict = data[0]
        image = data_dict["image"].float().unsqueeze(0)
        nucleus = data_dict["nucleus"].float().unsqueeze(0)
        model_input = torch.cat((image, nucleus), dim=1).to(device)
        cluster_id = data_dict["cluster_id"].int()
        labels = data_dict["label"].float()
        outputs = model(model_input).squeeze(0).cpu()
        print(outputs.shape)  # torch.Size([1, 400, 400])
        print(labels.shape)  # torch.Size([1, 400, 400])
        print(cluster_id.shape)  # torch.Size([1, 400, 400])

        break


if __name__ == "__main__":
    main()
