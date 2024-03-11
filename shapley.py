import torch
import numpy as np
import shap

from pathlib import Path
from train_unet.models import ResUNet
from monai_pipeline.monai_dataset import get_dataset


def get_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the model with specified weights."""
    model = ResUNet(n_classes=1, in_channels=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def model_wrapper(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Wrap the model function for compatibility with SHAP."""
    with torch.no_grad():
        return model(x).detach().cpu().numpy()


def get_shap_explainer(
    model: torch.nn.Module, background: torch.Tensor, device: torch.device
) -> shap.DeepExplainer:
    """Initialize and return the SHAP DeepExplainer."""

    model.to(device)
    model.eval()
    
    # doesn't work -> maybe other explainers? Explainer? KernelExplainer?
    explainer = shap.DeepExplainer(model, background.to(device)) 
    return explainer


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    weights_path = Path(r"G:\snapshots\lo_frac\res-unet-epoch-95.pth")
    dataset_path = Path(r"G:\Hackathon\hires_lofrac_short")

    model = get_model(weights_path, device)
    min_step = 3000

    keys = ["image", "label", "nucleus"]  # , "cluster_id"]

    background_data = get_dataset(
        dataset_path,
        keys,
        "train",
        min_step,
        distance_transform=False,
        augmentation=False,
        cache=False,
    )

    data = []
    for i, x in enumerate(background_data):
        if i >= 2:
            break
        image = x["image"].astype(torch.float32)
        nucleus = x["nucleus"].astype(torch.float32)
        model_input = torch.cat((image, nucleus), dim=0)
        data.append(model_input[None, ...])
    background_data = torch.concat(data, dim=0)
    print(background_data.shape)

    # Initialize SHAP explainer
    # : probably test first with only one of the images  and then all of them
    explainer = get_shap_explainer(model, background_data.to(device), device)

    # Compute SHAP values
    shap_values = explainer.shap_values(background_data.to(device))

    # shap.initjs()

    shap.image_plot(shap_values, -background_data.cpu().numpy())


if __name__ == "__main__":
    main()
